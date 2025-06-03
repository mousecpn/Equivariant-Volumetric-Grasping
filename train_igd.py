import argparse
from pathlib import Path
from datetime import datetime

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Average, Accuracy, Precision, Recall
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
import torch
from torch.utils import tensorboard
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from dataset.dataset_voxel import DatasetVoxelOccFile
from model.igd import IGD


LOSS_KEYS = ['loss_all', 'loss_qual', 'loss_rot', 'loss_width', 'loss_occ']

def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.num_workers, "pin_memory": True,"multiprocessing_context": "fork" if args.num_workers>0 else None} if use_cuda else {}

    # create log directory
    if args.savedir == '':
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        description = "{}_dataset={},augment={},net={},batch_size={},lr={:.0e},{}".format(
            time_stamp,
            args.dataset.name,
            args.augment,
            args.net,
            args.batch_size,
            args.lr,
            args.description,
        ).strip(",")
        logdir = args.logdir / description
    else:
        logdir = Path(args.savedir)

    # create data loaders
    train_loader, val_loader = create_train_val_loaders(
        args.dataset, args.dataset_raw, args.batch_size, args.val_split, args.augment, kwargs)
    
    # build the network or load
    net = IGD().to(device)
    if args.load_path != '':
        net.load_state_dict(torch.load(args.load_path, map_location=device))

    # define optimizer and metrics
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    metrics = {
        "accuracy": Accuracy(lambda out: (torch.round(out[1][0]), out[2][0])),
        "ap": AveragePrecision(lambda out: (out[1][0], out[2][0])),
        "recall": Recall(lambda out: (torch.round(out[1][0]), out[2][0])),
        "roc_auc": ROC_AUC(lambda out: (out[1][0], out[2][0])),
    }
    for k in LOSS_KEYS:
        metrics[k] = Average(lambda out, sk=k: out[3][sk])

    # create ignite engines for training and validation
    trainer = create_trainer(net, optimizer, metrics, device)
    evaluator = create_evaluator(net, metrics, device)


    # log training progress to the terminal and tensorboard
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(trainer)
    ProgressBar(persist=True, ascii=True, dynamic_ncols=True, disable=args.silence).attach(evaluator)

    train_writer, val_writer = create_summary_writers(net, device, logdir)
    
    step_size = [int(8*(args.epochs/12)), int(11*(args.epochs/12))]
    lr_scheduler = StepLR(optimizer, step_size=step_size[0], gamma=0.1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_results(engine):
        epoch, metrics = trainer.state.epoch, trainer.state.metrics
        for k, v in metrics.items():
            train_writer.add_scalar(k, v, epoch)

        msg = 'Train'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        epoch, metrics = trainer.state.epoch, evaluator.state.metrics
        
        if epoch > step_size[0]:
            lr_scheduler.step_size = step_size[1]
        lr_scheduler.step()
        print('current lr:{}'.format(optimizer.param_groups[0]['lr']))
        
        for k, v in metrics.items():
            val_writer.add_scalar(k, v, epoch)
            
        msg = 'Val'
        for k, v in metrics.items():
            msg += f' {k}: {v:.4f}'
        print(msg)
        

    def default_score_fn(engine):
        score = engine.state.metrics['ap']
        return score

    # checkpoint model
    checkpoint_handler = ModelCheckpoint(
        logdir,
        "igd",
        n_saved=1,
        require_empty=True,
    )
    best_checkpoint_handler = ModelCheckpoint(
        logdir,
        "best_igd",
        n_saved=1,
        score_name="val_ap",
        score_function=default_score_fn,
        require_empty=True,
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {args.net: net}
    )
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, best_checkpoint_handler, {args.net: net}
    )

    # run the training loop
    trainer.run(train_loader, max_epochs=args.epochs)


def create_train_val_loaders(root, root_raw, batch_size, val_split, augment, kwargs):
    # load the dataset

    dataset = DatasetVoxelOccFile(root, root_raw)
    # split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    # create loaders for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs
    )
    return train_loader, val_loader


def prepare_batch(batch, device):
    pc, (label, rotations, width), pos, pos_occ, occ_value = batch
    pc = pc.float().to(device)
    label = label.float().to(device)
    rotations = rotations.float().to(device)
    width = width.float().to(device)
    pos.unsqueeze_(1) # B, 1, 3
    pos = pos.float().to(device)
    pos_occ = pos_occ.float().to(device)
    occ_value = occ_value.float().to(device)
    return pc, (label, rotations, width, occ_value), pos, pos_occ


def select(out):
    qual_out, rot_out, width_out, occ = out
    rot_out = rot_out.squeeze(1)
    # qual_out = qual_out.sigmoid()
    # occ = torch.sigmoid(occ) # to probability
    return qual_out.squeeze(-1), rot_out, width_out.squeeze(-1), occ




def create_trainer(net, optimizer, metrics, device):
    def _update(_, batch):
        net.train()
        optimizer.zero_grad()

        # forward
        x, y, pos, pos_occ = prepare_batch(batch, device)

        loss, loss_dict, y_pred = net.compute_loss(x, pos, p_tsdf=pos_occ, target=y)

        
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    p.grad.zero_()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step() 

        return x, y_pred, y, loss_dict
    
    trainer = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def create_evaluator(net, metrics, device):
    def _inference(_, batch):
        net.eval()
        with torch.no_grad():
            x, y, pos, pos_occ = prepare_batch(batch, device)
            loss, loss_dict, y_pred = net.compute_loss(x, pos, p_tsdf=pos_occ, target=y, validate=True)
        return x, y_pred, y, loss_dict

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def create_summary_writers(net, device, log_dir):
    train_path = log_dir / "train"
    val_path = log_dir / "validation"

    train_writer = tensorboard.SummaryWriter(train_path, flush_secs=60)
    val_writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    return train_writer, val_writer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", default="igd")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset_raw", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, default="data/igd_runs")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--savedir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--silence", action="store_true")
    parser.add_argument("--load-path", type=str, default='')
    args = parser.parse_args()
    print(args)
    main(args)
