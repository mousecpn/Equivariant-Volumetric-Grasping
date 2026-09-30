import torch
import torch.nn as nn
import sys
import os
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from model.backbone import CyclicUnet3d, Tri_UNet
from model.triunet import JointTriUNet
from model.triunet_induced import JointTriUNetInduced, InducedBilinearSampler
from model.decoder import BilinearSampler, TriFullEquiDecoder, TriTimeEquiDecoder, TimeLocalDecoder
import escnn.nn as enn
from escnn.group import CyclicGroup
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space, rot2dOnR2
import torch.nn.functional as F
from utils.transform import matrix_to_quaternion,quaternion_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d, irreps2rot, rot2irreps, padding
import time
from scipy import stats
from utils.transform import Rotation, Transform
import numpy as np
from model.diffusion import Diffusion,FlowMatching
from model.attention import EquiDeformableAttn, EquiGraspSO3DeformableAttn, EquiDeformableAttn2,EquiGraspSO3DeformableAttn2
from utils.loss import *


def gaussian_filter_3d(values, spatial_shape, sigma=1.0, truncate=4.0):
    if sigma <= 0:
        return values

    radius = int(truncate * sigma + 0.5)
    coordinates = torch.arange(
        -radius, radius + 1, device=values.device, dtype=values.dtype
    )
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    volume = values.reshape(values.shape[0], 1, *spatial_shape)
    for dimension in range(3):
        kernel_shape = [1, 1, 1]
        kernel_shape[dimension] = kernel.numel()
        axis_kernel = kernel.reshape(1, 1, *kernel_shape)
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - dimension)] = radius
        padding[2 * (2 - dimension) + 1] = radius
        volume = F.conv3d(F.pad(volume, padding, mode="replicate"), axis_kernel)
    return volume.reshape_as(values)


class TriEquiIGD(nn.Module):
    def __init__(self, N=4):
        super(TriEquiIGD, self).__init__()
        self._init = 'he'
        self.N = N
        hidden_dim = 36 if self.N % 3 == 0 else 32
        self.gs = no_base_space(CyclicGroup(self.N))
        self.backbone = JointTriUNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N, plane_resolution=40)
        # self.backbone = Tri_UNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N)
        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr]*(hidden_dim//self.N) + [self.gs.trivial_repr]*hidden_dim*2)

        decoder_hidden_dim = hidden_dim * 3
        mixed_repr = [self.gs.regular_repr] + [self.gs.trivial_repr]*8
        hidden_type = FieldType(self.gs, mixed_repr*(decoder_hidden_dim//(8+self.N)))

        irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
        self.irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)]*3)

        self.decoder_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64, feature_sampler=EquiDeformableAttn2(decoder_in_type, N=self.N))
        # self.decoder_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64, feature_sampler=BilinearSampler())
        self.decoder_grasp_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0,  N=self.N, hidden_size=64, feature_sampler=EquiGraspSO3DeformableAttn(decoder_in_type, hidden_type, N=self.N))
        # self.decoder_grasp_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0,  N=self.N, hidden_size=64, concat_type=self.irrep_3x2, feature_sampler=BilinearSampler())
        self.decoder_width = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64, feature_sampler=BilinearSampler())
        # self.decoder_rot = TriTimeEquiDecoder(in_type=decoder_in_type, out_type=self.irrep_3x2, N=self.N, hidden_size=hidden_type, feature_sampler=EquiDeformableAttn(decoder_in_type, hidden_type))
        self.decoder_rot = TriTimeEquiDecoder(in_type=decoder_in_type, out_type=self.irrep_3x2, N=self.N, hidden_size=hidden_type)
        self.decoder_rot.feature_sampler = EquiDeformableAttn2(decoder_in_type, N=self.N) #, concat_type=(self.decoder_rot.out_type+self.decoder_rot.t_type)

        # self.decoder_rot = TriTimeEquiDecoder(in_type=decoder_in_type, out_type=self.irrep_3x2, N=self.N, hidden_size=hidden_type, feature_sampler=BilinearSampler())
        # self.decoder_rot = TimeLocalDecoder(dim=6, out_dim=6, c_dim=96, hidden_size=128, feature_sampler=BilinearSampler())
        self.decoder_tsdf = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64, feature_sampler=BilinearSampler())
        # self.grasp_sampler = Diffusion(schedulers="DDPM", condition_mask=[0,0,0,0,0,0],  prediction_type='epsilon', beta_schedule='linear', num_inference_steps=10)
        self.grasp_sampler = FlowMatching(condition_mask=[0,0,0,0,0,0], denosing_steps=10)


    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        if target is None:
            return self.inference(inputs, p, sample_rounds = 1)


        label, rot_gt, width_gt, occ_value = target
        rot_gt_6d = matrix_to_rotation_6d(quaternion_to_matrix(rot_gt[:, np.random.randint(0,2), :].unsqueeze(1)))
        rot_irreps = rot2irreps(rot_gt_6d)
        grasp = torch.cat((p, rot_irreps), dim=-1)


        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)
        # grasp_qual = self.decoder_grasp_qual(grasp, c)
        width = self.decoder_width(p, c)
        noise = torch.rand((self.batch_size, self.sample_num, len(self.grasp_sampler.condition_mask)), device=p.device)
        irreps = self.grasp_sampler.sample_data(noise, self.decoder_rot.query_feature(p, c), self.decoder_rot)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))

        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width



    @torch.no_grad()
    def inference(self, inputs, p, sample_rounds=1, low_th=0.05,
                  gaussian_filter_sigma=1.0, **kwargs):
        assert self.batch_size==1, "batch size should be 1 in this mode"
        # visualize_tsdf(inputs.cpu().numpy()[0])
        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c).sigmoid()
        width = self.decoder_width(p, c)
        dim = len(self.grasp_sampler.condition_mask)

        spatial_shape = inputs.shape[-3:]
        if qual.numel() == self.batch_size * np.prod(spatial_shape):
            qual = gaussian_filter_3d(
                qual, spatial_shape, sigma=gaussian_filter_sigma
            )

        mask = (qual > low_th)
        if not mask.any():
            mask.reshape(-1)[qual.reshape(-1).argmax()] = True

        p_postive = p[mask].reshape(1, -1, 3)
        feature_positive = self.decoder_rot.query_feature(p_postive, c)

        # loop mode
        for i in range(sample_rounds):
            noise = torch.randn((self.batch_size, p_postive.shape[1], dim), device=p.device) #* (1-self.backbone.test_eq_flag)
            rot_6d = irreps2rot(self.grasp_sampler.sample_data(noise, feature_positive, self.decoder_rot))
            grasp = torch.cat((p_postive, rot_6d), dim=-1)
            grasp_qual = self.decoder_grasp_qual(grasp, c)
            if i == 0:
                last_rot = rot_6d
                last_grasp_qual = grasp_qual
                continue
            comparing_grasp_qual = torch.stack((last_grasp_qual, grasp_qual), dim=1) # (bs, 2, pos_ns)
            comparing_rot = torch.stack((last_rot, rot_6d), dim=1) # (bs, 2, pos_ns, 7)
            last_grasp_qual, indices = torch.max(comparing_grasp_qual, dim=1)
            indices = indices.reshape(self.batch_size, 1, -1, 1).repeat(1,1,1,dim)
            last_rot = comparing_rot.reshape(self.batch_size, 2, -1, dim).gather(1, indices)
            last_rot = last_rot.squeeze(1)

        rot = torch.randn(self.batch_size, p.shape[1], 4).to(last_rot.device)

        rot[mask] = matrix_to_quaternion(rotation_6d_to_matrix(last_rot.reshape(-1,dim)))


        grasp_qual = torch.zeros_like(qual)
        grasp_qual[mask] = torch.sigmoid(last_grasp_qual.reshape(-1))

        qual = (qual*grasp_qual).sqrt()

        return qual, rot, width


    def compute_loss(self, inputs, p, target, p_tsdf, validate=False):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        label, rot_gt, width_gt, occ_value = target

        c = self.backbone(inputs)
        ##### debug ######
        qual = self.decoder_qual(p, c)
        neg_p, neg_label = negative_sampling(p, None, rotation=False)
        neg_qual = self.decoder_qual(neg_p, c)

        all_qual = torch.cat((qual.flatten(), neg_qual.flatten()), dim=0)
        all_label = torch.cat((label.flatten(), neg_label.flatten()), dim=0)
        loss_qual = sigmoid_focal_loss(all_qual, all_label,  sigmoid=False, reduction='mean', smoothing=0.1)

        ##### debug ######
        width = self.decoder_width(p, c)
        tsdf = self.decoder_tsdf(p_tsdf, c)
        rot_gt_6d = matrix_to_rotation_6d(quaternion_to_matrix(rot_gt))
        rot_irreps = rot2irreps(rot_gt_6d)

        ##### debug ######
        grasp = torch.cat((p, rot_gt_6d[:,np.random.randint(0,2),:].unsqueeze(1)), dim=-1)
        grasp_qual = self.decoder_grasp_qual(grasp, c)
        neg_grasp, neg_label = negative_sampling(grasp, None)

        grasp_qual_neg = self.decoder_grasp_qual(neg_grasp, c)

        all_grasp_qual = torch.cat((grasp_qual.flatten(), grasp_qual_neg.flatten()), dim=0)
        all_grasp_label = torch.cat((label.flatten(), neg_label.flatten()), dim=0)

        loss_grasp_qual = sigmoid_focal_loss(all_grasp_qual, all_grasp_label, sigmoid=False, reduction='mean', smoothing=0.1)

        ##### debug ######
        if validate is False:
            if label.sum() > 0:
                model_input = {
                    'data': rot_irreps[label==1],
                    'context': self.decoder_rot.query_feature(p, c)[label==1].repeat(1, rot_irreps.shape[1], 1)
                }
                loss_rot = self.grasp_sampler.loss_fn(self.decoder_rot, model_input).mean()
            else:
                loss_rot = torch.zeros(1).sum()
        else:
            if label.sum() > 0:
                context = self.decoder_rot.query_feature(p, c)[label==1]
                noise = torch.rand((context.shape[0], 1, len(self.grasp_sampler.condition_mask)), device=p.device)
                irreps = self.grasp_sampler.sample_data(noise, context, self.decoder_rot)
                rot = irreps2rot(irreps)
                rot = matrix_to_quaternion(rotation_6d_to_matrix(rot)).squeeze(1)
                loss_rot = rot_loss_fn(rot, rot_gt[label==1]).mean()
            else:
                loss_rot = torch.zeros((), device=p.device)

        # loss_qual = qual_loss_fn(qual.squeeze(1), label, 0.1).mean() + loss_grasp_qual
        # loss_grasp_qual = _qual_loss_fn(grasp_qual.squeeze(1), label).mean()
        loss_width = (
            width_loss_fn(width[label==1].reshape(-1), width_gt[label==1]).mean()
            if label.sum() > 0 else torch.zeros((), device=p.device)
        )
        loss_occ = occ_loss_fn(tsdf, occ_value, 0.1).mean()

        loss = loss_qual +  loss_rot + 0.01 * loss_width + loss_occ + loss_grasp_qual
        loss_dict = {'loss_qual': loss_qual,
                    'loss_rot': loss_rot,
                    'loss_width': loss_width,
                    'loss_occ': loss_occ,
                    'loss_grasp': loss_grasp_qual,
                    'loss_all': loss}

        y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_gt[:,:1], width, tsdf)
        # y_pred = (qual.sigmoid(), rot_gt[:,:1], width, tsdf)

        return loss, loss_dict, y_pred

    def scene_compute_loss(self, inputs, p, target=None, p_tsdf=None, validate=False):
        """
        p (num_grasps, 3)
        inputs (bs, 1, 40, 40, 40)
        """
        self.batch_size = inputs.size(0)
        label, rot_gt, width_gt, num_grasps, occ_value = target

        c = self.backbone(inputs)
        padded_p = padding(p.squeeze(1), num_grasps=num_grasps)
        valid_mask = ~torch.isnan(padded_p).all(-1)


        qual = self.decoder_qual(padded_p, c)[valid_mask]
        width = self.decoder_width(padded_p, c)[valid_mask]
        tsdf = self.decoder_tsdf(p_tsdf, c)
        rot_gt_6d = matrix_to_rotation_6d(quaternion_to_matrix(rot_gt))
        rot_irreps = rot2irreps(rot_gt_6d)

        ##### debug ######
        idx = torch.randint(0, 2, (rot_gt_6d.shape[0], 1, 1), dtype=torch.long).to(p.device)
        rot_gt_6d = torch.gather(rot_gt_6d, 1, idx.expand(-1,-1,rot_gt_6d.shape[-1])).squeeze(1)
        grasp = torch.cat((p,rot_gt_6d ), dim=-1)

        padded_rot_gt_6d = padding(rot_gt_6d, num_grasps=num_grasps)

        padded_grasp = torch.cat((padded_p, padded_rot_gt_6d), dim=-1)
        grasp_qual = self.decoder_grasp_qual(padded_grasp, c)[valid_mask]

        neg_grasp, neg_label = negative_sampling(grasp.unsqueeze(1), None)
        neg_number = neg_label.shape[-1]
        padded_neg_grasp = padding(neg_grasp, num_grasps=num_grasps).reshape(self.batch_size, -1, neg_grasp.shape[-1])

        grasp_qual_neg = self.decoder_grasp_qual(padded_neg_grasp, c).reshape(self.batch_size, -1, neg_number)[valid_mask]

        all_grasp_qual = torch.cat((grasp_qual, grasp_qual_neg.flatten()), dim=0)
        all_grasp_label = torch.cat((label, neg_label.flatten()), dim=0)

        loss_grasp_qual = sigmoid_focal_loss(all_grasp_qual, all_grasp_label, sigmoid=False, reduction='mean')

        ##### debug ######

        #### rotation diffusion ####
        if validate is False:
            if label.sum() > 0:
                context = self.decoder_rot.query_feature(padded_p, c)[valid_mask][label==1].unsqueeze(1).repeat(1, rot_irreps.shape[1], 1)
                model_input = {
                    'data': rot_irreps[label==1],
                    'context': context
                }
                loss_rot = self.grasp_sampler.loss_fn(self.decoder_rot, model_input).mean()
            else:
                loss_rot = torch.zeros(1).sum()
        else:
            if label.sum() > 0:
                context = self.decoder_rot.query_feature(padded_p, c)[valid_mask][label==1].unsqueeze(1)
                noise = torch.rand((context.shape[0], 1, len(self.grasp_sampler.condition_mask)), device=p.device)
                irreps = self.grasp_sampler.sample_data(noise, context, self.decoder_rot)
                rot = irreps2rot(irreps)
                rot = matrix_to_quaternion(rotation_6d_to_matrix(rot)).squeeze(1)
                loss_rot = rot_loss_fn(rot, rot_gt[label==1]).mean()
            else:
                loss_rot = torch.zeros((), device=p.device)

        loss_qual = qual_loss_fn(qual, label, 0.1).mean() + loss_grasp_qual # + qual_loss_fn(grasp_qual, label).mean() #
        # loss_grasp_qual = _qual_loss_fn(grasp_qual.squeeze(1), label).mean()
        loss_width = (
            width_loss_fn(width[label==1].reshape(-1), width_gt[label==1]).mean()
            if label.sum() > 0 else torch.zeros((), device=p.device)
        )
        loss_occ = occ_loss_fn(tsdf, occ_value, 0.1).mean()

        loss = loss_qual +  loss_rot + 0.01 * loss_width + loss_occ
        loss_dict = {'loss_qual': loss_qual,
                    'loss_rot': loss_rot,
                    'loss_width': loss_width,
                    'loss_occ': loss_occ,
                    'loss_all': loss}

        # y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_gt[:,:1], width, tsdf)
        if validate:
            y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot, width, tsdf)
        else:
            y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_gt[:,:1], width, tsdf)

        return loss, loss_dict, y_pred

    def export(self, ):
        self.backbone = self.backbone.export()
        self.decoder_grasp_qual = self.decoder_grasp_qual.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        self.decoder_width = self.decoder_width.export()
        return self



    def test_eq(self,):
        p = torch.rand(1, 1, 3)-0.5
        input_ = torch.randn(1, 1, 40, 40, 40)
        self.backbone.test_eq_flag = True

        for g in self.gs.testing_elements:
            # if g.value == 0:
            #     continue
            input = self.backbone.voxel2repr(input_)
            input = GeometricTensor(input, self.backbone.in_type)
            input_transformed = input.transform(g)

            rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value*np.pi / 2.0])
            rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)

            p_trans = p.clone()
            p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1,3))).reshape(1,1,3)
            p_trans = torch.tensor(p_trans, dtype=torch.float32)

            qual, rot, width = self.forward(input, p)
            rot_6d = matrix_to_rotation_6d(quaternion_to_matrix(rot))

            qual_trans, rot_trans, width_trans = self.forward(input_transformed, p_trans)
            rot_6d_trans = matrix_to_rotation_6d(quaternion_to_matrix(rot_trans))
            # print('rot:', rot_6d_trans.reshape(2,3))
            rot_6d_trans_after = (rot_mat @ rot_6d.reshape(2,3)).reshape(1,1,6)
            # print('rot_after:', rot_6d_trans_after.reshape(2,3))

            print('rot equi test:  ' + ('YES' if torch.allclose(rot_6d_trans, rot_6d_trans_after, atol=1e-4, rtol=1e-4) else 'NO'))
            print('width inv test:  ' + ('YES' if torch.allclose(width, width_trans, atol=1e-4, rtol=1e-4) else 'NO'))
            print('qual inv test:  ' + ('YES' if torch.allclose(qual, qual_trans, atol=1e-4, rtol=1e-4) else 'NO'))
        return


class TriEquiIGDInduce(TriEquiIGD):
    """IGD with an induced-representation tri-plane backbone.

    The XZ/YZ pair is kept as ``Ind_C2^C4`` instead of being reduced to
    invariant side-plane channels.  Consequently every decoder receives a
    sum of regular C4 fields and queries the side planes through
    :class:`InducedBilinearSampler`.

    The loss, diffusion and inference methods are inherited unchanged from
    :class:`TriEquiIGD`; this class only changes the equivariant feature
    representation and decoder input wiring.
    """

    def __init__(self, N=4, s2tp=True, mix_repr=True):
        if N != 4:
            raise ValueError("TriEquiIGDInduce currently supports N=4 only")

        # Do not call TriEquiIGD.__init__: its decoders are built for the
        # non-induced (regular + invariant + invariant) tri-plane type.
        nn.Module.__init__(self)
        self._init = "he"
        self.N = N
        hidden_dim = 36 if N % 3 == 0 else 32
        self.gs = no_base_space(CyclicGroup(N))
        self.backbone = JointTriUNetInduced(
            in_channel=1,
            hidden_dim=hidden_dim,
            depth=3,
            N=N,
            plane_resolution=40,
            s2tp=s2tp,
        )

        # XY contributes hidden_dim channels and the induced XZ/YZ pair
        # contributes 2 * hidden_dim channels.  All are regular C4 fields.
        decoder_in_type = FieldType(
            self.gs,
            [self.gs.regular_repr] * (3 * hidden_dim // self.gs.regular_repr.size),
        )
        decoder_hidden_dim = 3 * hidden_dim
        mixed_repr = [self.gs.regular_repr] + [self.gs.trivial_repr] * 8
        hidden_type = (
            FieldType(self.gs, mixed_repr * (decoder_hidden_dim // (8 + self.N)))
            if mix_repr else decoder_in_type
        )

        irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
        self.irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)] * 3)
        sampler = InducedBilinearSampler()

        # EquiDeformableAttn2 is equivariant provided its feature sampler is
        # equivariant.  Replacing its default sampler is what makes its
        # offsets operate on the induced side-plane basis.
        qual_sampler = EquiDeformableAttn2(decoder_in_type)
        qual_sampler.feature_sampler = sampler
        self.decoder_qual = TriFullEquiDecoder(
            in_type=decoder_in_type,
            out_type=irrep_0,
            N=N,
            hidden_size=64,
            feature_sampler=qual_sampler,
        )

        grasp_sampler = EquiGraspSO3DeformableAttn(
            decoder_in_type, hidden_type
        )
        grasp_sampler.feature_sampler = sampler
        self.decoder_grasp_qual = TriFullEquiDecoder(
            in_type=decoder_in_type,
            out_type=irrep_0,
            N=N,
            hidden_size=64,
            feature_sampler=grasp_sampler,
        )
        self.decoder_width = TriFullEquiDecoder(
            in_type=decoder_in_type,
            out_type=irrep_0,
            N=N,
            hidden_size=64,
            feature_sampler=sampler,
        )
        self.decoder_rot = TriTimeEquiDecoder(
            in_type=decoder_in_type,
            out_type=self.irrep_3x2,
            N=N,
            hidden_size=hidden_type,
        )
        rot_sampler = EquiDeformableAttn2(decoder_in_type)
        rot_sampler.feature_sampler = sampler
        self.decoder_rot.feature_sampler = rot_sampler
        self.decoder_tsdf = TriFullEquiDecoder(
            in_type=decoder_in_type,
            out_type=irrep_0,
            N=N,
            hidden_size=64,
            feature_sampler=sampler,
        )
        self.grasp_sampler = FlowMatching(
            condition_mask=[0, 0, 0, 0, 0, 0], denosing_steps=10
        )

    def export(self):
        """Export all equivariant layers to standard PyTorch modules."""
        self.backbone = self.backbone.export()
        self.decoder_grasp_qual = self.decoder_grasp_qual.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        self.decoder_width = self.decoder_width.export()
        return self

    @torch.no_grad()
    def test_eq(self, atol=1e-4, rtol=1e-4, raise_on_error=True):
        """Check C4 equivariance of the deterministic induced IGD branches.

        Diffusion sampling is intentionally excluded: it introduces random
        noise, whereas the queried quality/width/TSDF branches and the raw
        time-conditioned rotation-irrep decoder have a deterministic C4
        transformation law.
        """
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        p = torch.rand(1, 1, 3, device=device, dtype=dtype) - 0.5
        input_ = torch.randn(1, 1, 40, 40, 40, device=device, dtype=dtype)
        # A single diffusion state is an element of three copies of irrep(1).
        rot_state = torch.randn(1, 1, self.irrep_3x2.size, device=device, dtype=dtype)
        time = torch.zeros(1, device=device, dtype=dtype)

        was_training = self.training
        self.eval()
        report = {}
        try:
            input_repr = GeometricTensor(self.backbone.voxel2repr(input_), self.backbone.in_type)
            features = self.backbone(input_repr)
            qual = self.decoder_qual(p, features)
            width = self.decoder_width(p, features)
            tsdf = self.decoder_tsdf(p, features)
            rot = self.decoder_rot(rot_state, self.decoder_rot.query_feature(p, features), time)

            for g in self.gs.testing_elements:
                rho_xy = torch.as_tensor(
                    self.gs.fibergroup.irrep(1)(g), device=device, dtype=dtype
                )
                p_trans = p.clone()
                p_trans[..., :2] = torch.einsum("ij,bqj->bqi", rho_xy, p[..., :2])
                transformed_features = self.backbone(input_repr.transform(g))

                qual_trans = self.decoder_qual(p_trans, transformed_features)
                width_trans = self.decoder_width(p_trans, transformed_features)
                tsdf_trans = self.decoder_tsdf(p_trans, transformed_features)
                rot_state_trans = torch.einsum(
                    "ij,bqj->bqi",
                    torch.as_tensor(
                        self.irrep_3x2.representation(g), device=device, dtype=dtype
                    ),
                    rot_state,
                )
                rot_trans = self.decoder_rot(
                    rot_state_trans,
                    self.decoder_rot.query_feature(p_trans, transformed_features),
                    time,
                )
                rot_expected = torch.einsum(
                    "ij,bqj->bqi",
                    torch.as_tensor(
                        self.irrep_3x2.representation(g), device=device, dtype=dtype
                    ),
                    rot,
                )

                checks = {
                    "qual": torch.allclose(qual_trans, qual, atol=atol, rtol=rtol),
                    "width": torch.allclose(width_trans, width, atol=atol, rtol=rtol),
                    "tsdf": torch.allclose(tsdf_trans, tsdf, atol=atol, rtol=rtol),
                    "rot": torch.allclose(rot_trans, rot_expected, atol=atol, rtol=rtol),
                }
                errors = {
                    "qual": float((qual_trans - qual).abs().max().item()),
                    "width": float((width_trans - width).abs().max().item()),
                    "tsdf": float((tsdf_trans - tsdf).abs().max().item()),
                    "rot": float((rot_trans - rot_expected).abs().max().item()),
                }
                value = int(g.value)
                report[value] = {"passed": all(checks.values()), "checks": checks, "max_abs": errors}
        finally:
            self.train(was_training)

        failed = [value for value, result in report.items() if not result["passed"]]
        if failed and raise_on_error:
            raise AssertionError(f"TriEquiIGDInduce C4 equivariance failed: {report}")
        return report



def negative_sampling(grasp, label=None, rotation=True):
    """
    grasp: (bs, ns ,dim)
    """
    neg_samples = torch.zeros_like(grasp[:,:0,:]) # (bs, 0, dim)
    # neg_sample = grasp.clone()
    bs, ns = grasp.shape[0], grasp.shape[1]
    sample_type = np.random.choice([0,1])
    # if rotation is False:
    #     trans_perturb_level = 0.3
    # else:
    trans_perturb_level = 0.1
    rot_perturb_level = 0.5
    num_trans_samples = 10
    num_rotations = 5
    # neg_label = label.clone()

    if rotation is True:
        yaws = np.linspace(0.0, np.pi, num_rotations)
        for yaw in yaws[1:-1]:
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    for i in range(num_trans_samples):
        neg_sample = grasp.clone()
        # if rotation is False:
        #     neg_sample[..., :3] = torch.rand_like(grasp[...,:3]) - 0.5
        # else:
        noise = torch.randn_like(grasp[...,:3]) * trans_perturb_level
        neg_sample[..., :3] += noise
        neg_samples = torch.cat((neg_samples, neg_sample), dim=1)
        if rotation is True:
            yaws = np.linspace(0.0, np.pi, num_rotations)
            yaw = np.random.choice(yaws[1:-1])
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    return neg_samples, torch.zeros_like(neg_samples[...,0])

if __name__=="__main__":
    p = torch.rand(1, 10, 3)
    input = torch.randn(1, 1, 40, 40, 40)
    model = TriEquiIGDInduce().eval()#.cuda()
    model.export()

    # caculate params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


    model = TriEquiIGD().eval()#.cuda()
    model.export()

    # caculate params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    # out = model(input, p)
    # export_model = model.export()
    # export_out = export_model(input, p)

    # for i in range(len(out)):
    #     print('export test:  ' + ('YES' if torch.allclose(out[i],export_out[i], atol=1e-4, rtol=1e-4) else 'NO'))

    # model.test_eq()
    # report = model.test_eq()
    # print(report)
    # exported = model.export()
