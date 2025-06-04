import torch
import torch.nn as nn
import os
import sys
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from model.backbone import CyclicUnet3d, Tri_UNet, LocalVoxelEncoder
from model.triunet import JointTriUNet 
from model.decoder import BilinearSampler, TriFullEquiDecoder, TriTimeEquiDecoder, TimeLocalDecoder, LocalDecoder
import escnn.nn as enn
from escnn.group import CyclicGroup
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space, rot2dOnR2
import torch.nn.functional as F
import time
from scipy import stats
from utils.transform import Rotation, Transform
import numpy as np
from model.diffusion import Diffusion
from model.attention import GraspSO3DeformableAttn, DeformableAttn
import matplotlib.pyplot as plt
from utils.loss import *

class IGD(nn.Module):
    def __init__(self):
        super(IGD, self).__init__()
        self.N = 4
        hidden_dim = 32
        num_heads = 1
        grid_scale = 80
        sp = 2

        self.encoder = LocalVoxelEncoder(c_dim=32, unet=True, plane_resolution=40)

        self.grasp_sampler = Diffusion(schedulers="DDPM", condition_mask=[1,1,1,0,0,0,0],  prediction_type='epsilon', beta_schedule='linear', num_inference_steps=10)
        self.decoder_grasp_qual = LocalDecoder(dim=3, out_dim=1, c_dim=96, hidden_size=hidden_dim, feature_sampler=GraspSO3DeformableAttn(96, out_dim=96, feature_sampler=BilinearSampler(padding=0.1,plane_type=['xz', 'xy', 'yz']), num_heads=num_heads, zero_offset=True, fixed_control_points=False))
        self.decoder_rot = TimeLocalDecoder(dim=7, out_dim=7, c_dim=96, hidden_size=128, feature_sampler=BilinearSampler(padding=0.1))
        self.decoder_qual =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=hidden_dim, feature_sampler=DeformableAttn(96, out_dim=96, feature_sampler=BilinearSampler(padding=0.1, plane_type=['xz', 'xy', 'yz']), num_heads=num_heads, grid_scale=grid_scale, sample_point_per_axis=sp))
        self.decoder_width =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=hidden_dim, feature_sampler=BilinearSampler(padding=0.1, plane_type=['xz', 'xy', 'yz']))
        self.decoder_tsdf =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=hidden_dim, feature_sampler=BilinearSampler(padding=0.1, plane_type=['xz', 'xy', 'yz']))
        self.feature_sampler = DeformableAttn(feature_dim=96, out_dim=96, feature_sampler=BilinearSampler(padding=0.1, plane_type=['xz', 'xy', 'yz']), num_heads=num_heads, grid_scale=grid_scale, sample_point_per_axis=sp)

        
    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        if target is None:
            return self.inference(inputs, p)
            
        
        label, rot_gt, width_gt, occ_value = target
        rot_gt_ = rot_gt[:, np.random.randint(0,2), :].unsqueeze(1)
        # rot_gt_6d = matrix_to_rotation_6d(quaternion_to_matrix()
        # rot_irreps = self.rot2irreps(rot_gt_6d)
        grasp = torch.cat((p, rot_gt_), dim=-1)
        
        
        # c = self.backbone(inputs)
        c = self.encoder(inputs)
        qual = self.decoder_qual(p, c)
        grasp_qual = self.decoder_grasp_qual(grasp, c)
        width = self.decoder_width(p, c)
        noise = torch.rand((self.batch_size, self.sample_num, 4), device=p.device)
        noise = torch.cat((p, noise), dim=-1)
        rot_pred = self.grasp_sampler.sample_data(noise, self.feature_sampler(p, c), self.decoder_rot)
        rot_pred = F.normalize(rot_pred[...,3:], dim=-1)
        
        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return (qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_pred, width, tsdf
        else:
            return (qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_pred, width
    
    
    
    def inference(self, inputs, p, sample_rounds = 1, low_th=0.1, **kwargs):
        assert self.batch_size==1, "batch size should be 1 in this mode" 
        # visualize_tsdf(inputs.cpu().numpy()[0])
        c = self.encoder(inputs)
        
        p = p.reshape(self.batch_size, self.sample_num,-1)
        
        feature = self.feature_sampler(p, c)
        feature_dim = feature.shape[-1]
        
        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        
        mask = (qual > low_th)
        
        p_postive = p[mask].reshape(self.batch_size, -1, 3)
        
        
        # loop mode
        for i in range(sample_rounds):
            noise = torch.rand((1, p_postive.shape[1], 4), device=p.device)
            noise = torch.cat((p_postive, noise), dim=-1)
            grasp = self.grasp_sampler.sample_data(noise, feature[mask].reshape(self.batch_size, -1, 96), self.decoder_rot)
            grasp[...,3:] = nn.functional.normalize(grasp[...,3:], dim=-1)
            grasp_qual = self.decoder_grasp_qual(grasp, c, **kwargs)
            if i == 0:
                last_grasp = grasp
                last_grasp_qual = grasp_qual
                continue
            comparing_grasp_qual = torch.stack((last_grasp_qual, grasp_qual), dim=1) # (bs, 2, pos_ns)
            comparing_grasp = torch.stack((last_grasp, grasp), dim=1) # (bs, 2, pos_ns, 7)
            last_grasp_qual, indices = torch.max(comparing_grasp_qual, dim=1)
            indices = indices.reshape(self.batch_size, 1, -1, 1).repeat(1,1,1,7)
            last_grasp = comparing_grasp.reshape(self.batch_size, 2, -1, 7).gather(1, indices)
            last_grasp = last_grasp.squeeze(1)
    
        grasp = torch.randn(self.batch_size, self.sample_num, 7).to(last_grasp.device)
        grasp[...,3:] = nn.functional.normalize(grasp[...,3:], dim=-1)
        grasp[mask] = last_grasp.reshape(-1,7)
        grasp_qual = torch.zeros_like(qual)
        grasp_qual[mask] = torch.sigmoid(last_grasp_qual.reshape(-1))

        rot = grasp[...,3:]
        
        width = self.decoder_width(grasp, c, **kwargs)
        
        # qual = grasp_qual
        qual = (qual*grasp_qual).sqrt()
        # qual = reg_qual
        
        return qual, rot, width
    

    
    def compute_loss(self, inputs, p, target, p_tsdf):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        label, rot_gt, width_gt, occ_value = target
            
        c = self.encoder(inputs)
        qual = self.decoder_qual(p, c)
        width = self.decoder_width(p, c)
        tsdf = self.decoder_tsdf(p_tsdf, c)

        grasp = torch.cat((p.repeat(1,2,1), rot_gt), dim=-1)
        
        if label.sum() > 0:
            model_input = {
                'data': grasp[label==1],
                'context': self.feature_sampler(p.repeat(1,2,1), c)[label==1]
            }
            loss_rot = self.grasp_sampler.loss_fn(self.decoder_rot, model_input).mean()
        else:
            loss_rot = torch.zeros(1).sum()
        
        loss_qual = qual_loss_fn(qual.squeeze(1), label).mean() 
        # loss_grasp_qual = _qual_loss_fn(grasp_qual.squeeze(1), label).mean()
        loss_width = width_loss_fn(width[label==1].reshape(-1), width_gt[label==1]).mean()
        loss_occ = occ_loss_fn(tsdf, occ_value).mean()
        
        grasp_qual = self.decoder_grasp_qual(grasp, c).mean(-1, keepdim=True)
        
        neg_sample, neg_label = self.massive_negative_sampling(grasp, label)
    
        grasp_qual_neg = self.decoder_grasp_qual(neg_sample, c)
        
        all_grasp_qual = torch.cat((grasp_qual, grasp_qual_neg), dim=1)
        label = torch.cat((label.reshape(-1, 1), neg_label), dim=1)
        
        loss_qual += sigmoid_focal_loss(all_grasp_qual.reshape(-1), label.reshape(-1), reduction='mean', sigmoid=False)
        
        
        loss = loss_qual +  loss_rot + 0.01 * loss_width + loss_occ
        loss_dict = {'loss_qual': loss_qual,
                    'loss_rot': loss_rot,
                    'loss_width': loss_width,
                    'loss_occ': loss_occ,
                    'loss_all': loss}
        
        y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_gt[:,:1], width, tsdf)
        
        return loss, loss_dict, y_pred
    
    
    def massive_negative_sampling(self, grasp, label):
        neg_samples = torch.zeros_like(grasp[:,:0,:])
        # neg_sample = grasp.clone()
        bs, ns = grasp.shape[0], grasp.shape[1]
        sample_type = np.random.choice([0,1])
        trans_perturb_level = 0.1
        rot_perturb_level = 0.5
        num_trans_samples = 10
        num_rotations = 6
        # neg_label = label.clone()

        yaws = np.linspace(0.0, np.pi, num_rotations)
        for yaw in yaws[1:-1]:
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_quat()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = neg_rot.reshape(bs,ns,4)
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

        for i in range(num_trans_samples):
            neg_sample = grasp.clone()
            noise = torch.randn_like(grasp[...,:3]) * trans_perturb_level
            neg_sample[..., :3] += noise
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)
            yaws = np.linspace(0.0, np.pi, num_rotations)
            yaw = np.random.choice(yaws[1:-1])
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_quat()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = neg_rot.reshape(bs,ns,4)
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

        return neg_samples, torch.zeros_like(neg_samples[...,0])
            
    
    def irreps2rot(self, irrep):
        cos1 = irrep[..., 0:1]
        sin1 = irrep[..., 1:2]
        cos2 = irrep[..., 2:3]
        sin2 = irrep[..., 3:4]
        cos3 = irrep[..., 4:5]
        sin3 = irrep[..., 5:6]
        return torch.cat((cos1, cos2, cos3, sin1, sin2, sin3), dim=-1)

    def rot2irreps(self, rot):
        cos1 = rot[..., 0:1]
        cos2 = rot[..., 1:2]
        cos3 = rot[..., 2:3]
        sin1 = rot[..., 3:4]
        sin2 = rot[..., 4:5]
        sin3 = rot[..., 5:6]
        return torch.cat((cos1, sin1, cos2, sin2, cos3, sin3), dim=-1)
        

        
    def test_eq(self,):
        p = torch.rand(1, 1, 3)-0.5
        input_ = torch.randn(1, 1, 40, 40, 40)
        for g in self.gs.testing_elements:
            if g.value == 0:
                continue
            input = self.backbone.voxel2repr(input_)
            input = GeometricTensor(input, self.backbone.in_type)
            input_transformed = input.transform(g)

            rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value*np.pi / 2.0])
            rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)

            p_trans = p.clone()
            p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1,3))).reshape(1,1,3)
            p_trans = torch.tensor(p_trans, dtype=torch.float32)

            qual, rot, width = self.forward(input, p)
            qual_trans, rot_trans, width_trans = self.forward(input_transformed, p_trans)

            rot_trans_after = (rot_mat @ rot.reshape(2,3)).reshape(1,1,6)

            print('rot equi test:  ' + ('YES' if torch.allclose(rot_trans, rot_trans_after, atol=1e-4, rtol=1e-4) else 'NO'))
            print('width inv test:  ' + ('YES' if torch.allclose(width, width_trans, atol=1e-4, rtol=1e-4) else 'NO'))
            print('qual inv test:  ' + ('YES' if torch.allclose(qual, qual_trans, atol=1e-4, rtol=1e-4) else 'NO'))
        return
    

if __name__=="__main__":
    model = IGD()#.cuda()
    # p = torch.rand(128, 1, 3).cuda()
    # input = torch.randn(128, 1, 40, 40, 40).cuda()
    # model(input, p)

    model.test_eq()