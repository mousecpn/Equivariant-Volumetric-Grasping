import torch
import torch.nn as nn
from model.backbone import CyclicUnet3d, Tri_UNet
from model.triunet import JointTriUNet
from model.decoder import EquiLocalDecoder, LocalDecoder, TriEquiLocalDecoder,BilinearSampler, TriFullEquiDecoder
import escnn.nn as enn
import os
import sys
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)

from escnn.group import CyclicGroup
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space, rot2dOnR2
import torch.nn.functional as F
from utils.transform import matrix_to_quaternion,quaternion_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d, irreps2rot, rot2irreps, padding, negative_sampling, Rotation
import time
from scipy import stats
import numpy as np
from utils.loss import *

class EquiGIGA3d(nn.Module):
    def __init__(self):
        super(EquiGIGA3d, self).__init__()
        self._init = 'he'
        self.N = 4
        self.gs = no_base_space(CyclicGroup(self.N))
        self.backbone = CyclicUnet3d(in_channel=1, out_channel=96, hidden_dim=96, depth=3, N=self.N)
        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr]*(self.backbone.out_type.size//self.N))

        irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
        irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)]*3)
        self.decoder_qual = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        self.decoder_width = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        self.decoder_rot = TriFullEquiDecoder(decoder_in_type, irrep_3x2, dim=3, N=self.N, hidden_size=32)
        self.decoder_tsdf = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        # self.device = device
        
        # self.decoder_qual = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)
        # self.decoder_width = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)
        # self.decoder_rot = LocalDecoder(dim=3, c_dim=96, out_dim=4,  hidden_size=32)
        # self.decoder_tsdf = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)    

        
    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        c = self.backbone(inputs.unsqueeze(1))
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        
        if p_tsdf is not None:

            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width
    
        


class TriEquiGIGA(nn.Module):
    def __init__(self):
        super(TriEquiGIGA, self).__init__()
        self._init = 'he'
        self.N = 4
        hidden_dim = 32
        self.gs = no_base_space(CyclicGroup(self.N))
        # self.backbone = Tri_UNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N)
        self.backbone = JointTriUNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N, plane_resolution=40)
        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr]*(hidden_dim//self.N) + [self.gs.trivial_repr]*hidden_dim*2)

        irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
        irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)]*3)
        
        self.decoder_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=128, feature_sampler=BilinearSampler())
        self.decoder_width = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=128, feature_sampler=BilinearSampler())
        self.decoder_rot = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_3x2, N=self.N, hidden_size=128, feature_sampler=BilinearSampler())
        self.decoder_tsdf = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=128, feature_sampler=BilinearSampler()) 
         


    def export(self):
        self.backbone = self.backbone.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_width = self.decoder_width.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        return self  
    

    
    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        
        if p_tsdf is not None:

            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

        
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
    model = TriEquiGIGA().cuda()
    # p = torch.rand(128, 1, 3).cuda()
    # input = torch.randn(128, 1, 40, 40, 40).cuda()
    # model(input, p)

    # model.test_eq()
    p = torch.rand(4, 5, 3).cuda() - 0.5
    input = torch.randn(4, 1, 40, 40, 40).cuda()
    p[:,3:] = float('nan')
    
    model.scene_forward(input, p)
    