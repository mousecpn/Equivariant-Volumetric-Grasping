import torch
import torch.nn as nn
import os
import sys
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from model.backbone import LocalVoxelEncoder
from model.decoder import LocalDecoder, BilinearSampler
import torch.nn.functional as F
import time
from utils.transform import matrix_to_quaternion,quaternion_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d


class GIGA(nn.Module):
    def __init__(self):
        super(GIGA, self).__init__()
        self.encoder = LocalVoxelEncoder(c_dim=32, unet=True, plane_resolution=40)
        self.decoder_qual = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler(plane_type=['xz', 'xy', 'yz']))
        self.decoder_width = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler(plane_type=['xz', 'xy', 'yz']))
        self.decoder_rot = LocalDecoder(dim=3, c_dim=96, out_dim=4,  hidden_size=32, feature_sampler=BilinearSampler(plane_type=['xz', 'xy', 'yz']))
        self.decoder_tsdf = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler(plane_type=['xz', 'xy', 'yz']))    

        
    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)
        
        c = self.encoder(inputs)
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        rot = self.decoder_rot(p, c)
        # rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        rot = nn.functional.normalize(rot, dim=-1)
        
        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width
    

