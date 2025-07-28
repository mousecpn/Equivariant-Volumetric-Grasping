import torch
import sys
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np
import matplotlib.pyplot as plt
from utils.transform import quaternion_to_matrix, matrix_to_quaternion, rotation_6d_to_matrix, matrix_to_rotation_6d
from model.decoder import BilinearSampler
import escnn.nn as enn
from escnn.gspaces import no_base_space, rot2dOnR2
from escnn.group import CyclicGroup
from model.export_layers import *

class DeformableAttn(nn.Module):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 1,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2
    ):
        super().__init__()
        self.feature_sampler = feature_sampler
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.grid_scale = grid_scale
        self.sp = sample_point_per_axis
        self.embed_dim = feature_dim//num_heads
        self.offset_context = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_k = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_v = nn.Linear(self.feature_dim, self.embed_dim)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = nn.Linear(self.feature_dim, out_dim)
        
        self.scale = self.embed_dim ** -0.5

        self.act = nn.ReLU(inplace=True)
        
        sp = self.sp
        self.to_offset = nn.Linear(self.feature_dim, sp*sp*sp*num_heads*3, bias = True)
        anchor = torch.zeros((sp,sp,sp))
        grid = create_grid_like(anchor)
        grid = anchor + grid
        grid_scaled = normalize_grid(grid, dim=0) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        grid_scaled = grid_scaled.unsqueeze(3).repeat(1,1,1,num_heads,1)
        
        constant_init(self.to_offset, 0.)
        self.to_offset.bias.data = grid_scaled.view(-1)    

        xavier_init(self.to_v, distribution='uniform', bias=0.)
        xavier_init(self.to_k, distribution='uniform', bias=0.)
        xavier_init(self.to_q, distribution='uniform', bias=0.)    
        xavier_init(self.to_out, distribution='uniform', bias=0.) 
        
        self.sample_points = None   

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 3)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        sp = self.sp
        # anchor = torch.zeros((bs*ns,3,sp,sp,sp)).to(query_pos.device)
        # grid = create_grid_like(anchor)
        # grid = anchor + grid
        # grid_scaled = normalize_grid(grid) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        # grid_scaled = grid_scaled.reshape(bs, ns, -1, 3)
        
        # aux_sample_point = query_pos.unsqueeze(2) + grid_scaled # (bs, ns,sp*sp*sp,3)
        # aux_feature = self.feature_sampler(aux_sample_point.reshape(bs*ns, -1, 3), c).reshape(bs, ns,sp*sp*sp, self.num_heads, self.embed_dim)
        
        # feature fusion for offset
        # local_context = self.act(self.offset_context(torch.mean(aux_feature, dim=2,keepdim=True))) + aux_feature
        
        aux_sample_point_offset = self.to_offset(feature).reshape(bs,ns,-1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 1, 3)
        
        self.sample_points = aux_sample_point
        
        # aux_sample_point_offset = aux_sample_point.unsqueeze(3) + offset # (bs, ns, sp*sp*sp, n_heads, 3)
        
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, sp*sp*sp, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
        
        
        k = self.to_k(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        v = self.to_v(aux_feature_offset).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out
    
class SO3DeformableAttn(nn.Module):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 4,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2
    ):
        super().__init__()
        self.feature_sampler = feature_sampler
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.grid_scale = grid_scale
        self.sp = sample_point_per_axis
        self.embed_dim = feature_dim//num_heads
        self.offset_context = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.to_q = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_k = nn.Linear(self.feature_dim, self.embed_dim)
        self.to_v = nn.Linear(self.feature_dim, self.embed_dim)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = nn.Linear(self.feature_dim, out_dim)
        
        self.scale = self.embed_dim ** -0.5

        self.act = nn.ReLU(inplace=True)

        sp = self.sp
        self.to_offset = nn.Linear(self.embed_dim, 3, bias = False)
        constant_init(self.to_offset, 0.)
        # self.to_offset = nn.Linear(self.feature_dim, sp*sp*sp*num_heads*3, bias = True)
        anchor = torch.zeros((sp,sp,sp))
        grid = create_grid_like(anchor)
        grid = anchor + grid
        grid_scaled = normalize_grid(grid, dim=0) / self.grid_scale # (sp,sp,sp,3)
        grid_scaled = grid_scaled.unsqueeze(3) # (sp,sp,sp, 3)
        self.grid_scaled = nn.Parameter(grid_scaled, requires_grad=True)
        
        # constant_init(self.to_offset, 0.)
        # self.to_offset.bias.data = grid_scaled.view(-1)    

        xavier_init(self.to_v, distribution='uniform', bias=0.)
        xavier_init(self.to_k, distribution='uniform', bias=0.)
        xavier_init(self.to_q, distribution='uniform', bias=0.)    
        xavier_init(self.to_out, distribution='uniform', bias=0.)    

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:]
        query_pos = query_pos[...,:3]
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        sp = self.sp
        
        grid_scaled = self.grid_scaled.unsqueeze(0).repeat(bs*ns,1,1,1,1,1).reshape(bs,ns,-1,3)
        
        # rotation
        rot_SO3 = quaternion_to_matrix(query_ori.reshape(-1,4)).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        grid_scaled = torch.einsum('bnpd,bngd->bngp', rot_SO3, grid_scaled)
        
        anchor_sample_point = query_pos.unsqueeze(2) + grid_scaled # (bs, ns,sp*sp*sp,3)
        
        anchor_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs, ns,sp*sp*sp, self.num_heads, self.embed_dim)
        
        # feature fusion for offset
        context_anchor_feature = self.act(self.offset_context(torch.mean(anchor_feature, dim=2, keepdim=True))) + anchor_feature # (bs, ns, sp*sp*sp, self.num_heads, self.embed_dim)
        
        anchor_offset = self.to_offset(context_anchor_feature).reshape(bs, ns, -1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)

        sample_point = anchor_offset+anchor_sample_point.reshape(bs, ns, -1, 1, 3)
                
        sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, sp*sp*sp, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
        
        
        k = self.to_k(sample_feature).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        v = self.to_v(sample_feature).transpose(1,2) # (bs*ns, n_heads, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out


class GraspSO3DeformableAttn(SO3DeformableAttn):
    def __init__(
        self,
        feature_dim,
        out_dim,
        feature_sampler,
        num_heads = 4,
        dropout = 0.1,
        grid_scale = 80.,
        sample_point_per_axis = 2,
        zero_offset = False,
        fixed_control_points = False
    ):
        super().__init__(
            feature_dim = feature_dim,
            out_dim= out_dim,
            feature_sampler=feature_sampler,
            num_heads = num_heads,
            dropout = dropout,
            grid_scale = grid_scale,
            sample_point_per_axis = sample_point_per_axis
        )
        
        control_points = [[0.0, 0.0, 0.0], # wrist
                          [0.0, -0.04, 0.05], # right_finger_tip
                          [0.0, 0.04, 0.05],  # left_finger_tip
                          [0.0, -0.04, 0.00], # palm
                          [0.0, 0.04, 0.00],
                          ]
        
        ### complicate gripper #######
        # num_inter_points = 5
        # palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        # palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
        # finger_z = np.linspace(0.0, 0.05, num_inter_points)
        # left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
         ### complicate gripper #######
        
        ### intersection gripper #######
        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_z = np.linspace(0.0, 0.05, num_inter_points)
        palm_x = np.zeros(1)
        control_points = np.stack(np.meshgrid(palm_x, palm_y,palm_z),axis=-1).reshape(-1,3)
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
        ### intersection gripper #######

        
        control_points = torch.from_numpy(np.array(control_points).astype('float32')) / 0.3
        if fixed_control_points:
            self.register_buffer('control_points', control_points)
        else:
            self.control_points = nn.Parameter(control_points)
        self.zero_offset = zero_offset
        self.to_offset = nn.Linear(self.feature_dim, self.num_heads*3, bias = False)
        constant_init(self.to_offset, 0.)
        # 
        
    def forward(self, query_pos, c, voxel_grid=None):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:]
        query_pos = query_pos[...,:3]
        
        
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        
        # sp = self.sp
        # anchor = torch.zeros((bs*ns,3,sp,sp,sp)).to(query_pos.device)
        # grid = create_grid_like(anchor)
        # grid = anchor + grid
        # grid_scaled = normalize_grid(grid) / self.grid_scale # (bs*ns,sp,sp,sp,3)
        # grid_scaled = grid_scaled.reshape(bs, ns, -1, 3) # grasp2offset
        
        # rotation
        rot_SO3 = quaternion_to_matrix(query_ori.reshape(-1,4)).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        # control_points_ = control_points.clone()
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        # control_points = torch.einsum('bndp,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

        # print_grasp(voxel_grid[0].detach().cpu().numpy(), rot_SO3[0,0].detach().cpu().numpy(), (query_pos[0,0].detach().cpu().numpy()+0.5)*0.3)

        
        
        #### test ########
        # R_z_ = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        # R_z = torch.from_numpy(R_z_.as_matrix().astype('float32')).to(rot_SO3.device)
        # control_points_z = torch.einsum('pd, bngd -> bngp', R_z , control_points_)
        # control_points_2 = torch.einsum('bnpd, bngd -> bngp', rot_SO3 , control_points_z)
        
        # rot_SO3_2 = torch.einsum('bngd, dp -> bngp', rot_SO3 , R_z)
        # # rot_SO3_2 = torch.einsum('bngd, dp -> bngp', rot_SO3_2 , R_z)
        # control_points_4 = torch.einsum('bnpd, bngd -> bngp', rot_SO3_2 , control_points_z)
        
        # term = SO3()
        # term.update(rot_SO3_2.reshape(-1,3,3))
        # term = term.to_quaternion().reshape(bs,ns,4)
        
        # rot_SO3_z_ = Rotation.from_matrix(rot_SO3.reshape(-1,3,3).detach().cpu().numpy())
        # R_z_ =  Rotation.from_matrix(R_z_.as_matrix().repeat(bs*ns).reshape(-1,3,3))
        # rot_SO3_z_ = (rot_SO3_z_ * R_z_).as_quat()
        # rot_SO3_z = torch.from_numpy(rot_SO3_z_.astype('float32')).to(rot_SO3.device)
        # rot_SO3_z = SO3(quat_scipy2theseus(rot_SO3_z.reshape(-1,4))).to_matrix().reshape(bs,ns,3,3)

        # control_points_3 = torch.einsum('bnpd, bngd -> bngp', rot_SO3_z.reshape(bs,ns,3,3) , control_points_)
        # print()
        #### test ########
                
        # feature fusion for offset
        if self.zero_offset:
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            """
            anchor_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs, ns, -1, self.num_heads, self.embed_dim)
            context_anchor_feature = self.act(self.offset_context(torch.mean(anchor_feature, dim=2, keepdim=True))) + anchor_feature # (bs, ns, sp*sp*sp, self.num_heads, self.embed_dim)
            anchor_offset = self.to_offset(context_anchor_feature).reshape(bs, ns, -1, self.num_heads, 3) # (bs, ns, sp*sp*sp, n_heads, 3)
            sample_point = anchor_offset+anchor_sample_point.reshape(bs, ns, -1, 1, 3)
            sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns, -1, self.num_heads, self.feature_dim) # (bs*ns, sp*sp*sp, feature_dim)
            """
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
                    
            sample_feature = self.feature_sampler(sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
            
        
        k = self.to_k(sample_feature).transpose(1,2) # (bs*ns, n_heads, ncp, embed_dim)
        v = self.to_v(sample_feature).transpose(1,2) # (bs*ns, n_heads, ncp, embed_dim)
        
        q = self.to_q(feature).reshape(bs*ns, 1, self.embed_dim).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = self.to_out(out)
        out += feature
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out


class EquiDeformableAttn(nn.Module):
    def __init__(
        self,
        in_type,
        hidden_type=None,
        num_samples = 8
    ):
        super(EquiDeformableAttn, self).__init__()
        self.in_type = in_type
        self.gs = no_base_space(CyclicGroup(4))
        self.num_samples = num_samples
        coord_type = enn.FieldType(self.gs, num_samples*[self.gs.fibergroup.irrep(1), self.gs.fibergroup.irrep(0)])
        if hidden_type is None:
            hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (self.in_type.size//4))

        self.to_q = enn.Linear(self.in_type, hidden_type)
        self.to_k = enn.Linear(self.in_type, hidden_type)
        self.to_v = enn.Linear(self.in_type, hidden_type)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = enn.Linear(hidden_type, self.in_type)
        
        self.scale = hidden_type.size ** -0.5
        
        self.to_offset = enn.Linear(self.in_type, coord_type, bias = True)
        
        self.sample_points = None   
        self.feature_sampler = BilinearSampler()
        # self.pe = RotaryPositionEncoding3D(hidden_type.size)
        self.pe = None

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 3)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        
        feature = self.feature_sampler(query_pos, c)

        feature = enn.GeometricTensor(feature.reshape(bs*ns, -1), self.in_type)
        

        aux_sample_point_offset = self.to_offset(feature).tensor.reshape(bs,ns,-1, 3) # (bs, ns, sp*sp*sp, 3)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 3)
        # return enn.GeometricTensor(aux_sample_point.reshape(-1, self.to_offset.out_type.size), self.to_offset.out_type)
        
        self.sample_points = aux_sample_point
                
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns*self.num_samples, self.in_type.size) # (bs*ns*sp*sp*sp, feature_dim)
        aux_feature_offset = enn.GeometricTensor(aux_feature_offset, self.in_type)
        # return aux_feature_offset
        
        k = self.to_k(aux_feature_offset).tensor.reshape(bs*ns, self.num_samples, -1) # (bs*ns, sp*sp*sp, embed_dim)
        v = self.to_v(aux_feature_offset).tensor.reshape(bs*ns, self.num_samples, -1) # (bs*ns, sp*sp*sp, embed_dim)
        
        q = self.to_q(feature).tensor.reshape(bs*ns, 1, -1) # (bs*ns, embed_dim)

        if self.pe is not None:
            q_pe = self.pe(query_pos).reshape(bs*ns, 1, -1, 2)
            k_pe = self.pe(aux_sample_point.reshape(bs,-1,3)).reshape(bs*ns, -1, self.pe.feature_dim, 2)
            q_cos, q_sin = q_pe[..., 0], q_pe[..., 1]
            k_cos, k_sin = k_pe[..., 0], k_pe[..., 1]
            q = self.pe.embed_rotary(q, q_cos, q_sin)
            k = self.pe.embed_rotary(k, k_cos, k_sin)

        ##### attention part #####
        q = q / self.scale
        sim = einsum('b i d, b j d -> b i j', q, k) # (bs*ns, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)        
        
        out = einsum('b i j, b j d -> b i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
        out = enn.GeometricTensor(out.reshape(-1, self.to_out.in_type.size), self.to_out.in_type)
        out = self.to_out(out)
        ##### attention part #####

        out = out + feature
        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out.tensor.reshape(bs, ns, -1)

class EquiDeformableAttn2(nn.Module):
    def __init__(
        self,
        in_type,
        hidden_type=None,
        num_samples = 8
    ):
        super(EquiDeformableAttn2, self).__init__()
        self.in_type = in_type
        self.gs = no_base_space(CyclicGroup(4))
        self.num_samples = num_samples
        coord_type = enn.FieldType(self.gs, num_samples*[self.gs.fibergroup.irrep(1), self.gs.fibergroup.irrep(0)])
        weight_type = enn.FieldType(self.gs, num_samples*[self.gs.fibergroup.irrep(0)])
        if hidden_type is None:
            hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (self.in_type.size//4))

        self.to_v = enn.Linear(self.in_type, hidden_type)
        # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.to_out = enn.Linear(hidden_type, self.in_type)
        
        self.scale = hidden_type.size ** -0.5
        
        self.to_offset = enn.Linear(self.in_type, coord_type, bias = True)
        self.to_weight = enn.Linear(self.in_type, weight_type, bias = True)
        
        self.sample_points = None   
        self.feature_sampler = BilinearSampler()
        # self.pe = RotaryPositionEncoding3D(hidden_type.size)
        self.pe = None

    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 3)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        
        feature = self.feature_sampler(query_pos, c)

        feature = enn.GeometricTensor(feature.reshape(bs*ns, -1), self.in_type)

        aux_sample_point_offset = self.to_offset(feature).tensor.reshape(bs,ns,-1, 3) # (bs, ns, sp*sp*sp, 3)
        weight_offset = self.to_weight(feature).tensor.reshape(bs*ns, self.num_samples, 1) # (bs, ns, sp*sp*sp, 1)

        aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 3)
        # return enn.GeometricTensor(aux_sample_point.reshape(-1, self.to_offset.out_type.size), self.to_offset.out_type)
        
        self.sample_points = aux_sample_point
                
        aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns*self.num_samples, self.in_type.size) # (bs*ns*sp*sp*sp, feature_dim)
        aux_feature_offset = enn.GeometricTensor(aux_feature_offset, self.in_type)
        
        
        # return aux_feature_offset
        v = self.to_v(aux_feature_offset).tensor.reshape(bs*ns, self.num_samples, -1) # (bs*ns, sp*sp*sp, embed_dim)
        
        out = (v * weight_offset).sum(-2)
        
        out = enn.GeometricTensor(out.reshape(-1, self.to_out.in_type.size), self.to_out.in_type)
        out = self.to_out(out)
        ##### attention part #####

        out = out + feature
        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out.tensor.reshape(bs, ns, -1)
    
    def export(self,):
        model = ExportEquiDeformableAttn()
        model.feature_sampler = self.feature_sampler
        model.num_samples = self.num_samples
        model.to_out_indim = self.to_out.in_type.size
        model.in_dim = self.in_type.size
        model.to_offset = self.to_offset.export()
        model.to_weight = self.to_weight.export()
        model.to_v = self.to_v.export()
        model.to_out = self.to_out.export()
        return model

# class EquiDeformableAttn(nn.Module):
#     def __init__(
#         self,
#         in_type,
#         hidden_type,
#         concat_type=None,
#         sample_point_per_axis = 2
#     ):
#         super(EquiDeformableAttn, self).__init__()
#         self.in_type = in_type
#         self.gs = no_base_space(CyclicGroup(4))
#         self.sp = sp = sample_point_per_axis
#         coord_type = enn.FieldType(self.gs, sp*sp*sp*[self.gs.fibergroup.irrep(1), self.gs.fibergroup.irrep(0)])
#         if hidden_type is None:
#             hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (self.in_type.size//4))

#         self.to_q = enn.Linear(self.in_type, hidden_type)
#         self.to_k = enn.Linear(self.in_type, hidden_type)
#         self.to_v = enn.Linear(self.in_type, hidden_type)
#         # self.to_out = nn.Linear(self.embed_dim, self.embed_dim)

#         self.to_out = enn.Linear(hidden_type, self.in_type)
        
#         self.scale = hidden_type.size ** -0.5
        
#         self.to_offset = enn.Linear(self.in_type, coord_type, bias = True)
        
#         if concat_type is not None:
#             self.concat_type = concat_type
#             self.to_feature = enn.Linear(self.in_type+self.concat_type , self.in_type, bias = True)
        
#         self.sample_points = None   
#         self.feature_sampler = BilinearSampler()

#     def forward(self, query_pos, c, feature=None):
#         """
#         query_pos: torch.tensor(bs, ns, 3)
#         c: {'xz','xy','yz'}
#         """
#         bs, ns, _ = query_pos.shape
#         sp = self.sp
        
#         if feature is not None:
#             feature = enn.GeometricTensor(feature.reshape(bs*ns, -1), self.in_type+self.concat_type)
#             feature = self.to_feature(feature)
#         else:
#             feature = self.feature_sampler(query_pos, c)
#             feature = enn.GeometricTensor(feature.reshape(bs*ns, -1), self.in_type)
        

#         aux_sample_point_offset = self.to_offset(feature).tensor.reshape(bs,ns,-1, 3) # (bs, ns, sp*sp*sp, 3)

#         aux_sample_point = aux_sample_point_offset+query_pos.reshape(bs, ns, 1, 3)
#         # return enn.GeometricTensor(aux_sample_point.reshape(-1, self.to_offset.out_type.size), self.to_offset.out_type)
        
#         self.sample_points = aux_sample_point
                
#         aux_feature_offset = self.feature_sampler(aux_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns*sp*sp*sp, self.in_type.size) # (bs*ns*sp*sp*sp, feature_dim)
#         aux_feature_offset = enn.GeometricTensor(aux_feature_offset, self.in_type)
#         # return aux_feature_offset
        
#         k = self.to_k(aux_feature_offset).tensor.reshape(bs*ns, sp*sp*sp, -1) # (bs*ns, sp*sp*sp, embed_dim)
#         v = self.to_v(aux_feature_offset).tensor.reshape(bs*ns, sp*sp*sp, -1) # (bs*ns, sp*sp*sp, embed_dim)
        
#         q = self.to_q(feature).tensor.reshape(bs*ns, 1, -1) # (bs*ns, embed_dim)

#         ##### attention part #####
#         q = q / self.scale
#         sim = einsum('b i d, b j d -> b i j', q, k) # (bs*ns, 1, sp*sp*sp)
        
#         sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
#         attn = sim.softmax(dim = -1)        
        
#         out = einsum('b i j, b j d -> b i d', attn, v).transpose(1,2).reshape(bs,ns,-1) # (bs, ns, n_heads*embed_dim)
#         out = enn.GeometricTensor(out.reshape(-1, self.to_out.in_type.size), self.to_out.in_type)
#         out = self.to_out(out)
#         ##### attention part #####

#         out = out + feature
#         # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
#         return out.tensor.reshape(bs, ns, -1)

class EquiGraspSO3DeformableAttn(nn.Module):
    def __init__(
        self,
        in_type,
        hidden_size=128,
        num_heads = 1,
        zero_offset = False,
        fixed_control_points = False
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.num_heads = num_heads
        self.feature_dim = in_type.size
        self.gs = no_base_space(CyclicGroup(4))
        self.feature_sampler = BilinearSampler()

        if isinstance(hidden_size, enn.FieldType):
            self.hidden_type = hidden_size
        else:
            self.hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (hidden_size//4))

        self.to_q = enn.Linear(self.in_type, self.hidden_type)
        self.to_k = enn.Linear(self.in_type, self.hidden_type)
        self.to_v = enn.Linear(self.in_type, self.hidden_type)
        self.to_out = enn.Linear(self.hidden_type, self.in_type)
        self.scale = self.hidden_type.size ** -0.5
        # self.dropout_layer = nn.Dropout(dropout)

        
        
        ### fork gripper #######
        # control_points = [[0.0, 0.0, 0.0], # wrist
        #                   [0.0, -0.04, 0.05], # right_finger_tip
        #                   [0.0, 0.04, 0.05],  # left_finger_tip
        #                   [0.0, -0.04, 0.00], # palm
        #                   [0.0, 0.04, 0.00],
        #                   ]
        # num_inter_points = 5
        # palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        # palm_points = np.stack((np.zeros(palm_y.shape), palm_y, np.zeros(palm_y.shape)),axis=-1)
        # finger_z = np.linspace(0.0, 0.05, num_inter_points)
        # left_finger = np.stack((np.zeros(finger_z.shape), 0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # right_finger = np.stack((np.zeros(finger_z.shape), -0.04*np.ones(finger_z.shape), finger_z),axis=-1)
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
         ### fork gripper #######
        
        ### intersection gripper #######
        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_z = np.linspace(0.0, 0.05, num_inter_points)
        palm_x = np.zeros(1)
        control_points = np.stack(np.meshgrid(palm_x, palm_y,palm_z),axis=-1).reshape(-1,3)
        self.ncp = control_points.shape[0]
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
        ### intersection gripper #######
        # self.pe = RotaryPositionEncoding3D(self.hidden_type.size)
        self.pe_k = nn.Parameter(torch.rand(self.ncp, self.hidden_type.size))
        self.pe_q = nn.Parameter(torch.rand(1, self.hidden_type.size))
        # self.pe = None

        
        control_points = torch.from_numpy(np.array(control_points).astype('float32')) / 0.3
        if fixed_control_points:
            self.register_buffer('control_points', control_points)
        else:
            self.control_points = nn.Parameter(control_points)
        self.zero_offset = zero_offset
        self.to_offset = nn.Linear(self.feature_dim, self.num_heads*3, bias = False)
        constant_init(self.to_offset, 0.)
        
        
    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:] # should be rot_6d
        query_pos = query_pos[...,:3]
        assert query_ori.shape[-1] == 6
        
        # with torch.no_grad():
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        # rotation
        rot_SO3 = rotation_6d_to_matrix(query_ori).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        # control_points_ = control_points.clone()
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        # control_points = torch.einsum('bndp,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

        # print_grasp(voxel_grid[0].detach().cpu().numpy(), rot_SO3[0,0].detach().cpu().numpy(), (query_pos[0,0].detach().cpu().numpy()+0.5)*0.3)

                
        # feature fusion for offset
        if self.zero_offset:
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            anchor_sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
        
        

        sample_feature = enn.GeometricTensor(sample_feature.reshape(-1, self.feature_dim), self.in_type)
        feature = enn.GeometricTensor(feature.reshape(-1, self.feature_dim), self.in_type)
        # return feature
        
        k = self.to_k(sample_feature).tensor.reshape(bs*ns,  -1, self.num_heads, self.hidden_type.size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        v = self.to_v(sample_feature).tensor.reshape(bs*ns,  -1, self.num_heads, self.hidden_type.size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        
        q = self.to_q(feature).tensor.reshape(bs*ns, 1, self.hidden_type.size).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (bs*ns, n_heads, 1, embed_dim)
        
        q = q + self.pe_q.reshape(1,1,1,-1)
        k = k + self.pe_k.reshape(1, 1, self.ncp, -1)
        
        # if self.pe is not None:
        #     q_pe = self.pe(query_pos).reshape(bs*ns, 1, -1, 2)
        #     k_pe = self.pe(anchor_sample_point.reshape(bs,-1,3)).reshape(bs*ns, -1, self.pe.feature_dim, 2)
        #     q_cos, q_sin = q_pe[..., 0], q_pe[..., 1]
        #     k_cos, k_sin = k_pe[..., 0], k_pe[..., 1]
        #     q = self.pe.embed_rotary(q.reshape(bs*ns, -1, self.hidden_type.size), q_cos, q_sin).reshape(bs*ns, self.num_heads, 1, self.hidden_type.size)
        #     k = self.pe.embed_rotary(k.reshape(bs*ns, -1, self.hidden_type.size), k_cos, k_sin).reshape(bs*ns, self.num_heads, -1, self.hidden_type.size)

        
        q = q / self.scale
        
        sim = einsum('b n i d, b n j d -> b n i j', q, k) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        attn = sim.softmax(dim = -1)
        
        # attn = self.dropout_layer(attn) # (bs*ns, n_heads, 1, sp*sp*sp)
        
        
        out = einsum('b n i j, b n j d -> b n i d', attn, v).transpose(1,2).reshape(bs*ns,-1) # (bs, ns, n_heads*embed_dim)
        
        out = enn.GeometricTensor(out, self.hidden_type)
        out = self.to_out(out)#.tensor.reshape(bs, ns, -1)
        out += feature#.tensor.reshape(out.shape)
        

        # out = torch.cat((feature, aux_feature_offset),dim=1).reshape(bs, ns, -1)
        return out.tensor.reshape(bs, ns, -1)
    
    def export(self,):
        model = ExportEquiGraspSO3DeformableAttn()
        model.num_heads = self.num_heads
        model.feature_dim = self.feature_dim
        model.feature_sampler = self.feature_sampler
        model.to_v = self.to_v.export()
        model.to_out = self.to_out.export()
        model.to_k = self.to_k.export()
        model.to_q = self.to_q.export()
        model.pe_q = self.pe_q
        model.pe_k = self.pe_k
        model.ncp = self.ncp
        model.to_weight = self.to_weight.export()      
        model.control_points = self.control_points
        model.zero_offset = self.zero_offset
        model.hidden_size = self.hidden_type.size
        return model


class EquiGraspSO3DeformableAttn2(nn.Module):
    def __init__(
        self,
        in_type,
        hidden_size=128,
        num_heads = 1,
        zero_offset = True,
        fixed_control_points = False
    ):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.num_heads = num_heads
        self.feature_dim = in_type.size
        self.gs = no_base_space(CyclicGroup(4))
        self.feature_sampler = BilinearSampler()

        if isinstance(hidden_size, enn.FieldType):
            self.hidden_type = hidden_size
        else:
            self.hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (hidden_size//4))

        self.to_v = enn.Linear(self.in_type, self.hidden_type)
        self.to_out = enn.Linear(self.hidden_type, self.in_type)
        

        ### intersection gripper #######
        num_inter_points = 5
        palm_y = np.linspace(-0.04, 0.04, num_inter_points)
        palm_z = np.linspace(0.0, 0.05, num_inter_points)
        palm_x = np.zeros(1)
        control_points = np.stack(np.meshgrid(palm_x, palm_y,palm_z),axis=-1).reshape(-1,3)
        self.ncp = control_points.shape[0]
        # control_points = np.concatenate((palm_points, left_finger, right_finger), axis=0)
        ### intersection gripper #######


        weight_type = enn.FieldType(self.gs, self.ncp*[self.gs.fibergroup.irrep(0)])
        self.to_weight = enn.Linear(self.in_type, weight_type, bias = True)
        
        control_points = torch.from_numpy(np.array(control_points).astype('float32')) / 0.3
        if fixed_control_points:
            self.register_buffer('control_points', control_points)
        else:
            self.control_points = nn.Parameter(control_points)
        self.zero_offset = zero_offset
        
        
    def forward(self, query_pos, c):
        """
        query_pos: torch.tensor(bs, ns, 7)
        c: {'xz','xy','yz'}
        """
        bs, ns, _ = query_pos.shape
        query_ori = query_pos[...,3:] # should be rot_6d
        query_pos = query_pos[...,:3]
        assert query_ori.shape[-1] == 6
        
        # with torch.no_grad():
        feature = self.feature_sampler(query_pos, c).reshape(bs, ns, self.feature_dim) # (bs, ns, feature_dim)
        
        # control points
        control_points = self.control_points # (1, ncp, 3)
        control_points = control_points.repeat(bs*ns,1,1).reshape(bs, ns, -1, 3)
        
        # rotation
        rot_SO3 = rotation_6d_to_matrix(query_ori).reshape(bs,ns,3,3) # (bs*ns, 3, 3) world2grasp   
        # control_points_ = control_points.clone()
        control_points = torch.einsum('bnpd,bngd->bngp', rot_SO3, control_points)
        # control_points = torch.einsum('bndp,bngd->bngp', rot_SO3, control_points)
        anchor_sample_point = query_pos.unsqueeze(2) + control_points # (bs, ns,ncp, 3)

        # print_grasp(voxel_grid[0].detach().cpu().numpy(), rot_SO3[0,0].detach().cpu().numpy(), (query_pos[0,0].detach().cpu().numpy()+0.5)*0.3)

                
        # feature fusion for offset
        if self.zero_offset:
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs, -1, 3), c).reshape(bs*ns, -1, 1, self.feature_dim).repeat(1, 1,self.num_heads, 1)
        else:
            ######## simplified offset calculation #############
            anchor_offset = self.to_offset(feature).reshape(bs, ns, self.num_heads, 3) # (bs, ns, num_heads, 3)

            anchor_sample_point = anchor_offset.unsqueeze(3)+anchor_sample_point.reshape(bs, ns, 1, -1, 3) # (bs, ns, num_heads, ncp, 3)
            # with torch.no_grad():
            sample_feature = self.feature_sampler(anchor_sample_point.reshape(bs,-1, 3), c).reshape(bs*ns,  -1, self.num_heads, self.feature_dim) # (bs, ns, ncp, num_heads, feature_dim)
        
        

        sample_feature = enn.GeometricTensor(sample_feature.reshape(-1, self.feature_dim), self.in_type)
        feature = enn.GeometricTensor(feature.reshape(-1, self.feature_dim), self.in_type)
        weights = self.to_weight(feature).tensor.reshape(bs*ns, self.ncp, self.num_heads, 1) # (bs*ns, ncp, n_heads, 1)
        # return feature
        
        v = self.to_v(sample_feature).tensor.reshape(bs*ns,  -1, self.num_heads, self.hidden_type.size).transpose(1,2) # (bs*ns, ncp, n_heads, embed_dim)
        
        out = (v * weights).sum((1,2))
        out = enn.GeometricTensor(out, self.hidden_type)
        out = self.to_out(out)#.tensor.reshape(bs, ns, -1)
        out += feature#.tensor.reshape(out.shape)

        return out.tensor.reshape(bs, ns, -1)
    
    def export(self,):
        model = ExportEquiGraspSO3DeformableAttn2()
        model.num_heads = self.num_heads
        model.feature_dim = self.feature_dim
        model.feature_sampler = self.feature_sampler
        model.to_v = self.to_v.export()
        model.to_out = self.to_out.export()
        model.ncp = self.ncp
        model.to_weight = self.to_weight.export()      
        model.control_points = self.control_points
        model.zero_offset = self.zero_offset
        model.hidden_size = self.hidden_type.size
        return model

def create_grid_like(t, dim = 0):
    f, h, w, device = *t.shape[-3:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid



def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    f, h, w = grid.shape[-3:]
    grid_f, grid_h, grid_w = grid.unbind(dim = dim)

    grid_f = 2.0 * grid_f / max(f - 1, 1) - 1.0
    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_f, grid_h, grid_w), dim = out_dim)

def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
    
def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)


if __name__=="__main__":
    from scipy.spatial.transform import Rotation
    context = {}
    bs = 1
    c_dim = 32
    x_dim=y_dim=z_dim = 40 # h-->x, w-->y, z-->z
    xy_feat = xz_feat = yz_feat = torch.randn((bs, c_dim, x_dim, y_dim))
    
    context['xz'] = xz_feat * 0
    context['yz'] = yz_feat * 0

    pos = torch.rand((bs,1,3))*2 - 1
    grasp = torch.randn((bs,2,9))
    grasp[...,3:] = matrix_to_rotation_6d(quaternion_to_matrix(nn.functional.normalize(grasp[...,3:7], dim=-1)))
    gs = no_base_space(CyclicGroup(4))
    in_type = enn.FieldType(gs, [gs.regular_repr]*8+[gs.trivial_repr]*32*2)
    out_type = enn.FieldType(gs, [gs.regular_repr]*(96//4))
    gs2 = rot2dOnR2(4)
    
    # attn = EquiDeformableAttn2(in_type, None)
    # for g in attn.gs.testing_elements:
    #     if g.value == 0:
    #         continue
    #     context['xy'] = xy_feat.permute(0,1,3,2)
    #     feat = attn(pos, context)

    #     xy_feat_ = enn.GeometricTensor(xy_feat, enn.FieldType(gs2, [gs2.regular_repr]*8))
    #     context['xy']  = xy_feat_.transform(g).tensor.permute(0,1,3,2)
        
    #     rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value*np.pi / 2.0])
    #     rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)

    #     p_trans = pos.clone()
    #     p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1,3)), dtype=torch.float32).reshape(bs,1,3)
        
    #     feat_trasnform = attn(p_trans, context)
    #     feat_transformed_after = feat.transform(g)
        

    #     print('rot equi test:  ' + ('YES' if torch.allclose(feat_trasnform.tensor, feat_transformed_after.tensor, atol=1e-1, rtol=1e-1) else 'NO'))
        
        
    grasp_attn = EquiGraspSO3DeformableAttn(in_type, hidden_size=144)
    for g in grasp_attn.gs.testing_elements:
        if g.value == 0:
            continue
        context['xy'] = xy_feat.permute(0,1,3,2)
        feat = grasp_attn(grasp, context)

        xy_feat_ = enn.GeometricTensor(xy_feat, enn.FieldType(gs2, [gs2.regular_repr]*8))
        context['xy']  = xy_feat_.transform(g).tensor.permute(0,1,3,2)
        
        rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value*np.pi / 2.0])
        rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)

        p_trans = grasp[...,:3].clone()
        p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1,3)), dtype=torch.float32).reshape(bs,2,3)

        rot_trans = grasp[...,3:].reshape(bs,2,6).clone()
        rot_trans = torch.bmm(torch.tensor(rot.as_matrix(), dtype=torch.float32).unsqueeze(0).repeat(bs*2, 1,1), rotation_6d_to_matrix(rot_trans).reshape(-1,3,3))
        rot_trans = matrix_to_rotation_6d(rot_trans).reshape(bs,2,6)
        grasp_trans = torch.cat((p_trans, rot_trans), dim=-1)
        
        feat_trasnform = grasp_attn(grasp_trans, context)
        feat_transformed_after = feat.transform(g)
        

        print('rot equi test:  ' + ('YES' if torch.allclose(feat_trasnform.tensor, feat_transformed_after.tensor, atol=1e-2, rtol=1e-2) else 'NO'))
        