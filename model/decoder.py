import torch
import torch.nn as nn
import escnn.nn as enn
from escnn.group import so3_group, directsum, CyclicGroup, so2_group
from escnn.gspaces import no_base_space
import sys
sys.path.append('/home/pinhao/IGDv2')
from utils.common import normalize_coordinate, normalize_3d_coordinate
import torch.nn.functional as F
from escnn.nn import FieldType, GeometricTensor
from model.pos_encoding import SinusoidalPosEmb
from scipy import stats
from model.layer import ResnetBlockFC
from model.equi_layer2d import EquiResnetBlockFC
from einops import rearrange
from model.export_layers import *

class EquiLocalDecoder(nn.Module):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None):
        super(EquiLocalDecoder, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = 'bilinear'
        self.c_dim = self.in_type.size//N

        if self.c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(self.c_dim, hidden_size) for i in range(n_blocks)
            ])
        
        self.gs = no_base_space(CyclicGroup(self.N))

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])
        self.hidden_type = FieldType(self.gs, (hidden_size)*[self.gs.regular_repr])

        self.out_layer = enn.Linear(self.hidden_type, self.out_type)
        

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
    
    
    def forward(self, p, c_voxel):
        if isinstance(c_voxel, torch.Tensor):
            c = self.sample_grid_feature(p, c_voxel)
        elif isinstance(c_voxel, GeometricTensor):
            c = self.sample_grid_feature(p, c_voxel.tensor)
        else:
            raise ValueError('Invalid input type')
        c = c.transpose(1,2)
        batch_size, sample_size = p.shape[0], p.shape[1]

        c = rearrange(c, "b s (c f) -> (b f) s c", f=self.N)

        p = p[...,:self.dim].float() # (b,s,dim)

        net = torch.zeros(c.size(0), c.size(1), self.hidden_size).to(p.device)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        net = rearrange(net, "(b f) s c -> (b s) (c f)", f=self.N)
        net = enn.GeometricTensor(net, self.hidden_type)
        
        out = self.out_layer(net)
        out = out.tensor.reshape(batch_size, sample_size, -1).squeeze(-1)
            
        return out



class LocalDecoder(nn.Module):
    def __init__(self, dim=3, c_dim=32,
                 hidden_size=32, 
                 n_blocks=5, 
                 out_dim=1, 
                 leaky=False, 
                 no_xyz=False,
                 padding=0.0,
                 feature_sampler=None):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        self.padding = padding
        self.sample_mode = 'bilinear'

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.dim = dim
        self.feature_sampler = feature_sampler

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
    
    def query_feature(self, p, c_plane):
        if self.feature_sampler is not None:
            return self.feature_sampler(p, c_plane)
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            else:
                c = []
                if 'xy' in c_plane:
                    c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
                if 'xz' in c_plane:
                    c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
                if 'yz' in c_plane:
                    c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
                c = torch.cat(c, dim=1)
        c = c.transpose(1, 2)
        return c
        

    def forward(self, xw, c_plane, **kwargs):
        p = xw[...,:self.dim].float()
        if self.feature_sampler is not None:
            if isinstance(self.feature_sampler, BilinearSampler):
                c = self.feature_sampler(p, c_plane)
            else:
                c = self.feature_sampler(xw, c_plane)
        else:
            c = self.query_feature(p, c_plane)
        
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size).to(p.device)
        else:
            net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out
    

class TimeLocalDecoder(LocalDecoder):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, 
                 n_blocks=5, 
                 out_dim=1, 
                 leaky=False, 
                 sample_mode='bilinear', 
                 padding=0.1,
                 feature_sampler=None,
                 no_xyz=False,
                 t_dim=16):
        super(LocalDecoder, self).__init__()
        self.dim = dim
        self.feature_sampler = feature_sampler
        c_dim +=  t_dim #+ self.points.shape[0] * self.points.shape[1]
        self.c_dim = c_dim 
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

    def forward(self, data, c , time, **kwargs):
        batch_size, sample_num = data.shape[0], data.shape[1]
        
        net = self.fc_p(data)
        
        if len(time.shape) == 1:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size, 1,-1).repeat(1, sample_num, 1)
        else:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size, sample_num,-1)

        c = torch.cat([c, t], dim=-1)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


class TriEquiLocalDecoder(nn.Module):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None, inv_map=False):
        super(TriEquiLocalDecoder, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = 'bilinear'
        self.inv_map = inv_map
        self.gs = no_base_space(CyclicGroup(self.N))
        
        if inv_map:
            self.inv_fun = enn.GroupPooling(self.in_type)
            self.out_layer = nn.Linear(hidden_size, self.out_type.size)
            self.c_dim = self.inv_fun.out_type.size
        else:
            self.hidden_type = FieldType(self.gs, (hidden_size)*[self.gs.regular_repr])
            self.lifting = enn.Linear(self.in_type, self.hidden_type)
            self.out_layer = enn.Linear(self.hidden_type, self.out_type)
            self.c_dim = hidden_size

        # if self.c_dim != 0:
        self.fc_c = nn.ModuleList([
            nn.Linear(self.c_dim, hidden_size) for i in range(n_blocks)
        ])
        
        

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c
    

    def query_feature(self, p, c_plane):
        c = []
        if 'xy' in c_plane:
            c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
        if 'xz' in c_plane:
            c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
        if 'yz' in c_plane:
            c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
        c = torch.cat(c, dim=1)
        c = c.transpose(1, 2)
        return c

    
    def forward(self, p, c_plane):
        c = self.query_feature(p, c_plane)
        batch_size, sample_size = p.shape[0], p.shape[1]

        c = GeometricTensor(c.reshape(-1,self.in_type.size), self.in_type)
        if self.inv_map:
            c = self.inv_fun(c).tensor.reshape(batch_size, sample_size, -1)
        else:
            c = self.lifting(c)
            c = rearrange(c.tensor, "(b s) (c f) -> (b f) s c", b=batch_size, f=self.N)

        p = p[...,:self.dim].float() # (b,s,dim)

        net = torch.zeros(c.size(0), c.size(1), self.hidden_size).to(p.device)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        if not self.inv_map:
            net = rearrange(net, "(b f) s c -> (b s) (c f)", f=self.N)
            net = enn.GeometricTensor(net, self.hidden_type)
            net = self.out_layer(net).tensor
        else:
            net = self.out_layer(net)
        net = net.reshape(batch_size, sample_size, -1).squeeze(-1)
        return net

class TriFullEquiDecoder(TriEquiLocalDecoder):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None, inv_map=False, feature_sampler=None):
        super(TriEquiLocalDecoder, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        self.feature_sampler = feature_sampler
        
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = 'bilinear'
        self.inv_map = inv_map
        self.gs = no_base_space(CyclicGroup(self.N))
        
        # if inv_map:
        #     self.inv_fun = enn.GroupPooling(self.in_type)
        #     self.out_layer = nn.Linear(hidden_size, self.out_type.size)
        #     self.c_dim = self.inv_fun.out_type.size
        # else:
        if isinstance(hidden_size, FieldType):
            self.hidden_type = hidden_size
        else:
            self.hidden_type = FieldType(self.gs, (hidden_size//self.N)*[self.gs.regular_repr])

        self.hidden_size = self.hidden_type.size
        self.lifting = enn.Linear(self.in_type, self.hidden_type)
        self.act = enn.ReLU(self.hidden_type, inplace=True)
        self.out_layer = enn.Linear(self.hidden_type, self.out_type)

        # if self.c_dim != 0:
        # if self.concat_type is not None:
        #     self.fc_c = nn.ModuleList([
        #         enn.Linear(self.hidden_type+self.concat_type, self.hidden_type) for i in range(n_blocks)
        #     ])
        # else:
        self.fc_c = nn.ModuleList([
            enn.Linear(self.hidden_type, self.hidden_type) for i in range(n_blocks)
        ])
        if self.concat_type is not None:
            self.fc_p = enn.Linear(self.concat_type, self.hidden_type)
        

        self.blocks = nn.ModuleList([
            EquiResnetBlockFC(self.hidden_type) for i in range(n_blocks)
        ])
    
    def query_feature(self, p, c_plane):
        if self.feature_sampler is not None:
            if isinstance(self.feature_sampler, BilinearSampler):
                c = self.feature_sampler(p, c_plane)
            else:
                c = self.feature_sampler(p, c_plane)
        else:
            c = 0
            if 'grid' in c_plane:
                c = self.sample_grid_feature(p, c_plane['grid'])
                c = c.transpose(1,2)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
    
    def forward(self, p, c_plane):
        batch_size, sample_size = p.shape[0], p.shape[1]
        c = self.query_feature(p, c_plane)
        valid_mask = ~(torch.isnan(p).all(-1))
        

        c = GeometricTensor(c[valid_mask].reshape(-1,self.in_type.size), self.in_type)
        # if self.inv_map:
        #     c = self.inv_fun(c).tensor.reshape(batch_size, sample_size, -1)
        # else:
        c = self.act(self.lifting(c))
        # if self.concat_type is not None:
        #     c = GeometricTensor(torch.cat((c.tensor, p[...,3:].reshape(batch_size*sample_size, -1)), dim=-1), self.hidden_type+self.concat_type)

        if self.concat_type is not None:
            net = GeometricTensor(p[valid_mask][...,3:].reshape(-1, self.concat_type.size), self.concat_type)
            net = self.fc_p(net)
        else:
            net = GeometricTensor(torch.zeros(c.shape[0], self.hidden_type.size).to(p.device), self.hidden_type)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        # if not self.inv_map:
        # net = rearrange(net, "(b f) s c -> (b s) (c f)", f=self.N)
        # net = enn.GeometricTensor(net, self.hidden_type)
        net = self.out_layer(net).tensor
        # else:
        #     net = self.out_layer(net)
        out = torch.full((batch_size, sample_size, net.shape[-1]), float('nan')).to(net.device)
        out[valid_mask] = net
        # net = net.reshape(batch_size, sample_size, -1).squeeze(-1)
        return out.squeeze(-1)
    
    def export(self, ):
        model = ExportTriEquiDecoder()
        model.in_dim = self.in_type.size
        model.hidden_dim = self.hidden_type.size
        model.N = self.N
        model.lifting = self.lifting.export()
        try:
            model.feature_sampler = self.feature_sampler.export()
        except:
            model.feature_sampler = self.feature_sampler
        blocks = []
        for i in range(len(self.blocks)):
            blocks.append(self.blocks[i].export())
        model.blocks = nn.ModuleList(blocks)
        model.n_blocks = len(self.blocks)

        fc_c = []
        for i in range(len(self.fc_c)):
            fc_c.append(self.fc_c[i].export())
        model.fc_c = nn.ModuleList(fc_c)
        
        if self.concat_type is not None:
            model.concat_dim = self.concat_type.size
            model.fc_p = self.fc_p.export()
        model.out_layer = self.out_layer.export()
        return model



class TriTimeEquiDecoder(TriEquiLocalDecoder):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None, feature_sampler=None, t_dim=16):
        super(TriEquiLocalDecoder, self).__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = 'bilinear'
        self.gs = no_base_space(CyclicGroup(self.N))
        self.t_type = FieldType(self.gs, (t_dim)*[self.gs.trivial_repr])
        self.feature_sampler = feature_sampler
        

        if isinstance(hidden_size, FieldType):
            self.hidden_type = hidden_size
        else:
            self.hidden_type = FieldType(self.gs, (hidden_size//self.N)*[self.gs.regular_repr])

        self.hidden_size = self.hidden_type.size
        
        self.lifting = enn.Linear(self.in_type, self.hidden_type)
        self.act = enn.ReLU(self.hidden_type, inplace=True)
        self.out_layer = enn.Linear(self.hidden_type, self.out_type)

        # if self.c_dim != 0:
        self.fc_c = nn.ModuleList([
            enn.Linear(self.hidden_type+self.t_type, self.hidden_type) for i in range(n_blocks)
        ])
        
        self.fc_p = enn.Linear(self.out_type, self.hidden_type)
        

        self.blocks = nn.ModuleList([
            EquiResnetBlockFC(self.hidden_type) for i in range(n_blocks)
        ])
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        
    def query_feature(self, p, c_plane, extra_data=None):
        if isinstance(self.feature_sampler, BilinearSampler):
            c = self.feature_sampler(p, c_plane)
        else:
            # p = torch.cat((p, extra_data), dim=-1)
            c = self.feature_sampler(p, c_plane)
        return c
        
    
    def forward(self, data, c_plane, time):
        if isinstance(c_plane, dict):
            c = self.query_feature(data, c_plane)
        else:
            c = c_plane
        batch_size, sample_size = data.shape[0], data.shape[1]
        if len(time.shape) == 1:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size, 1,-1).repeat(1, sample_size, 1).reshape(batch_size*sample_size, self.t_type.size)
        else:
            t = self.time_mlp(time.reshape(-1)).reshape(batch_size*sample_size,-1)
        

        c = GeometricTensor(c.reshape(-1,self.in_type.size), self.in_type)
        # if self.inv_map:
        #     c = self.inv_fun(c).tensor.reshape(batch_size, sample_size, -1)
        # else:
        c = self.act(self.lifting(c))
        net = GeometricTensor(data.reshape(batch_size*sample_size, -1), self.out_type)
        net = self.fc_p(net)

        
        
        c = GeometricTensor(torch.cat((c.tensor, t), dim=-1), self.hidden_type+self.t_type)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
        
        # if not self.inv_map:
        # net = rearrange(net, "(b f) s c -> (b s) (c f)", f=self.N)
        # net = enn.GeometricTensor(net, self.hidden_type)
        net = self.out_layer(net).tensor
        # else:
        #     net = self.out_layer(net)
        net = net.reshape(batch_size, sample_size, -1).squeeze(-1)
        return net
    
    def export(self,):
        model = ExportTimeEquiDecoder()
        model.in_dim = self.in_type.size
        model.hidden_dim = self.hidden_type.size
        model.N = self.N
        model.lifting = self.lifting.export()
        blocks = []
        for i in range(len(self.blocks)):
            blocks.append(self.blocks[i].export())
        model.blocks = nn.ModuleList(blocks)
        model.n_blocks = len(self.blocks)

        fc_c = []
        for i in range(len(self.fc_c)):
            fc_c.append(self.fc_c[i].export())
        model.fc_c = nn.ModuleList(fc_c)

        model.fc_p = self.fc_p.export()
        model.out_layer = self.out_layer.export()
        model.time_mlp = self.time_mlp
        model.t_dim = self.t_type.size
        try:
            model.feature_sampler = self.feature_sampler.export()
        except:
            model.feature_sampler = self.feature_sampler
        return model
        
        
        
class BilinearSampler(nn.Module):
    def __init__(self, c_dim=128, sample_mode='bilinear', padding=0.0, concat_feat=True, plane_type=['xy', 'xz', 'yz']):
        super().__init__()
        self.c_dim = c_dim
        self.sample_mode = sample_mode
        self.padding = padding
        self.concat_feat=concat_feat
        self.plane_type = plane_type
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                for plane in self.plane_type:
                    c.append(self.sample_plane_feature(p, c_plane[plane], plane=plane))   #### 2025 02 24 xz xy yz => xy xz yz    
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                for plane in self.plane_type:
                    c += self.sample_plane_feature(p, c_plane[plane], plane=plane)
                c = c.transpose(1, 2)
        return c
    
    
    
if __name__=="__main__":
    p = torch.rand(10, 1, 6)
    c = {'grid':torch.rand(10, 96, 32, 32, 32)}
    c = torch.rand(10, 1, 96)
    t = torch.rand_like(p)[:,0,0]
    # decoder = TriFullEquiDecoder(in_type=FieldType(no_base_space(CyclicGroup(4)), [CyclicGroup(4).irrep(1)]*48), out_type=FieldType(no_base_space(CyclicGroup(4)), [CyclicGroup(4).irrep(1)]), N=4, dim=3, hidden_size=32, n_blocks=5)
    decoder = TriTimeEquiDecoder(in_type=FieldType(no_base_space(CyclicGroup(4)), [CyclicGroup(4).irrep(1)]*48), out_type=FieldType(no_base_space(CyclicGroup(4)), [CyclicGroup(4).irrep(1)]*3), N=4, dim=3, hidden_size=32, n_blocks=5)

    out = decoder(p, c, t)
    export_decoder = decoder.export()
    out2 = export_decoder(p,c,t)
    print('export test:  ' + ('YES' if torch.allclose(out,out2, atol=1e-4, rtol=1e-4) else 'NO'))
    # print(out.shape)
    # print(out)