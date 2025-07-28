from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *
from model.equi_layer3d import EquiDownConv, EquiUpConv
from model.equi_layer2d import EquiDownConv2d, EquiUpConv2d
from model.unet import UNet
import torch
from torch import nn

import time
from einops import rearrange
from utils.common import normalize_coordinate, normalize_3d_coordinate, coordinate2index
try:
    from torch_scatter import scatter_mean
except:
    pass


class CyclicUnet3d(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, depth, N = 4):
        super(CyclicUnet3d, self).__init__()
        self.gs = rot2dOnR3(N)
        self._init = 'he'
        
        self.in_type = FieldType(self.gs, [self.gs.trivial_repr]* in_channel)
        hidden_type = FieldType(self.gs, [self.gs.regular_repr]* (hidden_dim//N))
        
        hidden_dims = [hidden_dim * 2 ** k for k in range(depth)]
        
        encoders = []
        self.lifting = R3Conv(self.in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True)
        self.act = ELU(hidden_type, inplace=True)
        
        for i, out_feature_num in enumerate(hidden_dims):
            if i == 0:
                encoder = EquiDownConv(hidden_dim, out_feature_num, N=N, pooling=False)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                # currently pools with a constant kernel: (2, 2, 2)
                encoder = EquiDownConv(hidden_dims[i - 1], out_feature_num, N=N, pooling=True)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoders = nn.Sequential(*encoders)
        
        decoders = []
        reversed_f_maps = list(reversed(hidden_dims))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = EquiUpConv(in_feature_num, out_feature_num, N=N)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        
        self.out_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//N))
        
        self.final_conv = R3Conv(decoders[-1].out_type, self.out_type, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = self.voxel2repr(x)
            x = GeometricTensor(x, self.in_type)
        x = self.act(self.lifting(x))
        # encoder part
        encoders_features = []
        
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            
        
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        x = self.repr2voxel(x)
        
        x = {'grid': x}
        
        return x
    
    def repr2voxel(self, repr):
        repr = repr.tensor
        shape_len = len(repr.shape)
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        voxel = rearrange(repr, "b c d w h -> b c h w d")
        return voxel
    
    def voxel2repr(self, voxel):
        shape_len = len(voxel.shape)
        if shape_len == 5:
            repr = rearrange(voxel, "b c h w d -> b c d w h")
        elif shape_len == 4:
            repr = rearrange(voxel, "b h w d -> b d w h")
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        return repr
    
    def test_eq(self):
        input_ = torch.randn(8, 1, 40, 40, 40)#.cuda()
        for g in self.gs.testing_elements:
            # permutation = [1, 2, 0]
            permutation = [2, 0, 1]
            # permutation = [0,1,2]

            input = GeometricTensor(input_, self.in_type)
            input_transformed = input.transform(g)

            features = self.forward(input)
            transform_features = self.forward(input_transformed)

            features_transformed_after = features.transform(g)
            print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-4, rtol=1e-4) else 'NO'))
        return
        

class CyclicUnet2d(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_dim, depth, N = 4):
        super(CyclicUnet2d, self).__init__()
        self.gs = rot2dOnR2(N)
        self._init = 'he'
        
        if in_channel == 1:
            first_dim = hidden_dim*2
            self.in_type = FieldType(self.gs, [self.gs.trivial_repr]* in_channel)
            hidden_type = FieldType(self.gs, [self.gs.regular_repr]* (first_dim//N))
            self.lifting = R2Conv(self.in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True)
            self.act = ReLU(hidden_type, inplace=True)
        else:
            self.in_type = FieldType(self.gs, [self.gs.regular_repr]* (in_channel//N))
            first_dim = self.in_type.size
        hidden_dims = [hidden_dim * 2 ** k for k in range(depth)]
        
        encoders = []
        
        
        for i, out_feature_num in enumerate(hidden_dims):
            if i == 0:
                encoder = EquiDownConv2d(first_dim, out_feature_num, N=N, pooling=False)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                # currently pools with a constant kernel: (2, 2, 2)
                encoder = EquiDownConv2d(hidden_dims[i - 1], out_feature_num, N=N, pooling=True)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoders = nn.Sequential(*encoders)
        
        decoders = []
        reversed_f_maps = list(reversed(hidden_dims))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = EquiUpConv2d(in_feature_num, out_feature_num, N=N)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        
        self.out_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//N))
        
        self.final_conv = R2Conv(decoders[-1].out_type, self.out_type, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = GeometricTensor(x, self.in_type)
        if self.in_type.size == 1:
            x = self.act(self.lifting(x))
        # encoder part
        encoders_features = []
        
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            
        
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)
        
        return x
    
    def test_eq(self):
        input_ = torch.randn(8, 1, 40, 40)#.cuda()
        for g in self.gs.testing_elements:
            # permutation = [1, 2, 0]
            permutation = [2, 0, 1]
            # permutation = [0,1,2]

            input = GeometricTensor(input_, self.in_type)
            input_transformed = input.transform(g)

            features = self.forward(input)
            transform_features = self.forward(input_transformed)

            features_transformed_after = features.transform(g)
            print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-4, rtol=1e-4) else 'NO'))
        return


class Tri_UNet2(nn.Module):
    def __init__(self ,in_channel=1, hidden_dim=32, depth=3, plane_resolution=40, plane_type=['xy', 'xz', 'yz'], padding=0.0):
        super(Tri_UNet2, self).__init__()
        self.lifting_dim = hidden_dim
                
        self.lifting = nn.Conv3d(in_channel, hidden_dim, kernel_size=3, padding=1, stride=1)
        self.act = nn.ReLU(inplace=True)
        
        self.unet_xy = UNet(num_classes=hidden_dim, in_channels=hidden_dim, depth=depth, start_filts=32, merge_mode='concat')
        self.unet_2 = UNet(num_classes=hidden_dim, in_channels=hidden_dim, depth=depth, start_filts=32, merge_mode='concat')
        
        
    
        self.reso_plane = plane_resolution

        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, c, plane='xz'): # transform pixel coordinate to real coordinate
        # acquire indices of features in plane
        if plane == 'xy':
            reduce_dim = -1
        elif plane == 'xz':
            reduce_dim = -2
        elif plane == 'yz':
            reduce_dim = -3
        fea_plane = c.permute(0,2,1).reshape(-1, self.lifting_dim, self.reso_plane, self.reso_plane, self.reso_plane)
        fea_plane = fea_plane.mean(reduce_dim)
        # process the plane features with UNet
        if plane == 'xy':
            fea_plane = self.unet_xy(fea_plane)
            fea_plane = fea_plane.permute(0,1,3,2) # from pixel coord to real coord

            # fea_plane = GeometricTensor(fea_plane, self.unet_xy.out_type)
        else:
            # fea_plane = GeometricTensor(fea_plane, self.inv_map.in_type)
            # fea_plane = self.inv_map(fea_plane).tensor
            fea_plane = self.unet_2(fea_plane)
            fea_plane = fea_plane.permute(0,1,3,2) # from pixel coord to real coord
        
        return fea_plane
    

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        # Acquire voxel-wise feature
        c = self.act(self.lifting(x))
        # inv_voxel = self.inv_map(c)
        batch_size, dim = c.shape[:2]
        c = c.reshape(batch_size, dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(c, plane='xz') 
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(c, plane='xy') 
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(c, plane='yz')

        return fea
        
class Tri_UNet(nn.Module):
    def __init__(self ,in_channel=1, hidden_dim=32, N=4, depth=3, plane_resolution=40, plane_type=['xy', 'xz', 'yz'], padding=0.0):
        super(Tri_UNet, self).__init__()
        self.N = N
        self.gs = rot2dOnR3(self.N)
        self.lifting_dim = hidden_dim
        self.test_eq_flag = 0
        
        
        self.in_type = FieldType(self.gs, [self.gs.trivial_repr]* in_channel)
        hidden_type = FieldType(self.gs, [self.gs.regular_repr]* (self.lifting_dim//N))
        
        self.lifting = SequentialModule(
            R3Conv(self.in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True),
            ReLU(hidden_type, inplace=True)
        )
        # self.lifting = R3Conv(self.in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True)
        # self.act = ReLU(hidden_type, inplace=True)
        # self.lifting = CyclicResBlock(in_channel=in_channel, out_channel=self.lifting_dim, N=self.N)
        
        self.unet_xy = CyclicUnet2d(hidden_type.size, hidden_dim, hidden_dim, depth=depth, N=self.N)
        self.unet_2 = UNet(num_classes=hidden_dim, in_channels=hidden_type.size, depth=depth, start_filts=32, merge_mode='concat')
        
        self.gs2d = rot2dOnR2(self.N)
        
        irrep_0 = FieldType(self.gs2d, [self.gs2d.trivial_repr] * hidden_dim)
        irrep_1 = FieldType(self.gs2d, [self.gs2d.fibergroup.irrep(1)])
        self.inv_map = R2Conv(self.unet_xy.in_type, irrep_0, kernel_size=3, padding=1, stride=1, initialize=True)

        
        self.test_irrep1 = SequentialModule(
            PointwiseAdaptiveAvgPool2D(self.unet_xy.out_type, (1,1)),
            R2Conv(self.unet_xy.out_type, irrep_1, kernel_size=1, padding=0, stride=1, initialize=True)
        )
        
    
        self.reso_plane = plane_resolution

        self.plane_type = plane_type
        self.padding = padding
        
    """
    def generate_plane_features(self, p, c, plane='xz'): # transform pixel coordinate to real coordinate
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) 
        # cc = c.reshape(8, 32, 40, 40, 40).mean(-1).permute(0,1,3,2)
        # cc = torch.flip(cc, (2,3))

        # process the plane features with UNet
        if plane == 'xy':
            fea_plane = fea_plane.permute(0,1,3,2) # real coord to pixel coordinate
            fea_plane = self.unet_xy(fea_plane)
            fea_plane = fea_plane.tensor.permute(0,1,3,2) # pixel coordinate to real coord 
            fea_plane = GeometricTensor(fea_plane, self.unet_xy.out_type)
        else:
            # fea_plane = GeometricTensor(fea_plane, self.inv_map.in_type)
            # fea_plane = self.inv_map(fea_plane).tensor
            fea_plane = fea_plane.permute(0,1,3,2) # from real coord to pixel coord
            fea_plane = self.unet_2(fea_plane)
            fea_plane = fea_plane.permute(0,1,3,2) # from pixel coord to real coord
        
        return fea_plane
    """
    
    def generate_plane_features(self, c, plane='xz'): # transform pixel coordinate to real coordinate
        # acquire indices of features in plane
        if plane == 'xy':
            reduce_dim = -1
        elif plane == 'xz':
            reduce_dim = -2
        elif plane == 'yz':
            reduce_dim = -3
        fea_plane = c.permute(0,2,1).reshape(-1, self.lifting_dim, self.reso_plane, self.reso_plane, self.reso_plane)
        fea_plane = fea_plane.mean(reduce_dim)
        # process the plane features with UNet
        if plane == 'xy':
            fea_plane = self.unet_xy(fea_plane)
            self.vector = self.test_irrep1(fea_plane).tensor.squeeze()
            if self.test_eq_flag == 0:
                fea_plane = fea_plane.tensor.permute(0,1,3,2) # from pixel coord to real coord

            # fea_plane = GeometricTensor(fea_plane, self.unet_xy.out_type)
        else:
            # fea_plane = GeometricTensor(fea_plane, self.inv_map.in_type)
            # fea_plane = self.inv_map(fea_plane).tensor
            fea_plane = self.unet_2(fea_plane)
            fea_plane = fea_plane.permute(0,1,3,2) # from pixel coord to real coord
        
        return fea_plane
    
    """
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = x.unsqueeze(1)
            x = self.voxel2repr(x)
            x = GeometricTensor(x, self.in_type)
        batch_size = x.tensor.size(0)
        device = x.tensor.device
        n_voxel = x.tensor.size(2) * x.tensor.size(3) * x.tensor.size(4)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.tensor.size(2)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.tensor.size(3)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.tensor.size(4)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x.tensor.squeeze(1))
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x.tensor.squeeze(1))
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x.tensor.squeeze(1))
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)

        # Acquire voxel-wise feature
        c = self.act(self.lifting(x))
        # inv_voxel = self.inv_map(c)
        c = self.repr2voxel(c)
        c = c.reshape(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
            # fea['yz'] = c.permute(0,2,1).reshape(-1,32,40,40,40).mean(0)
        # fea['inv'] = inv_voxel
        return fea
    """
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = x.unsqueeze(1)
            x = self.voxel2repr(x)
            x = GeometricTensor(x, self.in_type)

        # Acquire voxel-wise feature
        # c = self.act(self.lifting(x))
        c = self.lifting(x)
        # inv_voxel = self.inv_map(c)
        c = self.repr2voxel(c)
        batch_size, dim = c.shape[:2]
        c = c.reshape(batch_size, dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}

        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(c, plane='xz') * (1-self.test_eq_flag) 
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(c, plane='xy') 
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(c, plane='yz') * (1-self.test_eq_flag) 
        fea['vector'] = self.vector

        return fea
    
    def test_eq(self):
        input_ = torch.randn(8, 1, 40, 40, 40)#.cuda()
        self.test_eq_flag = 1
        for g in self.gs.testing_elements:
            if g.value < 1:
                continue
            permutation = [2, 0, 1]
            
            input = self.voxel2repr(input_)
            input = GeometricTensor(input, self.in_type)
            input_transformed = input.transform(g)

            # input_transformed_voxel = self.repr2voxel(input_transformed)
            # input_90 = input_.rot90(1, (2,3))

            plane = self.forward(input)
            features = plane['xy']
            vector = plane['vector']
            transform_plane = self.forward(input_transformed)
            transform_features = transform_plane['xy']
            transform_vector = transform_plane['vector']
            
            inv_feature = plane['yz']
            inv_feature_transformed = transform_plane['xz'] if g.value % 2 == 1 else transform_plane['yz'] 
            

            features_transformed_after = features.transform(g)
            print('xy equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-4, rtol=1e-4) else 'NO'))
            # print('equi test:  ' + ('YES' if torch.allclose(xz_feature, yz_feature_transformed, atol=1e-4, rtol=1e-4) else 'NO'))
            print('equi test:  ' + ('YES' if torch.allclose(inv_feature, inv_feature_transformed, atol=1e-4, rtol=1e-4) else 'NO'))
        self.test_eq_flag = 0
        return
    
    def repr2voxel(self, repr):
        repr = repr.tensor
        shape_len = len(repr.shape)
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        voxel = rearrange(repr, "b c d w h -> b c h w d")
        return voxel
    
    def voxel2repr(self, voxel):
        shape_len = len(voxel.shape)
        if shape_len == 5:
            repr = rearrange(voxel, "b c h w d -> b c d w h")
        elif shape_len == 4:
            repr = rearrange(voxel, "b h w d -> b d w h")
        repr = torch.flip(repr, (shape_len-3, shape_len-2))
        return repr
    

class LocalVoxelEncoder(nn.Module):
    ''' 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    
    '''

    def __init__(self, dim=3, c_dim=128, unet=False, plane_resolution=40, plane_type=['xy', 'xz', 'yz'], kernel_size=3, padding=0.0):
        super().__init__()
        self.actvn = nn.functional.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, start_filts=32, depth=3, )
        else:
            self.unet = None


        self.c_dim = c_dim

        self.reso_plane = plane_resolution

        self.plane_type = plane_type
        self.padding = padding
        # self.pointnet = PointNetPlusPlus(c_dim=c_dim)
        # self.equi3d = EquivariantVoxelEncoderCyclic(obs_channel=1, n_out=c_dim//4, N=4)


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid


    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)

        # voxel 3D coordintates
        coord1 = torch.linspace(-0.5, 0.5, x.size(1)).to(device)
        coord2 = torch.linspace(-0.5, 0.5, x.size(2)).to(device)
        coord3 = torch.linspace(-0.5, 0.5, x.size(3)).to(device)

        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)

        # Acquire voxel-wise feature
        x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')
                # fea['yz'] = c.permute(0,2,1).reshape(-1,32,40,40,40).mean(0)

        return fea

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    
    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_eq2d():
    from einops import rearrange
    gs = rot2dOnR2(4)
    x = torch.randn(1, 1, 40, 40).cuda()
    in_type = FieldType(gs, [gs.trivial_repr])
    # x_ = rearrange(x, "b c h w d -> b c d w h")
    # x_ = torch.flip(x_, (2, 3))
    x_ = GeometricTensor(x, in_type)
    g = list(gs.testing_elements)[1]
    
    x_transformed = x_.transform(g)
    # x_transformed = torch.flip(x_transformed.tensor, (2, 3))
    # x_transformed = rearrange(x, "b c d w h -> b c h w d")
    
    term90 = x.rot90(1, (2,3))
    
    error = (term90-x_transformed.tensor).abs().sum()
    print("error:", error)

def check_eq3d():
    
    gs = rot2dOnR3(4)
    x = torch.randn(1, 1, 40, 40, 40).cuda()
    in_type = FieldType(gs, [gs.trivial_repr])
    x_ = rearrange(x, "b c h w d -> b c d w h")
    x_ = torch.flip(x_, (2, 3))
    x_ = GeometricTensor(x_, in_type)
    g = list(gs.testing_elements)[1]
    
    x_transformed = x_.transform(g).tensor
    x_transformed = torch.flip(x_transformed, (2, 3))
    x_transformed = rearrange(x_transformed, "b c d w h -> b c h w d")
    
    term90 = x.rot90(1, (2,3))
    
    error = (term90-x_transformed).abs().sum()
    print("error:", error)



if __name__ == '__main__':
    # check_eq3d()
    # build the SE(3) equivariant model
    # m = CyclicUnet2d(in_channel=1, out_channel=32, hidden_dim=32, depth=3)
    index = torch.arange(0,1600, 1).reshape(40,40)
    m = Tri_UNet(in_channel=1, hidden_dim=32, N=4)
    # m.init()
    # total_params = count_parameters(m)
    # print(f"Total number of trainable parameters: {total_params}")

    device = 'cuda'
    # m.cuda()
    
    # m.eval()
    # m_compiled = torch.compile(m)
    
    
    # x = rearrange(x, "b t c h w d -> b t c d w h")
    
    t1 = time.time()
    # with torch.no_grad():
    m.test_eq()
    # for i in range(10):
    #     m(x)
    # t2 = time.time()
    # print("latency:", (t2 -t1)/10)
    