from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *
import os
import sys
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from model.equi_layer2d import EquiDownConv2d, EquiUpConv2d
from model.unet import conv1x1, upconv2x2 ,conv3x3
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from scipy import stats
import math
import time
from einops import rearrange
from model.export_layers import *


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True,deform=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels,deform=deform)
        self.conv2 = conv3x3(self.out_channels, self.out_channels,deform=deform)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pooling:
            x = self.pool(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.se(x)
        
        return x


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose',deform=True):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels,deform=deform)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels,deform=deform)
        self.conv2 = conv3x3(self.out_channels, self.out_channels,deform=deform)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.se(x)
        return x


class JointTriUNet(nn.Module):
    def __init__(self ,in_channel=1, hidden_dim=32, N=4, depth=3, plane_resolution=40, plane_type=['xy', 'xz', 'yz'], padding=0.0,deform=True):
        super(JointTriUNet, self).__init__()
        self.N = N
        self.gs = rot2dOnR3(self.N)
        self.lifting_dim = hidden_dim
        self.test_eq_flag = 0
        self.depth = depth
        self.deform = deform
        
        self.in_type = FieldType(self.gs, [self.gs.trivial_repr]* in_channel)
        self.hidden_type = FieldType(self.gs, [self.gs.regular_repr]* (self.lifting_dim//N))
        
        self.lifting = SequentialModule(
            R3Conv(self.in_type, self.hidden_type, kernel_size=3, padding=1, stride=1, initialize=True),
            ReLU(self.hidden_type, inplace=True)
        )

        # self.unet_2 = UNet(num_classes=hidden_dim, in_channels=self.hidden_type.size, depth=depth, start_filts=32, merge_mode='concat')
        
        self.gs2d = rot2dOnR2(self.N)
        
        
        ##### debug #####
        # irrep_0 = FieldType(self.gs2d, [self.gs2d.trivial_repr] * hidden_dim)
        # irrep_1 = FieldType(self.gs2d, [self.gs2d.fibergroup.irrep(1)])
        # self.inv_map = R2Conv(self.unet_xy.in_type, irrep_0, kernel_size=3, padding=1, stride=1, initialize=True)

        
        # self.test_irrep1 = SequentialModule(
        #     PointwiseAdaptiveAvgPool2D(self.unet_xy.out_type, (1,1)),
        #     R2Conv(self.unet_xy.out_type, irrep_1, kernel_size=1, padding=0, stride=1, initialize=True)
        # )
        ##### debug #####
        
    
        self.reso_plane = plane_resolution

        self.plane_type = plane_type
        self.padding = padding
        self.build_xyunet()
        self.build_sideunet()
        
    def build_xyunet(self):
        
        ###### xy unet ######        
        self.in_type_xy = FieldType(self.gs2d, [self.gs2d.regular_repr]* (self.lifting_dim//self.N))
        first_dim = self.in_type_xy.size
        hidden_dims = [self.lifting_dim * 2 ** k for k in range(self.depth)]
        
        encoders = []
        side2xy_layers = []
        
        for i, out_feature_num in enumerate(hidden_dims):
            if i == 0:
                encoder = EquiDownConv2d(first_dim, out_feature_num, N=self.N, pooling=False,deform=self.deform)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                # currently pools with a constant kernel: (2, 2, 2)
                encoder = EquiDownConv2d(hidden_dims[i - 1], out_feature_num, N=self.N, pooling=True,deform=self.deform)
            encoders.append(encoder)
            
            side2xy_type = encoder.out_type + FieldType(self.gs2d, out_feature_num*[self.gs2d.trivial_repr])
            side2xy = R2Conv(side2xy_type, encoder.out_type, kernel_size=1, padding=0, stride=1)
            side2xy_layers.append(side2xy)
            
        self.encoders_xy = nn.ModuleList(encoders)
        self.side2xy_encoders = nn.ModuleList(side2xy_layers)
        
        decoders = []
        side2xy_layers = []
        reversed_f_maps = list(reversed(hidden_dims))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = EquiUpConv2d(in_feature_num, out_feature_num, N=self.N,deform=self.deform)
            decoders.append(decoder)
            
            side2xy_type = decoder.out_type + FieldType(self.gs2d, out_feature_num*[self.gs2d.trivial_repr])
            side2xy = R2Conv(side2xy_type, decoder.out_type, kernel_size=1, padding=0, stride=1)
            side2xy_layers.append(side2xy)

        self.decoders_xy = nn.ModuleList(decoders)
        self.side2xy_decoders = nn.ModuleList(side2xy_layers)
        
        self.out_type_xy = FieldType(self.gs2d, [self.gs2d.regular_repr]*(self.lifting_dim//self.N))
        
        self.final_conv_xy = R2Conv(decoders[-1].out_type, self.out_type_xy, kernel_size=1, padding=0, stride=1)
        ###### xy unet ######
        
    def build_sideunet(self):
        ##### side unet #####
        out_channels_side = self.lifting_dim
        in_channels_side = self.hidden_type.size
        merge_mode='concat'
        up_mode='transpose'
        self.start_filts = 32

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(self.depth):
            ins = in_channels_side if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i > 0 else False

            down_conv = DownConv(ins, outs, pooling=pooling,deform=self.deform)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(self.depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode,deform=self.deform)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, out_channels_side)        
        
                
    
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
        
        fea_plane = c.permute(0,2,1).reshape(-1, self.lifting_dim, self.reso_plane, self.reso_plane, self.reso_plane)
        # fea['xy'] = fea_plane.mean(-1)
        # fea['xz'] = fea_plane.mean(-2)
        # fea['yz']  = fea_plane.mean(-3)
        tri_feature = torch.stack((fea_plane.mean(-1),fea_plane.mean(-2),fea_plane.mean(-3)), dim=0)
        
        encoders_features = []
        
        for i in range(len(self.encoders_xy)):
            xy_feature = self.encoders_xy[i](tri_feature[0]).tensor
            xz_feature = self.down_convs[i](tri_feature[1])
            yz_feature = self.down_convs[i](tri_feature[2])
            
            add_xy = (xz_feature.mean(-1)[:,:,:,None] + yz_feature.mean(-1)[:,:,None,:])*(1-self.test_eq_flag) 
            xy_feature = GeometricTensor(torch.cat((xy_feature, add_xy), dim=1),self.side2xy_encoders[i].in_type)
            xy_feature = self.side2xy_encoders[i](xy_feature).tensor
            
            tri_feature = torch.stack((xy_feature, xz_feature, yz_feature), dim=0)
            
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, tri_feature)
            
        
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        
        # decoder part
        for i, (decoder_xy, encoder_features) in enumerate(zip(self.decoders_xy, encoders_features)):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            xy_feature = decoder_xy(encoder_features[0], tri_feature[0]).tensor
            
            xz_feature = self.up_convs[i](encoder_features[1], tri_feature[1])
            yz_feature = self.up_convs[i](encoder_features[2], tri_feature[2])
            
            add_xy = (xz_feature.mean(-1)[:,:,:,None] + yz_feature.mean(-1)[:,:,None,:])*(1-self.test_eq_flag) 
            xy_feature = GeometricTensor(torch.cat((xy_feature, add_xy), dim=1),self.side2xy_decoders[i].in_type)
            xy_feature = self.side2xy_decoders[i](xy_feature).tensor
            
            tri_feature = torch.stack((xy_feature, xz_feature,yz_feature), dim=0)

        fea['xy'] = self.final_conv_xy(GeometricTensor(tri_feature[0], self.final_conv_xy.in_type))
        fea['xz'] = self.conv_final(tri_feature[1]) # yz before. although it is a bug before, it performs well
        fea['yz'] = self.conv_final(tri_feature[2])
        
        if self.test_eq_flag == 0:
            fea['xy'] = fea['xy'].tensor.permute(0,1,3,2)
        fea['xz'] = fea['xz'].permute(0,1,3,2) * (1-self.test_eq_flag) 
        fea['yz'] = fea['yz'].permute(0,1,3,2) * (1-self.test_eq_flag)  # from pixel coord to real coord
        
        # tri_feature = torch.stack((fea['xy'], fea['xz'], fea['yz']), dim=1)
        return fea

    
    def test_eq(self):
        input_ = torch.randn(1, 1, 40, 40, 40)#.cuda()
        # self.test_eq_flag = 1
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
            # vector = plane['vector']
            transform_plane = self.forward(input_transformed)
            transform_features = transform_plane['xy']
            # transform_vector = transform_plane['vector']
            
            inv_feature = plane['yz']
            if g.value == 1:
                inv_feature_transformed = transform_plane['xz']
            elif g.value == 2:
                inv_feature_transformed = transform_plane['yz'].flip(3)
            else:
                inv_feature_transformed = transform_plane['xz'].flip(3)
            

            features_transformed_after = features.transform(g)
            print('xy equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-4, rtol=1e-4) else 'NO'))
            # print('equi test:  ' + ('YES' if torch.allclose(xz_feature, yz_feature_transformed, atol=1e-4, rtol=1e-4) else 'NO'))
            print('equi test:  ' + ('YES' if torch.allclose(inv_feature, inv_feature_transformed, atol=1e-1, rtol=1e-1) else 'NO'))
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
    
    def export(self,):
        model = ExportJointTriUNet()
        model.lifting = self.lifting.export()
        encoder_xy = []
        side2xy_encoders = []
        for i in range(len(self.encoders_xy)):
            encoder_xy.append(self.encoders_xy[i].export())
            side2xy_encoders.append(self.side2xy_encoders[i].export())
        model.encoders_xy = nn.ModuleList(encoder_xy)
        model.side2xy_encoders = nn.ModuleList(side2xy_encoders)
        model.down_convs = self.down_convs
        decoder_xy = []
        side2xy_decoders = []
        for i in range(len(self.decoders_xy)):
            decoder_xy.append(self.decoders_xy[i].export())
            side2xy_decoders.append(self.side2xy_decoders[i].export())
        model.decoders_xy = nn.ModuleList(decoder_xy)
        model.side2xy_decoders = nn.ModuleList(side2xy_decoders)
        model.up_convs = self.up_convs
        model.final_conv_xy = self.final_conv_xy.export()
        model.conv_final = self.conv_final
        model.lifting_dim = self.lifting_dim
        model.reso_plane = self.reso_plane
        model.padding = self.padding
        model.plane_type = self.plane_type
        return model
    
    def test_export(self,):
        export_model = self.export()
        input_ = torch.randn(8, 1, 40, 40, 40)#.cuda()
        plane = self.forward(input_)
        export_plane = export_model(input_)
        for key in plane.keys():
            if key == 'vector':
                continue
            print(key + ' export test:  ' + ('YES' if torch.allclose(plane[key], export_plane[key], atol=1e-4, rtol=1e-4) else 'NO'))



if __name__ == '__main__':
    # check_eq3d()
    # build the SE(3) equivariant model
    # m = CyclicUnet2d(in_channel=1, out_channel=32, hidden_dim=32, depth=3)
    index = torch.arange(0,1600, 1).reshape(40,40)
    m = JointTriUNet(in_channel=1, hidden_dim=32, N=4)
    # m.test_export()
    m.test_eq()
