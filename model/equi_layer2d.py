
from typing import Tuple, Union
from collections import defaultdict
import os
import sys
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *


import torch
from torch import nn
import numpy as np

from scipy import stats
import time
from model.equi_layer3d import EquiSiLU
from model.equi_deform_conv import EquiScaleConv2d,EquiDeformConv2d
from model.export_layers import *

class EquiResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, in_type, out_type=None, hidden_type=None, normalize=False):
        super().__init__()
        # Attributes
        self.in_type = in_type
        self.gs = self.in_type.gspace
        if out_type is None:
            self.out_type = in_type

        if hidden_type is None:
            self.hidden_type = self.out_type 

        self.normalize = normalize
        # Submodules
        bias = not normalize
        self.fc_0 = Linear(self.in_type, self.hidden_type, bias=bias)
        self.fc_1 = Linear(self.hidden_type, self.out_type, bias=bias)
        self.actvn0 = ReLU(self.in_type, inplace=True)
        self.actvn1 = ReLU(self.hidden_type, inplace=True)
        # self.actvn = nn.SiLU()
        # if self.normalize:
        #     self.norm_0 = enn.BatchNorm1d(size_h)
        #     self.norm_1 = enn.BatchNorm1d(size_out)
        # else:
        #     self.norm_0 = nn.Identity()
        #     self.norm_1 = nn.Identity()

        if self.in_type == self.out_type:
            self.shortcut = None
        else:
            self.shortcut = Linear(self.in_type, self.out_type, bias=False)

    def forward(self, x):
        net = self.fc_0(self.actvn0(x))
        dx = self.fc_1(self.actvn1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
    
    def export(self,):
        model = EquiResnetBlockFC(self.in_type)
        model.actvn0 = self.actvn0.export()
        model.actvn1 = self.actvn1.export()
        model.fc_0 = self.fc_0.export()
        model.fc_1 = self.fc_1.export()
        if self.shortcut is not None:
            model.shortcut = self.shortcut.export()
        return model

class Convblock2d(EquivariantModule):
    def __init__(self, in_type: FieldType, out_type: FieldType, kernel_size=3, stride = 1, padding = 1, activation=True, deform=True):
        super(Convblock2d, self).__init__()
        self.gs = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type
        # self.norm = InnerBatchNorm(out_type, affine=True)
        if activation is True:
            self.act = ReLU(out_type, inplace=True)
            # self.act = ReLU(out_type, inplace=True)
        else:
            self.act = None
        if deform:
            self.conv = EquiDeformConv2d(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, stride=stride, initialize=True, bias=True)
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = GeometricTensor(x, self.in_type)
        x = self.conv(x)
        # x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
        
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape
    
    def test_eq(self, input):
        for g in self.gs.testing_elements:
            # permutation = [1, 2, 0]
            permutation = [2, 0, 1]
            # permutation = [0,1,2]
            input_transformed = input.transform(g)

            features = self.forward(input)
            transform_features = self.forward(input_transformed)

            features_transformed_after = features.transform(g)
            print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-5, rtol=1e-4) else 'NO'))
        return

    def export(self, ):
        if self.act is not None:
            model = nn.Sequential(
                    self.conv.export(),
                    nn.ReLU(inplace=True)
            )
            return model
        else:
            return self.conv.export()
        
    

class CyclicRes2Block2d(EquivariantModule):

    def __init__(self, in_channel, out_channel, N, scale=4, stride: int = 1):

        super(CyclicRes2Block2d, self).__init__()
        assert out_channel % (N*scale) == 0
        self.gs = rot2dOnR2(N)
        if in_channel == 1:
            self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        else:
            self.in_type = FieldType(self.gs, [self.gs.regular_repr]*(in_channel//N))
        self.out_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//N))
        self.hidden_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//(N*scale)))
        self.width = out_channel//(scale)
        self.scale = scale
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        self.conv1 = Convblock2d(self.in_type, self.out_type, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Convblock2d(self.out_type, self.out_type, kernel_size=1, stride=stride, padding=0)
        multi_scale_conv = []
        for i in range(self.nums):
            multi_scale_conv.append(Convblock2d(self.hidden_type, self.hidden_type, kernel_size=3, stride=stride, padding=1))
        self.multi_scale_conv = nn.ModuleList(multi_scale_conv)

        self.conv3 = Convblock2d(self.out_type, self.out_type, kernel_size=3, stride=stride, padding=1, activation=False)

        self.non_linearity = ReLU(self.out_type, inplace=True)


    def forward(self, input: GeometricTensor):
        if isinstance(input, torch.Tensor):
            input = GeometricTensor(input, self.in_type)
        assert input.type == self.in_type
        
        out = self.conv1(input)
        residual = out
        
        out = self.conv2(out)
        spx = torch.split(out.tensor, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.multi_scale_conv[i](sp).tensor
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        
        out += residual
        out = self.non_linearity(out)
        
        return out

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape
        

class CyclicResBlock2d(EquivariantModule):

    def __init__(self, in_channel, out_channel, N, stride: int = 1,deform=True):

        super(CyclicResBlock2d, self).__init__()
        self.gs = rot2dOnR2(N)
        if in_channel == 1:
            self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        else:
            self.in_type = FieldType(self.gs, [self.gs.regular_repr]*(in_channel//N))
        self.out_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//N))

        self.conv1 = Convblock2d(self.in_type, self.out_type, kernel_size=3, stride=stride, padding=1,deform=deform)
        self.conv2 = Convblock2d(self.out_type, self.out_type, kernel_size=3, stride=stride, padding=1,deform=deform)
        self.conv3 = Convblock2d(self.out_type, self.out_type, kernel_size=3, stride=stride, padding=1, activation=False,deform=deform)

        self.non_linearity = ReLU(self.out_type, inplace=True)


    def forward(self, input: GeometricTensor):
        if isinstance(input, torch.Tensor):
            input = GeometricTensor(input, self.in_type)
        assert input.type == self.in_type
        
        out = self.conv1(input)
        residual = out
        
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        out = self.non_linearity(out)
        
        return out

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape
    
    
    def test_eq(self, input):
        for g in self.gs.testing_elements:
            # permutation = [1, 2, 0]
            permutation = [2, 0, 1]
            # permutation = [0,1,2]
            input_transformed = input.transform(g)

            features = self.forward(input)
            transform_features = self.forward(input_transformed)

            features_transformed_after = features.transform(g)
            print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-5, rtol=1e-4) else 'NO'))
        return

    def export(self, ):
        model = ExportResBlock2d()
        model.conv1 = self.conv1.export()
        model.conv2 = self.conv2.export()
        model.conv3 = self.conv3.export()
        model.non_linearity = nn.ReLU(inplace=True)
        return model


class EquiDownConv2d(nn.Module):
    def __init__(self,
                in_channel,
                out_channel,
                N,
                kernel_size: int = 3,
                stride: int = 1, 
                pooling=True,
                initialize=False,
                deform=True):
        super(EquiDownConv2d, self).__init__()
        self.pooling = pooling        
        # self.conv = CyclicRes2Block2d(in_channel, out_channel, N, stride=stride)
        self.conv = CyclicResBlock2d(in_channel, out_channel, N, stride=stride,deform=deform)
        self.in_type = self.conv.in_type
        self.out_type = self.conv.out_type
        self.gs = self.in_type.gspace

        if self.pooling:
            self.pool = PointwiseAvgPool2D(self.in_type, kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = GeometricTensor(x, self.in_type)
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv(x)
        return x
    
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape
    
    def test_eq(self, input):
        for g in self.gs.testing_elements:
            permutation = [2, 0, 1]
            input_transformed = input.transform(g)

            features = self.forward(input)
            transform_features = self.forward(input_transformed)

            features_transformed_after = features.transform(g)
            print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-5, rtol=1e-4) else 'NO'))
        return
    def export(self, ):
        model = ExportDownConv2d()
        if self.pool is not None:
            model.pool = nn.AvgPool2d(kernel_size=self.pool.kernel_size, stride=self.pool.stride)
        model.conv = self.conv.export()
        return model



class EquiUpConv2d(torch.nn.Module):
    def __init__(self, 
                in_channel,
                out_channel, 
                N,
                kernel_size: int = 3,
                stride: int = 1, 
                merge_mode='concat', 
                up_mode='bilinear', # or nearest
                initialize=True,
                deform=True): 
        super(EquiUpConv2d, self).__init__()
        self.merge_mode = merge_mode
        if self.merge_mode == 'concat':
            hidden_channel = out_channel * 2
        else:
            hidden_channel = out_channel
        self.conv = CyclicResBlock2d(hidden_channel, out_channel, N, stride,deform=deform)
        # self.conv = CyclicRes2Block2d(hidden_channel, out_channel, N, stride)
        self.in_type = FieldType(self.conv.gs, [self.conv.gs.regular_repr]*(in_channel//N))
        self.out_type = self.conv.out_type
        
        # if up_mode == 'transpose':
        #     self.upconv = R3ConvTransposed(self.in_type, self.in_type, kernel_size=3, stride=1, padding=0, initialize=initialize)
        # else:
        self.upconv = SequentialModule(
            R2Upsampling(self.in_type, scale_factor=2, mode=up_mode, align_corners=True),
            R2Conv(self.in_type, self.out_type, kernel_size=3, stride=1, padding=1, initialize=initialize),
        )
    
    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        if isinstance(from_up, torch.Tensor):
            from_up = GeometricTensor(from_up, self.in_type)
        if isinstance(from_down, torch.Tensor):
            from_down = GeometricTensor(from_down, self.out_type)  
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down
        elif self.merge_mode == 'concat':
            x = torch.cat((from_up.tensor, from_down.tensor), dim=1)
            
        x = self.conv(x)
        return x
    
    def export(self, ):
        model = ExportUpConv2d()
        model.merge_mode = self.merge_mode
        model.conv = self.conv.export()
        model.upconv = self.upconv.export()
        return model



if __name__=="__main__":
    gs = rot2dOnR2(4)
    in_type = FieldType(gs, [gs.regular_repr] )
    hidden_type = FieldType(gs, [gs.regular_repr]* (32//4))
    # conv = nn.Sequential(
    #     R2Conv(in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True),
    #     ELU(hidden_type, inplace=True),
    #         EquiDownConv2d(in_channel=32, out_channel=32, N=4, pooling=False),
    #                     EquiDownConv2d(in_channel=32, out_channel=64, N=4),
    #                         EquiDownConv2d(in_channel=64, out_channel=128, N=4))
    conv = CyclicResBlock2d(in_channel=4, out_channel=32, N=4)
    # normal_conv = conv.export()
                            
    
    input = torch.rand(8,4,40,40)
    input = GeometricTensor(input, in_type)
    
    
    # features = conv(input)
    # export_features = normal_conv(input.tensor)
    
    # print('export test:  ' + ('YES' if torch.allclose(features.tensor,export_features, atol=1e-5, rtol=1e-4) else 'NO'))
    
    
    
    for g in gs.testing_elements:
        input_transformed = input.transform(g)

        features = conv(input)
        transform_features = conv(input_transformed)

        features_transformed_after = features.transform(g)
        print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-5, rtol=1e-4) else 'NO'))
    
 