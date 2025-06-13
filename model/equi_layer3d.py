
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *


import torch
from torch import nn
import numpy as np

from scipy import stats
import time
import torch.nn.functional as F

class Convblock(EquivariantModule):
    def __init__(self, in_type: FieldType, out_type: FieldType, kernel_size=3, stride = 1, padding = 1, activation=True):
        super(Convblock, self).__init__()
        self.gs = in_type.gspace
        self.in_type = in_type
        self.out_type = out_type
        # self.norm = IIDBatchNorm3d(out_type, affine=True)
        if activation is True:
            self.act = ELU(out_type, inplace=True)
        else:
            self.act = None
        self.conv = R3Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, stride=stride, initialize=True, bias=True)
    
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
        
 

class CyclicResBlock(EquivariantModule):

    def __init__(self, in_channel, out_channel, N, stride: int = 1):

        super(CyclicResBlock, self).__init__()
        self.gs = rot2dOnR3(N)
        if in_channel == 1:
            self.in_type = FieldType(self.gs, [self.gs.trivial_repr])
        else:
            self.in_type = FieldType(self.gs, [self.gs.regular_repr]*(in_channel//N))
        self.out_type = FieldType(self.gs, [self.gs.regular_repr]*(out_channel//N))

        self.conv1 = Convblock(self.in_type, self.out_type, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Convblock(self.out_type, self.out_type, kernel_size=3, stride=stride, padding=1)
        self.conv3 = Convblock(self.out_type, self.out_type, kernel_size=3, stride=stride, padding=1, activation=False)

        self.non_linearity = ELU(self.out_type, inplace=True)


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
        

class EquiDownConv(nn.Module):
    def __init__(self,
                in_channel,
                out_channel,
                N,
                kernel_size: int = 3,
                stride: int = 1, 
                pooling=True,
                initialize=False):
        super(EquiDownConv, self).__init__()
        self.pooling = pooling        
        self.conv = CyclicResBlock(in_channel, out_channel, N, stride=stride)
        self.in_type = self.conv.in_type
        self.out_type = self.conv.out_type
        self.gs = self.in_type.gspace

        if self.pooling:
            self.pool = PointwiseAvgPool3D(self.in_type, kernel_size=2, stride=2)
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


class EquiUpConv(torch.nn.Module):
    def __init__(self, 
                in_channel,
                out_channel, 
                N,
                kernel_size: int = 3,
                stride: int = 1, 
                merge_mode='add', 
                up_mode='trilinear', # or nearest
                initialize=True): 
        super(EquiUpConv, self).__init__()
        self.merge_mode = merge_mode
        assert self.merge_mode == 'add' # "concat" not implemented yet

        self.conv = CyclicResBlock(out_channel, out_channel, N, stride)
        self.in_type = FieldType(self.conv.gs, [self.conv.gs.regular_repr]*(in_channel//N))
        self.out_type = self.conv.out_type
        
        # if up_mode == 'transpose':
        #     self.upconv = R3ConvTransposed(self.in_type, self.in_type, kernel_size=3, stride=1, padding=0, initialize=initialize)
        # else:
        self.upconv = SequentialModule(
            R3Upsampling(self.in_type, scale_factor=2, mode=up_mode, align_corners=True),
            R3Conv(self.in_type, self.out_type, kernel_size=3, stride=1, padding=1, initialize=initialize),
        )
    
    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down
        x = self.conv(x)
        return x


class EquiSiLU(EquivariantModule):
    
    def __init__(self, in_type: FieldType, inplace: bool = False):
        assert isinstance(in_type.gspace, GSpace)
        
        super(EquiSiLU, self).__init__()
        
        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)
        
        self.space = in_type.gspace
        self.in_type = in_type
        
        # the representation in input is preserved
        self.out_type = in_type
        
        self._inplace = inplace
    
    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies ELU function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after elu has been applied

        """
        
        assert input.type == self.in_type
        return GeometricTensor(
            F.silu(input.tensor, inplace=self._inplace),
            self.out_type, input.coords
        )

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        assert len(input_shape) >= 2
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        spatial_shape = input_shape[2:]

        return (b, self.out_type.size, *spatial_shape)


if __name__=="__main__":
    gs = rot2dOnR3(4)
    in_type = FieldType(gs, [gs.trivial_repr])
    hidden_type = FieldType(gs, [gs.regular_repr]* (32//4))
    conv = nn.Sequential(
        R3Conv(in_type, hidden_type, kernel_size=3, padding=1, stride=1, initialize=True),
        ELU(hidden_type, inplace=True),
            EquiDownConv(in_channel=32, out_channel=32, N=4, pooling=False),
                        EquiDownConv(in_channel=32, out_channel=64, N=4),
                            EquiDownConv(in_channel=64, out_channel=128, N=4))
    # conv = EquiDownConv(in_channel=32, out_channel=32, N=4)
                            
    
    input = torch.rand(8,1,40,40,40)
    input = GeometricTensor(input, in_type)
    
    for g in gs.testing_elements:
        input_transformed = input.transform(g)

        features = conv(input)
        transform_features = conv(input_transformed)

        features_transformed_after = features.transform(g)
        print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-4, rtol=1e-4) else 'NO'))
    
 