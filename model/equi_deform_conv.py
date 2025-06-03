import torchvision.ops as ops
import torch.nn as nn
import escnn.nn as enn
import torch
from escnn import gspaces
import torch.nn.functional as F
from model.export_layers import *


class EquiDeformConv2d(nn.Module):
    def __init__(self, in_type: enn.FieldType, out_type: enn.FieldType, kernel_size=3, stride = 1, padding = 1):
        super(EquiDeformConv2d, self).__init__()
        assert kernel_size == 3
        self.gs = in_type.gspace
        self.in_type = in_type
        self.N = self.gs.rotations_order
        self.out_type = out_type
        self.kernel_size = kernel_size
        offset_type = enn.FieldType(self.gs, [self.gs.irrep(1),self.gs.irrep(0),self.gs.irrep(0),self.gs.irrep(0)])
        # offset_type = enn.FieldType(self.gs, [self.gs.irrep(1),self.gs.irrep(0),self.gs.irrep(0),self.gs.irrep(0),self.gs.irrep(0)])
        # modulator_type = enn.FieldType(self.gs, ((kernel_size//2) * 2 +1)* [self.gs.irrep(0)])
        # self.modulator_conv = enn.R2Conv(in_type, modulator_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.conv_offset = enn.R2Conv(in_type, offset_type, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.conv = enn.R2Conv(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.conv2 = nn.Conv2d(in_type.size, out_type.size, kernel_size=kernel_size, stride=stride, padding=padding)
            
        base_offset = torch.tensor([[[[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]], [[2, 0], [2, 1], [2, 2]]]]).float()-1
        # base_offset = torch.tensor([[[[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]], [[0, 2], [1, 2], [2, 2]]]]).float()-1
        mask1 = torch.tensor([True, False, True, False, False, False, True, False, True]).unsqueeze(1).repeat(1,2).reshape(-1)
        mask2 = torch.tensor([False, True, False, True, False, True, False, True, False]).unsqueeze(1).repeat(1,2).reshape(-1)
        self.register_buffer('mask1', mask1)
        self.register_buffer('mask2', mask2)
        base_offset = base_offset.reshape(1, kernel_size**2 * 2, 1, 1)
        self.register_buffer('base_offset', base_offset)
        
        # self.out_type2 = enn.FieldType(self.gs, self.out_type.size * [self.gs.trivial_repr])
        return
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = enn.GeometricTensor(x, self.in_type)
        offset = self.conv_offset(x).tensor

        bs, _, h, w = offset.shape
        scale1 = F.relu(offset[:, 2:3]) + 1
        scale2 = F.relu(offset[:, 3:4]) + 1
        shift = offset[:, :2].repeat(1, 9, 1, 1)

        
        
        _filter, _bias = self.conv.expand_parameters()
        offset = torch.zeros_like(shift)
        scaled_offset = torch.zeros_like(self.base_offset*scale1)
        scaled_offset[:,self.mask1]  = self.base_offset[:,self.mask1] * scale1
        scaled_offset[:,self.mask2]  = self.base_offset[:,self.mask2] * scale2
        
        offset =  (scaled_offset - self.base_offset)  + shift

        x = ops.deform_conv2d(x.tensor, offset, _filter, _bias, stride=self.conv.stride, padding=self.conv.padding)#, mask=modulator)
        x = enn.GeometricTensor(x, self.out_type)

        return x
    
    def export(self,):
        model = ExportEquiDeformConv()
        model.conv = self.conv.export()
        model.conv_offset = self.conv_offset.export()
        model.mask1 = self.mask1
        model.mask2 = self.mask2
        model.base_offset = self.base_offset
        return model


if __name__=="__main__":
    gs = gspaces.rot2dOnR2(4)
    input = torch.rand(8,16,40,40)
    in_type = enn.FieldType(gs, [gs.regular_repr] * (16//4))
    hidden_type = enn.FieldType(gs, [gs.regular_repr]* (32//4))
    input = enn.GeometricTensor(input, in_type)
    conv = EquiDeformConv2d(in_type, hidden_type, kernel_size=3, stride=1, padding=1)
    export_conv = conv.export()

    print('export test:  ' + ('YES' if torch.allclose(conv(input).tensor, export_conv(input.tensor), atol=1e-1, rtol=1e-1) else 'NO'))
    
    for g in gs.testing_elements:
        if g.value == 0:
            continue
        input_transformed = input.transform(g)

        features = conv(input)
        transform_features = conv(input_transformed)

        features_transformed_after = features.transform(g)
        print('equi test:  ' + ('YES' if torch.allclose(features_transformed_after.tensor, transform_features.tensor, atol=1e-1, rtol=1e-1) else 'NO'))
    
    print()
