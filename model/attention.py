"""Equivariant attention layers used by the induced IGD model."""

import numpy as np
import torch
from torch import nn, einsum
import escnn.nn as enn
from escnn.group import CyclicGroup
from escnn.gspaces import no_base_space

from model.decoder import BilinearSampler
from model.export_layers import ExportEquiDeformableAttn, ExportEquiGraspSO3DeformableAttn
from utils.transform import rotation_6d_to_matrix


def constant_init(module, value, bias=0):
    nn.init.constant_(module.weight, value)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


class EquiDeformableAttn2(nn.Module):
    def __init__(self, in_type, hidden_type=None, num_samples=8):
        super().__init__()
        self.in_type = in_type
        self.gs = no_base_space(CyclicGroup(4))
        self.num_samples = num_samples
        coordinate_type = enn.FieldType(self.gs, num_samples * [self.gs.fibergroup.irrep(1), self.gs.fibergroup.irrep(0)])
        weight_type = enn.FieldType(self.gs, num_samples * [self.gs.fibergroup.irrep(0)])
        if hidden_type is None:
            hidden_type = enn.FieldType(self.gs, [self.gs.regular_repr] * (in_type.size // 4))
        self.to_v = enn.Linear(in_type, hidden_type)
        self.to_out = enn.Linear(hidden_type, in_type)
        self.scale = hidden_type.size ** -0.5
        self.to_offset = enn.Linear(in_type, coordinate_type, bias=True)
        self.to_weight = enn.Linear(in_type, weight_type, bias=True)
        self.sample_points = None
        self.feature_sampler = BilinearSampler()
        self.pe = None

    def forward(self, query_pos, planes):
        batch_size, sample_size, _ = query_pos.shape
        feature = enn.GeometricTensor(self.feature_sampler(query_pos, planes).reshape(batch_size * sample_size, -1), self.in_type)
        offsets = self.to_offset(feature).tensor.reshape(batch_size, sample_size, -1, 3)
        weights = self.to_weight(feature).tensor.reshape(batch_size * sample_size, self.num_samples, 1)
        sample_points = offsets + query_pos.reshape(batch_size, sample_size, 1, 3)
        self.sample_points = sample_points
        sampled = self.feature_sampler(sample_points.reshape(batch_size, -1, 3), planes).reshape(batch_size * sample_size * self.num_samples, self.in_type.size)
        values = self.to_v(enn.GeometricTensor(sampled, self.in_type)).tensor.reshape(batch_size * sample_size, self.num_samples, -1)
        output = (values * weights).sum(-2)
        output = self.to_out(enn.GeometricTensor(output.reshape(-1, self.to_out.in_type.size), self.to_out.in_type))
        return (output + feature).tensor.reshape(batch_size, sample_size, -1)

    def export(self):
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


class EquiGraspSO3DeformableAttn(nn.Module):
    def __init__(self, in_type, hidden_size=128, num_heads=1, zero_offset=False, fixed_control_points=False):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.num_heads = num_heads
        self.feature_dim = in_type.size
        self.gs = no_base_space(CyclicGroup(4))
        self.feature_sampler = BilinearSampler()
        self.hidden_type = hidden_size if isinstance(hidden_size, enn.FieldType) else enn.FieldType(self.gs, [self.gs.regular_repr] * (hidden_size // 4))
        self.to_q = enn.Linear(in_type, self.hidden_type)
        self.to_k = enn.Linear(in_type, self.hidden_type)
        self.to_v = enn.Linear(in_type, self.hidden_type)
        self.to_out = enn.Linear(self.hidden_type, in_type)
        self.scale = self.hidden_type.size ** -0.5
        palm_y = np.linspace(-0.04, 0.04, 5)
        palm_z = np.linspace(0.0, 0.05, 5)
        control_points = np.stack(np.meshgrid(np.zeros(1), palm_y, palm_z), axis=-1).reshape(-1, 3)
        self.ncp = control_points.shape[0]
        self.pe_k = nn.Parameter(torch.rand(self.ncp, self.hidden_type.size))
        self.pe_q = nn.Parameter(torch.rand(1, self.hidden_type.size))
        control_points = torch.from_numpy(control_points.astype("float32")) / 0.3
        if fixed_control_points:
            self.register_buffer("control_points", control_points)
        else:
            self.control_points = nn.Parameter(control_points)
        self.zero_offset = zero_offset
        self.to_offset = nn.Linear(self.feature_dim, num_heads * 3, bias=False)
        constant_init(self.to_offset, 0.0)

    def forward(self, query_pos, planes):
        batch_size, sample_size, _ = query_pos.shape
        orientation = query_pos[..., 3:]
        query_pos = query_pos[..., :3]
        feature = self.feature_sampler(query_pos, planes).reshape(batch_size, sample_size, self.feature_dim)
        control_points = self.control_points.repeat(batch_size * sample_size, 1, 1).reshape(batch_size, sample_size, -1, 3)
        control_points = torch.einsum("bnpd,bngd->bngp", rotation_6d_to_matrix(orientation).reshape(batch_size, sample_size, 3, 3), control_points)
        anchor_points = query_pos.unsqueeze(2) + control_points
        if self.zero_offset:
            sampled = self.feature_sampler(anchor_points.reshape(batch_size, -1, 3), planes).reshape(batch_size * sample_size, -1, 1, self.feature_dim).repeat(1, 1, self.num_heads, 1)
        else:
            offsets = self.to_offset(feature).reshape(batch_size, sample_size, self.num_heads, 3)
            sample_points = offsets.unsqueeze(3) + anchor_points.reshape(batch_size, sample_size, 1, -1, 3)
            sampled = self.feature_sampler(sample_points.reshape(batch_size, -1, 3), planes).reshape(batch_size * sample_size, -1, self.num_heads, self.feature_dim)
        sampled = enn.GeometricTensor(sampled.reshape(-1, self.feature_dim), self.in_type)
        feature = enn.GeometricTensor(feature.reshape(-1, self.feature_dim), self.in_type)
        keys = self.to_k(sampled).tensor.reshape(batch_size * sample_size, -1, self.num_heads, self.hidden_type.size).transpose(1, 2)
        values = self.to_v(sampled).tensor.reshape(batch_size * sample_size, -1, self.num_heads, self.hidden_type.size).transpose(1, 2)
        queries = self.to_q(feature).tensor.reshape(batch_size * sample_size, 1, self.hidden_type.size).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        queries = (queries + self.pe_q.reshape(1, 1, 1, -1)) / self.scale
        keys = keys + self.pe_k.reshape(1, 1, self.ncp, -1)
        attention = einsum("b n i d, b n j d -> b n i j", queries, keys)
        attention = (attention - attention.amax(dim=-1, keepdim=True).detach()).softmax(dim=-1)
        output = einsum("b n i j, b n j d -> b n i d", attention, values).transpose(1, 2).reshape(batch_size * sample_size, -1)
        output = self.to_out(enn.GeometricTensor(output, self.hidden_type))
        return (output + feature).tensor.reshape(batch_size, sample_size, -1)

    def export(self):
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
