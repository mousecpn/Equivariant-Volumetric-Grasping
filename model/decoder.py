"""Equivariant decoders used by the induced GIGA and IGD models."""

import torch
import torch.nn.functional as F
from torch import nn
import escnn.nn as enn
from escnn.group import CyclicGroup
from escnn.gspaces import no_base_space
from escnn.nn import FieldType, GeometricTensor

from model.equi_layer2d import EquiResnetBlockFC
from model.export_layers import ExportTimeEquiDecoder, ExportTriEquiDecoder
from model.pos_encoding import SinusoidalPosEmb
from utils.common import normalize_coordinate, normalize_3d_coordinate


class TriFullEquiDecoder(nn.Module):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None, inv_map=False, feature_sampler=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        self.feature_sampler = feature_sampler
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = "bilinear"
        self.inv_map = inv_map
        self.gs = no_base_space(CyclicGroup(N))
        self.hidden_type = hidden_size if isinstance(hidden_size, FieldType) else FieldType(self.gs, (hidden_size // N) * [self.gs.regular_repr])
        self.hidden_size = self.hidden_type.size
        self.lifting = enn.Linear(self.in_type, self.hidden_type)
        self.act = enn.ReLU(self.hidden_type, inplace=True)
        self.out_layer = enn.Linear(self.hidden_type, self.out_type)
        self.fc_c = nn.ModuleList([enn.Linear(self.hidden_type, self.hidden_type) for _ in range(n_blocks)])
        if concat_type is not None:
            self.fc_p = enn.Linear(concat_type, self.hidden_type)
        self.blocks = nn.ModuleList([EquiResnetBlockFC(self.hidden_type) for _ in range(n_blocks)])

    def query_feature(self, points, planes):
        if self.feature_sampler is not None:
            return self.feature_sampler(points, planes)
        if "grid" not in planes:
            raise ValueError("a feature sampler or grid feature is required")
        return self.sample_grid_feature(points, planes["grid"]).transpose(1, 2)

    def sample_grid_feature(self, points, feature):
        coordinates = normalize_3d_coordinate(points.clone(), padding=self.padding)[:, :, None, None].float()
        return F.grid_sample(feature, 2.0 * coordinates - 1.0, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)

    def forward(self, points, planes):
        batch_size, sample_size = points.shape[:2]
        features = self.query_feature(points, planes)
        valid_mask = ~torch.isnan(points).all(-1)
        features = GeometricTensor(features[valid_mask].reshape(-1, self.in_type.size), self.in_type)
        features = self.act(self.lifting(features))
        if self.concat_type is None:
            net = GeometricTensor(torch.zeros(features.shape[0], self.hidden_type.size, device=points.device), self.hidden_type)
        else:
            net = self.fc_p(GeometricTensor(points[valid_mask][..., 3:].reshape(-1, self.concat_type.size), self.concat_type))
        for layer, block in zip(self.fc_c, self.blocks):
            net = block(net + layer(features))
        output = self.out_layer(net).tensor
        result = torch.full((batch_size, sample_size, output.shape[-1]), float("nan"), device=output.device)
        result[valid_mask] = output
        return result.squeeze(-1)

    def export(self):
        model = ExportTriEquiDecoder()
        model.in_dim = self.in_type.size
        model.hidden_dim = self.hidden_type.size
        model.N = self.N
        model.lifting = self.lifting.export()
        model.feature_sampler = self.feature_sampler.export() if hasattr(self.feature_sampler, "export") else self.feature_sampler
        model.blocks = nn.ModuleList([block.export() for block in self.blocks])
        model.n_blocks = len(self.blocks)
        model.fc_c = nn.ModuleList([layer.export() for layer in self.fc_c])
        if self.concat_type is not None:
            model.concat_dim = self.concat_type.size
            model.fc_p = self.fc_p.export()
        model.out_layer = self.out_layer.export()
        return model


class TriTimeEquiDecoder(nn.Module):
    def __init__(self, in_type, out_type, N, dim=3, hidden_size=32, n_blocks=5, concat_type=None, feature_sampler=None, t_dim=16):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.N = N
        self.dim = dim
        self.n_blocks = n_blocks
        self.concat_type = concat_type
        self.padding = 0.0
        self.sample_mode = "bilinear"
        self.gs = no_base_space(CyclicGroup(N))
        self.t_type = FieldType(self.gs, t_dim * [self.gs.trivial_repr])
        self.feature_sampler = feature_sampler
        self.hidden_type = hidden_size if isinstance(hidden_size, FieldType) else FieldType(self.gs, (hidden_size // N) * [self.gs.regular_repr])
        self.hidden_size = self.hidden_type.size
        self.lifting = enn.Linear(self.in_type, self.hidden_type)
        self.act = enn.ReLU(self.hidden_type, inplace=True)
        self.out_layer = enn.Linear(self.hidden_type, self.out_type)
        self.fc_c = nn.ModuleList([enn.Linear(self.hidden_type + self.t_type, self.hidden_type) for _ in range(n_blocks)])
        self.fc_p = enn.Linear(self.out_type, self.hidden_type)
        self.blocks = nn.ModuleList([EquiResnetBlockFC(self.hidden_type) for _ in range(n_blocks)])
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(t_dim), nn.Linear(t_dim, t_dim * 2), nn.Mish(), nn.Linear(t_dim * 2, t_dim))

    def query_feature(self, points, planes):
        return self.feature_sampler(points, planes)

    def forward(self, data, planes, time):
        features = self.query_feature(data, planes) if isinstance(planes, dict) else planes
        batch_size, sample_size = data.shape[:2]
        if time.ndim == 1:
            time_features = self.time_mlp(time.reshape(-1)).reshape(batch_size, 1, -1).repeat(1, sample_size, 1).reshape(batch_size * sample_size, self.t_type.size)
        else:
            time_features = self.time_mlp(time.reshape(-1)).reshape(batch_size * sample_size, -1)
        features = self.act(self.lifting(GeometricTensor(features.reshape(-1, self.in_type.size), self.in_type)))
        net = self.fc_p(GeometricTensor(data.reshape(batch_size * sample_size, -1), self.out_type))
        conditioned = GeometricTensor(torch.cat((features.tensor, time_features), dim=-1), self.hidden_type + self.t_type)
        for layer, block in zip(self.fc_c, self.blocks):
            net = block(net + layer(conditioned))
        return self.out_layer(net).tensor.reshape(batch_size, sample_size, -1).squeeze(-1)

    def export(self):
        model = ExportTimeEquiDecoder()
        model.in_dim = self.in_type.size
        model.hidden_dim = self.hidden_type.size
        model.N = self.N
        model.lifting = self.lifting.export()
        model.blocks = nn.ModuleList([block.export() for block in self.blocks])
        model.n_blocks = len(self.blocks)
        model.fc_c = nn.ModuleList([layer.export() for layer in self.fc_c])
        model.fc_p = self.fc_p.export()
        model.out_layer = self.out_layer.export()
        model.time_mlp = self.time_mlp
        model.t_dim = self.t_type.size
        model.feature_sampler = self.feature_sampler.export() if hasattr(self.feature_sampler, "export") else self.feature_sampler
        return model


class BilinearSampler(nn.Module):
    """Legacy sampler retained because the induced attention defaults to it."""

    def __init__(self, c_dim=128, sample_mode="bilinear", padding=0.0, concat_feat=True, plane_type=("xy", "xz", "yz")):
        super().__init__()
        self.c_dim = c_dim
        self.sample_mode = sample_mode
        self.padding = padding
        self.concat_feat = concat_feat
        self.plane_type = plane_type

    def sample_plane_feature(self, points, feature, plane):
        coordinates = normalize_coordinate(points.clone(), plane=plane, padding=self.padding)[:, :, None].float()
        return F.grid_sample(feature, 2.0 * coordinates - 1.0, padding_mode="border", align_corners=True, mode=self.sample_mode).squeeze(-1)

    def forward(self, points, planes):
        features = [self.sample_plane_feature(points, planes[plane], plane) for plane in self.plane_type if plane in planes]
        return torch.cat(features, dim=1).transpose(1, 2)
