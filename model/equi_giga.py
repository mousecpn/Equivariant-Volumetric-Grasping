"""Induced-representation equivariant GIGA model."""

import torch
from torch import nn
from escnn.group import CyclicGroup
from escnn.gspaces import no_base_space
from escnn.nn import FieldType

from model.decoder import TriFullEquiDecoder
from model.triunet_induced import InducedBilinearSampler, JointTriUNetInduced
from utils.transform import irreps2rot, matrix_to_quaternion, rotation_6d_to_matrix


class TriEquiGIGAInduce(nn.Module):
    """C4-equivariant GIGA with an induced XZ/YZ tri-plane representation."""

    def __init__(self, N=4, s2tp=True, share_plane=False):
        super().__init__()
        if N != 4:
            raise ValueError("TriEquiGIGAInduce currently supports N=4 only")

        self.N = N
        hidden_dim = 32
        self.gs = no_base_space(CyclicGroup(N))
        self.backbone = JointTriUNetInduced(
            in_channel=1,
            hidden_dim=hidden_dim,
            depth=3,
            N=N,
            plane_resolution=40,
            s2tp=s2tp,
            share_plane_unet=share_plane,
        )
        decoder_in_type = FieldType(
            self.gs, [self.gs.regular_repr] * (3 * hidden_dim // N)
        )
        scalar_type = FieldType(self.gs, [self.gs.irrep(0)])
        rotation_type = FieldType(self.gs, [self.gs.irrep(1)] * 3)

        self.decoder_qual = TriFullEquiDecoder(
            decoder_in_type, scalar_type, N=N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_width = TriFullEquiDecoder(
            decoder_in_type, scalar_type, N=N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_rot = TriFullEquiDecoder(
            decoder_in_type, rotation_type, N=N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_tsdf = TriFullEquiDecoder(
            decoder_in_type, scalar_type, N=N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )

    def forward(self, inputs, points, target=None, p_tsdf=None):
        self.batch_size, self.sample_num = points.shape[:2]
        features = self.backbone(inputs)
        quality = self.decoder_qual(points, features)
        width = self.decoder_width(points, features)
        rotation = matrix_to_quaternion(
            rotation_6d_to_matrix(irreps2rot(self.decoder_rot(points, features)))
        )
        if p_tsdf is None:
            return quality, rotation, width
        return quality, rotation, width, self.decoder_tsdf(p_tsdf, features)

    def export(self):
        self.backbone = self.backbone.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_width = self.decoder_width.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        return self
