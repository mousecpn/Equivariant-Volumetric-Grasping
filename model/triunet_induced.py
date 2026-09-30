"""A fully C4-equivariant tri-plane UNet using an induced side-plane type.

The vertical planes are not converted to invariant scalars.  Instead, one
shared C2-reflection-equivariant UNet processes XZ and YZ.  The ordered pair
of its outputs carries ``Ind_{C2}^{C4}``; with regular C2 fields this induced
type is (a sum of) regular C4 fields.

The returned XZ/YZ grids are in two different, canonical fibre gauges.  Use
``InducedBilinearSampler`` (rather than the repository's ``BilinearSampler``)
to pack queried side-plane features into the standard C4 regular basis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

if __package__ in (None, ""):
    # Allow both ``python -m model.triunet_induced`` and direct execution.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from einops import rearrange
from escnn.group import CyclicGroup
from escnn.gspaces import flipRot2dOnR2, no_base_space, rot2dOnR2, rot2dOnR3
from escnn.nn import (
    FieldType,
    GeometricTensor,
    PointwiseAvgPool2D,
    R2Conv,
    R2Upsampling,
    R3Conv,
    ReLU,
    SequentialModule,
)
from torch import Tensor, nn

from utils.common import normalize_coordinate
from model.export_layers import (
    ExportEquivariantDown2d,
    ExportEquivariantResBlock2d,
    ExportEquivariantUp2d,
    ExportInducedJointTriUNet,
)


class _EquivariantResBlock2d(nn.Module):
    """A strictly equivariant residual block (no learned/deformable sampling).

    Mirrors ``CyclicResBlock2d`` from ``equi_layer2d.py`` (used by the plain
    ``JointTriUNet``): three stacked 3x3 convolutions where ``conv1`` and
    ``conv2`` are each followed by ReLU, ``conv3`` has no activation, and a
    ReLU is applied once at the end after the residual addition.  ``conv1``
    already maps ``in_type -> out_type``, so no separate 1x1 skip is needed.
    """

    def __init__(self, in_type: FieldType, out_type: FieldType):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.conv1 = R2Conv(in_type, out_type, kernel_size=3, padding=1)
        self.conv2 = R2Conv(out_type, out_type, kernel_size=3, padding=1)
        self.conv3 = R2Conv(out_type, out_type, kernel_size=3, padding=1)
        self.activation = ReLU(out_type, inplace=False)

    def forward(self, x: Tensor | GeometricTensor) -> GeometricTensor:
        if isinstance(x, Tensor):
            x = GeometricTensor(x, self.in_type)
        out = self.conv1(x)
        out = self.activation(out)  # relu(conv1(x))
        residual = out
        out = self.conv2(out)
        out = self.activation(out)  # relu(conv2(...))
        out = self.conv3(out)  # no activation
        out += residual
        return self.activation(out)

    def export(self) -> "ExportEquivariantResBlock2d":
        model = ExportEquivariantResBlock2d()
        model.conv1 = self.conv1.export()
        model.conv2 = self.conv2.export()
        model.conv3 = self.conv3.export()
        model.activation = self.activation.export()
        return model


class _EquivariantDown2d(nn.Module):
    def __init__(self, in_type: FieldType, out_type: FieldType, pooling: bool):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.pool = (
            PointwiseAvgPool2D(in_type, kernel_size=2, stride=2) if pooling else None
        )
        self.block = _EquivariantResBlock2d(in_type, out_type)

    def forward(self, x: Tensor | GeometricTensor) -> GeometricTensor:
        if isinstance(x, Tensor):
            x = GeometricTensor(x, self.in_type)
        if self.pool is not None:
            x = self.pool(x)
        return self.block(x)

    def export(self) -> "ExportEquivariantDown2d":
        model = ExportEquivariantDown2d()
        if self.pool is not None:
            model.pool = nn.AvgPool2d(
                kernel_size=self.pool.kernel_size, stride=self.pool.stride
            )
        model.block = self.block.export()
        return model


class _EquivariantUp2d(nn.Module):
    def __init__(self, in_type: FieldType, out_type: FieldType):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.up = SequentialModule(
            R2Upsampling(in_type, scale_factor=2, mode="bilinear", align_corners=True),
            R2Conv(in_type, out_type, kernel_size=3, padding=1),
        )
        self.block = _EquivariantResBlock2d(out_type + out_type, out_type)

    def forward(
        self, from_down: Tensor | GeometricTensor, from_up: Tensor | GeometricTensor
    ) -> GeometricTensor:
        if isinstance(from_down, Tensor):
            from_down = GeometricTensor(from_down, self.out_type)
        if isinstance(from_up, Tensor):
            from_up = GeometricTensor(from_up, self.in_type)
        up = self.up(from_up)
        merged = GeometricTensor(
            torch.cat((up.tensor, from_down.tensor), dim=1), self.block.in_type
        )
        return self.block(merged)

    def export(self) -> "ExportEquivariantUp2d":
        model = ExportEquivariantUp2d()
        model.up = self.up.export()
        model.block = self.block.export()
        return model


class InducedBilinearSampler(nn.Module):
    """Query tri-planes and pack the side pair as ``Ind_C2^C4``.

    Each side plane contains regular C2 fields with channel order ``[e, h]``.
    For every corresponding XZ/YZ field, the induced basis is converted from
    the coset-block order ``[xz_e, xz_h, yz_e, yz_h]`` to escnn's standard C4
    regular order ``[xz_e, yz_e, xz_h, yz_h]``.
    """

    def __init__(self, sample_mode: str = "bilinear", padding: float = 0.0):
        super().__init__()
        self.sample_mode = sample_mode
        self.padding = padding

    def _sample(self, p: Tensor, feature: Tensor, plane: str) -> Tensor:
        coordinates = normalize_coordinate(
            p.clone(), plane=plane, padding=self.padding
        )
        grid = 2.0 * coordinates[:, :, None].float() - 1.0
        return F.grid_sample(
            feature,
            grid,
            padding_mode="border",
            align_corners=True,
            mode=self.sample_mode,
        ).squeeze(-1)

    @staticmethod
    def pack_induced_pair(xz: Tensor, yz: Tensor) -> Tensor:
        """Pack ``[B, C, Q]`` side samples into regular C4 fields."""
        if xz.shape != yz.shape:
            raise ValueError(f"XZ and YZ shapes differ: {xz.shape} vs {yz.shape}")
        if xz.ndim != 3 or xz.shape[1] % 2:
            raise ValueError("side features must have shape [B, 2*k, Q]")

        batch, channels, queries = xz.shape
        fields = channels // 2
        xz = xz.reshape(batch, fields, 2, queries)
        yz = yz.reshape(batch, fields, 2, queries)
        # Q^T [a_e, a_h, b_e, b_h] = [a_e, b_e, a_h, b_h].
        packed = torch.stack((xz[:, :, 0], yz[:, :, 0], xz[:, :, 1], yz[:, :, 1]), dim=2)
        return packed.reshape(batch, 4 * fields, queries)

    def forward(self, p: Tensor, planes: Dict[str, Tensor]) -> Tensor:
        xy = self._sample(p, planes["xy"], "xy")
        xz = self._sample(p, planes["xz"], "xz")
        yz = self._sample(p, planes["yz"], "yz")
        side = self.pack_induced_pair(xz, yz)
        return torch.cat((xy, side), dim=1).transpose(1, 2)


class InducedJointTriUNet(nn.Module):
    """Tri-plane UNet with exact C4 equivariance in all three branches.

    ``hidden_dim`` and every encoder width must be divisible by four.  The XY
    branch uses regular C4 fields.  Each vertical plane uses regular C2 fields
    for the reflection ``u -> -u``; the two planes together form the induced
    C4 representation.
    """

    def __init__(
        self,
        in_channel: int = 1,
        hidden_dim: int = 32,
        N: int = 4,
        depth: int = 3,
        plane_resolution: int = 40,
        plane_type: Sequence[str] = ("xy", "xz", "yz"),
        padding: float = 0.0,
        s2tp=True,
        share_plane_unet: bool = False,
    ):
        super().__init__()
        if N != 4:
            raise ValueError("Ind_C2^C4 is implemented specifically for N=4")
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4")
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if plane_resolution % (2 ** (depth - 1)):
            raise ValueError("plane_resolution must be divisible by 2**(depth-1)")

        self.N = N
        self.depth = depth
        self.lifting_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.plane_type = tuple(plane_type)
        self.padding = padding
        # Kept for compatibility with the existing grasp-model wrappers.  It
        # is not used to hide terms during the equivariance test.
        self.test_eq_flag = 0
        # Side-to-XY fusion gate (module-level flag).  When disabled the
        # `side2xy_*` layers are never built, so the XY branch is a pure
        # C4 UNet without cross-plane information.
        self.use_side2xy = s2tp
        # The C4 XY fields and C2 side fields cannot use the same escnn
        # modules.  This flag is therefore applied by ``export()``: the
        # plain-PyTorch model routes XY/XZ/YZ through one shared copy of the
        # XY encoder/decoder.  The equivariant seed model itself stays
        # unchanged and can still be used to initialize the exported model.
        self.share_plane_unet = share_plane_unet

        self.gs = rot2dOnR3(N)
        self.gs2d = rot2dOnR2(N)
        # D1 is abstractly C2, but acts on (z, u) by reflecting u -> -u.
        self.side_gs = flipRot2dOnR2(1)

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr] * in_channel)
        self.hidden_type = FieldType(
            self.gs, [self.gs.regular_repr] * (hidden_dim // N)
        )
        self.lifting = SequentialModule(
            R3Conv(
                self.in_type,
                self.hidden_type,
                kernel_size=3,
                padding=1,
                stride=1,
                initialize=True,
            ),
            ReLU(self.hidden_type, inplace=True),
        )

        self._build_xy_unet()
        self._build_vertical_unet()

        # The YZ chart uses coset representative r.  rho(r)^-1 converts its
        # raw C4 lifting fibres to the canonical induced-representation gauge.
        generator = self.gs.fibergroup.element(1)
        gauge = torch.as_tensor(
            self.hidden_type.representation(~generator),
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("yz_gauge", gauge)

        output_group = CyclicGroup(4)
        self.decoder_type = FieldType(
            no_base_space(output_group),
            [output_group.regular_representation] * (3 * hidden_dim // 4),
        )
        self.feature_sampler = InducedBilinearSampler(padding=padding)

    def _build_xy_unet(self) -> None:
        self.in_type_xy = FieldType(
            self.gs2d,
            [self.gs2d.regular_repr] * (self.lifting_dim // self.N),
        )
        widths = [self.lifting_dim * 2**level for level in range(self.depth)]

        encoders = []
        side_fusions = []
        for level, width in enumerate(widths):
            encoder_in_type = (
                FieldType(
                    self.gs2d,
                    [self.gs2d.regular_repr] * (widths[level - 1] // 4),
                )
                if level
                else self.in_type_xy
            )
            encoder_out_type = FieldType(
                self.gs2d, [self.gs2d.regular_repr] * (width // 4)
            )
            encoder = _EquivariantDown2d(
                encoder_in_type, encoder_out_type, pooling=level > 0
            )
            encoders.append(encoder)
            if self.use_side2xy:
                induced_side_type = FieldType(
                    self.gs2d, [self.gs2d.regular_repr] * (width // 2)
                )
                side_fusions.append(
                    R2Conv(
                        encoder.out_type + induced_side_type,
                        encoder.out_type,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    )
                )
        self.encoders_xy = nn.ModuleList(encoders)
        # When `use_side2xy` is False these stay empty ModuleLists so the
        # forward loop simply iterates over nothing.
        self.side2xy_encoders = nn.ModuleList(side_fusions)

        decoders = []
        side_fusions = []
        reversed_widths = list(reversed(widths))
        for in_width, out_width in zip(reversed_widths, reversed_widths[1:]):
            decoder = _EquivariantUp2d(
                FieldType(
                    self.gs2d, [self.gs2d.regular_repr] * (in_width // 4)
                ),
                FieldType(
                    self.gs2d, [self.gs2d.regular_repr] * (out_width // 4)
                ),
            )
            decoders.append(decoder)
            if self.use_side2xy:
                induced_side_type = FieldType(
                    self.gs2d, [self.gs2d.regular_repr] * (out_width // 2)
                )
                side_fusions.append(
                    R2Conv(
                        decoder.out_type + induced_side_type,
                        decoder.out_type,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    )
                )
        self.decoders_xy = nn.ModuleList(decoders)
        self.side2xy_decoders = nn.ModuleList(side_fusions)

        self.out_type_xy = FieldType(
            self.gs2d,
            [self.gs2d.regular_repr] * (self.lifting_dim // self.N),
        )
        self.final_conv_xy = R2Conv(
            decoders[-1].out_type, self.out_type_xy, kernel_size=1
        )

    def _build_vertical_unet(self) -> None:
        # Res_H Reg(C4) is two copies of Reg(C2), without discarding channels.
        restricted_regular = self.gs2d.regular_repr.restrict(2)
        restricted_type = FieldType(
            self.side_gs,
            [restricted_regular] * (self.lifting_dim // 4),
        )
        side_input_type = FieldType(
            self.side_gs,
            [self.side_gs.regular_repr] * (self.lifting_dim // 2),
        )
        self.side_input_type = restricted_type
        self.side_input_adapter = R2Conv(
            restricted_type, side_input_type, kernel_size=1, padding=0
        )

        widths = [self.lifting_dim * 2**level for level in range(self.depth)]
        vertical_encoders = []
        for level, width in enumerate(widths):
            encoder_in_type = FieldType(
                self.side_gs,
                [self.side_gs.regular_repr]
                * ((widths[level - 1] if level else self.lifting_dim) // 2),
            )
            encoder_out_type = FieldType(
                self.side_gs, [self.side_gs.regular_repr] * (width // 2)
            )
            vertical_encoders.append(
                _EquivariantDown2d(
                    encoder_in_type, encoder_out_type, pooling=level > 0
                )
            )
        self.encoders_vertical = nn.ModuleList(vertical_encoders)

        reversed_widths = list(reversed(widths))
        self.decoders_vertical = nn.ModuleList(
            [
                _EquivariantUp2d(
                    FieldType(
                        self.side_gs,
                        [self.side_gs.regular_repr] * (in_width // 2),
                    ),
                    FieldType(
                        self.side_gs,
                        [self.side_gs.regular_repr] * (out_width // 2),
                    ),
                )
                for in_width, out_width in zip(reversed_widths, reversed_widths[1:])
            ]
        )
        self.out_type_vertical = FieldType(
            self.side_gs,
            [self.side_gs.regular_repr] * (self.lifting_dim // 2),
        )
        self.final_conv_vertical = R2Conv(
            self.decoders_vertical[-1].out_type,
            self.out_type_vertical,
            kernel_size=1,
        )

    @staticmethod
    def _side_to_xy(xz: Tensor, yz: Tensor) -> Tensor:
        """Lift an induced side pair to a regular-C4 XY feature field."""
        if xz.shape != yz.shape or xz.shape[1] % 2:
            raise ValueError("side tensors must match and contain regular C2 fields")
        batch, channels, _, size = xz.shape
        fields = channels // 2
        # Side layout is [B, C, Z, U]; integrate Z and retain U.
        xz = xz.mean(dim=-2).reshape(batch, fields, 2, size)
        yz = yz.mean(dim=-2).reshape(batch, fields, 2, size)
        xz_e = xz[:, :, 0, :, None].expand(-1, -1, -1, size)
        yz_e = yz[:, :, 0, None, :].expand(-1, -1, size, -1)
        xz_h = xz[:, :, 1, :, None].expand(-1, -1, -1, size)
        yz_h = yz[:, :, 1, None, :].expand(-1, -1, size, -1)
        return torch.stack((xz_e, yz_e, xz_h, yz_h), dim=2).reshape(
            batch, 4 * fields, size, size
        )

    def _canonical_side_inputs(self, volume: Tensor) -> tuple[Tensor, Tensor]:
        xz = volume.mean(dim=-2).transpose(-1, -2)
        yz = volume.mean(dim=-3).transpose(-1, -2)
        yz = torch.einsum("oi,bihw->bohw", self.yz_gauge.to(yz), yz)
        xz = self.side_input_adapter(GeometricTensor(xz, self.side_input_type)).tensor
        yz = self.side_input_adapter(GeometricTensor(yz, self.side_input_type)).tensor
        return xz, yz

    def forward(self, x: Tensor | GeometricTensor) -> Dict[str, Tensor]:
        if isinstance(x, Tensor):
            if x.ndim == 4:
                x = x.unsqueeze(1)
            x = GeometricTensor(self.voxel2repr(x), self.in_type)

        lifted = self.repr2voxel(self.lifting(x))
        volume = lifted.reshape(
            -1,
            self.lifting_dim,
            self.reso_plane,
            self.reso_plane,
            self.reso_plane,
        )
        xy = volume.mean(dim=-1)
        xz, yz = self._canonical_side_inputs(volume)

        skip_connections = []
        for level, (xy_encoder, side_encoder) in enumerate(
            zip(self.encoders_xy, self.encoders_vertical)
        ):
            xy = xy_encoder(xy).tensor
            xz = side_encoder(xz).tensor
            yz = side_encoder(yz).tensor
            if self.use_side2xy:
                side_on_xy = self._side_to_xy(xz, yz)
                fusion = self.side2xy_encoders[level]
                xy = fusion(
                    GeometricTensor(torch.cat((xy, side_on_xy), dim=1), fusion.in_type)
                ).tensor
            skip_connections.insert(0, (xy, xz, yz))

        for level, (xy_decoder, side_decoder) in enumerate(
            zip(self.decoders_xy, self.decoders_vertical)
        ):
            skip_xy, skip_xz, skip_yz = skip_connections[level + 1]
            xy = xy_decoder(skip_xy, xy).tensor
            xz = side_decoder(skip_xz, xz).tensor
            yz = side_decoder(skip_yz, yz).tensor
            if self.use_side2xy:
                side_on_xy = self._side_to_xy(xz, yz)
                fusion = self.side2xy_decoders[level]
                xy = fusion(
                    GeometricTensor(torch.cat((xy, side_on_xy), dim=1), fusion.in_type)
                ).tensor

        xy = self.final_conv_xy(GeometricTensor(xy, self.final_conv_xy.in_type)).tensor
        xz = self.final_conv_vertical(
            GeometricTensor(xz, self.final_conv_vertical.in_type)
        ).tensor
        yz = self.final_conv_vertical(
            GeometricTensor(yz, self.final_conv_vertical.in_type)
        ).tensor

        # Convert pixel convention back to the repository's real-coordinate grids.
        return {
            "xy": xy.permute(0, 1, 3, 2),
            # Side CNN layout is already [Z, U], which is grid_sample's
            # [row=Z, column=U] convention for coordinates (u, z).
            "xz": xz,
            "yz": yz,
        }

    def query(self, p: Tensor, planes: Dict[str, Tensor]) -> Tensor:
        """Return queried features with type ``self.decoder_type``."""
        return self.feature_sampler(p, planes)

    def export(
        self, share_plane_unet: Optional[bool] = None
    ) -> ExportInducedJointTriUNet:
        """Export to a plain-PyTorch fallback model (no GeometricTensor).

        All equivariant sub-modules are converted to their standard PyTorch
        counterparts via escnn's ``export()``.  By default, the constructor's
        ``share_plane_unet`` flag controls the exported architecture.  It can
        also be overridden for one export call.

        When plane sharing is disabled, the returned model exactly reproduces
        :meth:`forward`.  When enabled, XY/XZ/YZ all use the same ordinary
        2D encoder, decoder and final convolution.  This intentionally drops
        the equivariance guarantee, while retaining essentially the same
        FLOPs as the two-branch induced model (one XY pass plus two side
        passes).  The shared modules are initialized from the XY C4 branch
        and become unconstrained standard convolutions after export.
        """
        if share_plane_unet is None:
            share_plane_unet = self.share_plane_unet

        model = ExportInducedJointTriUNet()
        model.lifting = self.lifting.export()
        model.side_input_adapter = self.side_input_adapter.export()
        model.lifting_dim = self.lifting_dim
        model.reso_plane = self.reso_plane
        model.padding = self.padding
        model.plane_type = self.plane_type
        model.use_side2xy = self.use_side2xy
        model.share_plane_unet = share_plane_unet

        encoder_xy = []
        encoder_vertical = []
        shared_encoders = []
        side2xy_encoders = []
        for i in range(len(self.encoders_xy)):
            if share_plane_unet:
                shared_encoders.append(self.encoders_xy[i].export())
            else:
                encoder_xy.append(self.encoders_xy[i].export())
                encoder_vertical.append(self.encoders_vertical[i].export())
            if self.use_side2xy:
                side2xy_encoders.append(self.side2xy_encoders[i].export())
        model.encoders_xy = nn.ModuleList(encoder_xy)
        model.encoders_vertical = nn.ModuleList(encoder_vertical)
        model.shared_encoders = nn.ModuleList(shared_encoders)
        model.side2xy_encoders = nn.ModuleList(side2xy_encoders)

        decoder_xy = []
        decoder_vertical = []
        shared_decoders = []
        side2xy_decoders = []
        for i in range(len(self.decoders_xy)):
            if share_plane_unet:
                shared_decoders.append(self.decoders_xy[i].export())
            else:
                decoder_xy.append(self.decoders_xy[i].export())
                decoder_vertical.append(self.decoders_vertical[i].export())
            if self.use_side2xy:
                side2xy_decoders.append(self.side2xy_decoders[i].export())
        model.decoders_xy = nn.ModuleList(decoder_xy)
        model.decoders_vertical = nn.ModuleList(decoder_vertical)
        model.shared_decoders = nn.ModuleList(shared_decoders)
        model.side2xy_decoders = nn.ModuleList(side2xy_decoders)

        if share_plane_unet:
            model.final_conv_shared = self.final_conv_xy.export()
        else:
            model.final_conv_xy = self.final_conv_xy.export()
            model.final_conv_vertical = self.final_conv_vertical.export()
        model.yz_gauge = self.yz_gauge.clone()
        model.feature_sampler = self.feature_sampler
        return model

    def test_export(self, input_=None, atol: float = 1e-4, rtol: float = 1e-4):
        """Verify that :meth:`export` reproduces :meth:`forward`."""
        # This helper checks exact escnn -> PyTorch conversion.  The optional
        # shared-plane experiment changes the architecture by design.
        export_model = self.export(share_plane_unet=False)
        if input_ is None:
            input_ = torch.randn(2, 1, self.reso_plane, self.reso_plane, self.reso_plane)
        plane = self.forward(input_)
        export_plane = export_model(input_)
        for key in plane.keys():
            ok = torch.allclose(plane[key], export_plane[key], atol=atol, rtol=rtol)
            print(f"{key} export test: {'YES' if ok else 'NO'}")
        return export_model

    @torch.no_grad()
    def equivariance_errors(self, x: Tensor, p: Tensor) -> Dict[int, float]:
        """Measure ``max|F(gx, gp) - rho(g)F(x, p)|`` for all four rotations."""
        was_training = self.training
        self.eval()
        if x.ndim == 4:
            x = x.unsqueeze(1)
        base_input = GeometricTensor(self.voxel2repr(x), self.in_type)
        base_features = self.query(p, self(base_input))
        output_group = self.decoder_type.fibergroup
        errors: Dict[int, float] = {}
        for g in self.gs.fibergroup.elements:
            rotation = torch.as_tensor(
                self.gs.fibergroup.irrep(1)(g), device=p.device, dtype=p.dtype
            )
            transformed_points = p.clone()
            transformed_points[..., :2] = torch.einsum(
                "ij,bqj->bqi", rotation, p[..., :2]
            )
            actual = self.query(
                transformed_points, self(base_input.transform(g))
            )
            output_element = output_group.element(g.value)
            rho = torch.as_tensor(
                self.decoder_type.representation(output_element),
                device=base_features.device,
                dtype=base_features.dtype,
            )
            expected = torch.einsum("ij,bqj->bqi", rho, base_features)
            errors[int(g.value)] = (actual - expected).abs().max().item()
        self.train(was_training)
        return errors

    @torch.no_grad()
    def test_eq(
        self,
        input_: Optional[Tensor] = None,
        points: Optional[Tensor] = None,
        *,
        query_count: int = 128,
        seed: int = 0,
        atol: float = 2e-5,
        rtol: float = 1e-4,
        raise_on_error: bool = True,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, float | bool]]:
        """Numerically verify the complete tri-plane C4 intertwining law.

        The check includes the 3D lifting layer, the XY C4 UNet, both shared
        C2 side-plane UNets, side-to-XY fusion, bilinear querying, and the
        induced-representation basis conversion.  For every ``g in C4`` it
        compares

        ``query(g p, forward(g x))`` with ``rho_Ind(g) query(p, forward(x))``.

        Args:
            input_: Optional voxel tensor ``[B, Cin, S, S, S]`` (or
                ``[B, S, S, S]`` when ``Cin=1``).  A random tensor is used by
                default.
            points: Optional query points ``[B, Q, 3]`` in the repository's
                normalized physical coordinates.  Interior random points are
                used by default to avoid border-padding artefacts.
            query_count: Number of random query points when ``points`` is not
                supplied.
            seed: Seed used only for generated test data.
            atol: Absolute tolerance passed to ``torch.allclose``.
            rtol: Relative tolerance passed to ``torch.allclose``.
            raise_on_error: Raise ``AssertionError`` if any group element
                fails.  Set to ``False`` to obtain the report only.
            verbose: Print one compact result line per group element.

        Returns:
            A dictionary keyed by rotations ``0, 1, 2, 3``.  Each entry
            contains ``passed``, ``max_abs`` and ``max_normalized``.
        """
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        if input_ is None:
            input_ = torch.randn(
                1,
                self.in_type.size,
                self.reso_plane,
                self.reso_plane,
                self.reso_plane,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            input_ = input_.to(device=device, dtype=dtype)
            if input_.ndim == 4:
                input_ = input_.unsqueeze(1)

        if input_.ndim != 5 or input_.shape[1] != self.in_type.size:
            raise ValueError(
                "input_ must have shape [B, Cin, S, S, S] with "
                f"Cin={self.in_type.size}; got {tuple(input_.shape)}"
            )
        if tuple(input_.shape[-3:]) != (self.reso_plane,) * 3:
            raise ValueError(
                f"input spatial size must be {(self.reso_plane,) * 3}; "
                f"got {tuple(input_.shape[-3:])}"
            )

        batch_size = input_.shape[0]
        if points is None:
            # normalize_coordinate maps approximately [-.5, .5] to [0, 1].
            # Staying in the interior also makes the test independent of the
            # selected padding mode at the boundary.
            bound = 0.45 * (1.0 + self.padding)
            points = (
                torch.rand(
                    batch_size,
                    query_count,
                    3,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
                * (2.0 * bound)
                - bound
            )
        else:
            points = points.to(device=device, dtype=dtype)
        if points.ndim != 3 or points.shape[0] != batch_size or points.shape[-1] != 3:
            raise ValueError(
                "points must have shape [B, Q, 3] with the same batch size "
                f"as input_; got {tuple(points.shape)}"
            )

        was_training = self.training
        self.eval()
        try:
            base_input = GeometricTensor(self.voxel2repr(input_), self.in_type)
            base_features = self.query(points, self(base_input))
            output_group = self.decoder_type.fibergroup
            report: Dict[int, Dict[str, float | bool]] = {}

            for g in self.gs.fibergroup.elements:
                rotation = torch.as_tensor(
                    self.gs.fibergroup.irrep(1)(g),
                    device=device,
                    dtype=dtype,
                )
                transformed_points = points.clone()
                transformed_points[..., :2] = torch.einsum(
                    "ij,bqj->bqi", rotation, points[..., :2]
                )
                actual = self.query(
                    transformed_points, self(base_input.transform(g))
                )

                output_element = output_group.element(g.value)
                rho = torch.as_tensor(
                    self.decoder_type.representation(output_element),
                    device=device,
                    dtype=dtype,
                )
                expected = torch.einsum("ij,bqj->bqi", rho, base_features)
                difference = (actual - expected).abs()
                tolerance = atol + rtol * expected.abs()
                value = int(g.value)
                report[value] = {
                    "passed": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
                    "max_abs": float(difference.max().item()),
                    "max_normalized": float((difference / tolerance).max().item()),
                }
        finally:
            self.train(was_training)

        if verbose:
            for value, result in report.items():
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"C4 r^{value}: {status} | max_abs={result['max_abs']:.3e} "
                    f"| max_normalized={result['max_normalized']:.3e}"
                )

        failed = [value for value, result in report.items() if not result["passed"]]
        if failed and raise_on_error:
            raise AssertionError(
                f"C4 equivariance failed for rotations {failed}; report={report}"
            )
        return report

    @staticmethod
    def repr2voxel(representation: GeometricTensor) -> Tensor:
        tensor = representation.tensor
        tensor = torch.flip(tensor, (tensor.ndim - 3, tensor.ndim - 2))
        return rearrange(tensor, "b c d w h -> b c h w d")

    @staticmethod
    def voxel2repr(voxel: Tensor) -> Tensor:
        ndim = voxel.ndim
        if ndim == 5:
            representation = rearrange(voxel, "b c h w d -> b c d w h")
        elif ndim == 4:
            representation = rearrange(voxel, "b h w d -> b d w h")
        else:
            raise ValueError(f"expected a 4D or 5D voxel tensor, got {voxel.shape}")
        return torch.flip(representation, (ndim - 3, ndim - 2))


# Short alias for experiments which only swap the backbone import.
JointTriUNetInduced = InducedJointTriUNet


__all__ = [
    "InducedBilinearSampler",
    "InducedJointTriUNet",
    "JointTriUNetInduced",
]


if __name__ == "__main__":
    torch.manual_seed(0)
    model = InducedJointTriUNet()
    # model.test_eq()
    model.test_export()
