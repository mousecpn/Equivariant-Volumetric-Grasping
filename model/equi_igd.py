"""Induced-representation equivariant implicit grasp diffusion model."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from escnn.group import CyclicGroup
from escnn.gspaces import no_base_space
from escnn.nn import FieldType

from model.attention import EquiDeformableAttn2, EquiGraspSO3DeformableAttn
from model.decoder import TriFullEquiDecoder, TriTimeEquiDecoder
from model.diffusion import FlowMatching
from model.triunet_induced import InducedBilinearSampler, JointTriUNetInduced
from utils.loss import occ_loss_fn, rot_loss_fn, sigmoid_focal_loss, width_loss_fn
from utils.transform import (
    Rotation,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    irreps2rot,
    rot2irreps,
    rotation_6d_to_matrix,
)


def gaussian_filter_3d(values, spatial_shape, sigma=1.0, truncate=4.0):
    if sigma <= 0:
        return values
    radius = int(truncate * sigma + 0.5)
    coordinates = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    volume = values.reshape(values.shape[0], 1, *spatial_shape)
    for dimension in range(3):
        kernel_shape = [1, 1, 1]
        kernel_shape[dimension] = kernel.numel()
        axis_kernel = kernel.reshape(1, 1, *kernel_shape)
        padding = [0, 0, 0, 0, 0, 0]
        padding[2 * (2 - dimension):2 * (2 - dimension) + 2] = [radius, radius]
        volume = F.conv3d(F.pad(volume, padding, mode="replicate"), axis_kernel)
    return volume.reshape_as(values)


def negative_sampling(grasp, rotation=True):
    """Generate translation and optional yaw perturbations for grasp scoring."""
    samples = []
    for _ in range(10):
        translated = grasp.clone()
        translated[..., :3] += torch.randn_like(translated[..., :3]) * 0.1
        samples.append(translated)
    if rotation:
        for yaw in np.linspace(0.0, np.pi, 5)[1:-1]:
            rotated = grasp.clone()
            base = Rotation.from_matrix(
                rotation_6d_to_matrix(grasp[..., 3:]).reshape(-1, 3, 3).detach().cpu().numpy()
            )
            delta = Rotation.from_euler("z", yaw)
            matrices = torch.from_numpy((base * delta).as_matrix().astype("float32")).to(grasp.device)
            rotated[..., 3:] = matrix_to_rotation_6d(matrices.reshape(*grasp.shape[:2], 3, 3))
            samples.append(rotated)
    negatives = torch.cat(samples, dim=1)
    return negatives, torch.zeros_like(negatives[..., 0])


class TriEquiIGDInduce(nn.Module):
    """C4-equivariant IGD with an induced XZ/YZ tri-plane representation."""

    def __init__(self, N=4, s2tp=True, mix_repr=True):
        super().__init__()
        if N != 4:
            raise ValueError("TriEquiIGDInduce currently supports N=4 only")

        self.N = N
        hidden_dim = 32
        self.gs = no_base_space(CyclicGroup(N))
        self.backbone = JointTriUNetInduced(
            in_channel=1, hidden_dim=hidden_dim, depth=3, N=N,
            plane_resolution=40, s2tp=s2tp,
        )
        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr] * (3 * hidden_dim // N))
        mixed_repr = [self.gs.regular_repr] + [self.gs.trivial_repr] * 8
        hidden_type = FieldType(self.gs, mixed_repr * (3 * hidden_dim // (8 + N))) if mix_repr else decoder_in_type
        scalar_type = FieldType(self.gs, [self.gs.irrep(0)])
        self.irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)] * 3)
        sampler = InducedBilinearSampler()

        quality_sampler = EquiDeformableAttn2(decoder_in_type)
        quality_sampler.feature_sampler = sampler
        grasp_sampler = EquiGraspSO3DeformableAttn(decoder_in_type, hidden_type)
        grasp_sampler.feature_sampler = sampler
        rotation_sampler = EquiDeformableAttn2(decoder_in_type)
        rotation_sampler.feature_sampler = sampler

        self.decoder_qual = TriFullEquiDecoder(decoder_in_type, scalar_type, N=N, hidden_size=64, feature_sampler=quality_sampler)
        self.decoder_grasp_qual = TriFullEquiDecoder(decoder_in_type, scalar_type, N=N, hidden_size=64, feature_sampler=grasp_sampler)
        self.decoder_width = TriFullEquiDecoder(decoder_in_type, scalar_type, N=N, hidden_size=64, feature_sampler=sampler)
        self.decoder_rot = TriTimeEquiDecoder(decoder_in_type, self.irrep_3x2, N=N, hidden_size=hidden_type)
        self.decoder_rot.feature_sampler = rotation_sampler
        self.decoder_tsdf = TriFullEquiDecoder(decoder_in_type, scalar_type, N=N, hidden_size=64, feature_sampler=sampler)
        self.grasp_sampler = FlowMatching(condition_mask=[0, 0, 0, 0, 0, 0], denosing_steps=10)

    def compute_loss(self, inputs, points, target, p_tsdf, validate=False):
        self.batch_size, self.sample_num = points.shape[:2]
        labels, rotations, widths, occupancy = target
        features = self.backbone(inputs)
        quality = self.decoder_qual(points, features)
        negative_points, negative_labels = negative_sampling(points, rotation=False)
        negative_quality = self.decoder_qual(negative_points, features)
        loss_quality = sigmoid_focal_loss(
            torch.cat((quality.flatten(), negative_quality.flatten())),
            torch.cat((labels.flatten(), negative_labels.flatten())),
            sigmoid=False, reduction="mean", smoothing=0.1,
        )

        width = self.decoder_width(points, features)
        tsdf = self.decoder_tsdf(p_tsdf, features)
        rotation_6d = matrix_to_rotation_6d(quaternion_to_matrix(rotations))
        rotation_irreps = rot2irreps(rotation_6d)
        selected_rotation = rotation_6d[:, np.random.randint(0, 2)].unsqueeze(1)
        grasp = torch.cat((points, selected_rotation), dim=-1)
        grasp_quality = self.decoder_grasp_qual(grasp, features)
        negative_grasps, negative_grasp_labels = negative_sampling(grasp)
        negative_grasp_quality = self.decoder_grasp_qual(negative_grasps, features)
        loss_grasp = sigmoid_focal_loss(
            torch.cat((grasp_quality.flatten(), negative_grasp_quality.flatten())),
            torch.cat((labels.flatten(), negative_grasp_labels.flatten())),
            sigmoid=False, reduction="mean", smoothing=0.1,
        )

        positive = labels == 1
        if positive.any():
            context = self.decoder_rot.query_feature(points, features)[positive]
            if validate:
                noise = torch.rand(context.shape[0], 1, 6, device=points.device)
                sampled = self.grasp_sampler.sample_data(noise, context, self.decoder_rot)
                sampled_rotation = matrix_to_quaternion(rotation_6d_to_matrix(irreps2rot(sampled))).squeeze(1)
                loss_rotation = rot_loss_fn(sampled_rotation, rotations[positive]).mean()
            else:
                loss_rotation = self.grasp_sampler.loss_fn(
                    self.decoder_rot,
                    {"data": rotation_irreps[positive], "context": context.repeat(1, rotation_irreps.shape[1], 1)},
                ).mean()
            loss_width = width_loss_fn(width[positive].reshape(-1), widths[positive]).mean()
        else:
            loss_rotation = torch.zeros((), device=points.device)
            loss_width = torch.zeros((), device=points.device)

        loss_occupancy = occ_loss_fn(tsdf, occupancy, 0.1).mean()
        loss = loss_quality + loss_grasp + loss_rotation + 0.01 * loss_width + loss_occupancy
        loss_dict = {
            "loss_qual": loss_quality,
            "loss_grasp": loss_grasp,
            "loss_rot": loss_rotation,
            "loss_width": loss_width,
            "loss_occ": loss_occupancy,
            "loss_all": loss,
        }
        prediction_rotation = rotations[:, :1]
        if validate and positive.any():
            prediction_rotation = sampled_rotation.unsqueeze(1)
        return loss, loss_dict, ((quality.sigmoid() * grasp_quality.sigmoid()).sqrt(), prediction_rotation, width, tsdf)

    @torch.no_grad()
    def inference(self, inputs, points, sample_rounds=1, low_th=0.05, gaussian_filter_sigma=1.0):
        self.batch_size, self.sample_num = points.shape[:2]
        if self.batch_size != 1:
            raise ValueError("TriEquiIGDInduce inference requires batch size 1")
        features = self.backbone(inputs)
        quality = self.decoder_qual(points, features).sigmoid()
        if quality.numel() == int(np.prod(inputs.shape[-3:])):
            quality = gaussian_filter_3d(quality, inputs.shape[-3:], gaussian_filter_sigma)
        mask = quality > low_th
        if not mask.any():
            mask.reshape(-1)[quality.reshape(-1).argmax()] = True
        positive_points = points[mask].reshape(1, -1, 3)
        context = self.decoder_rot.query_feature(positive_points, features)
        best_rotation, best_score = None, None
        for _ in range(sample_rounds):
            noise = torch.randn(1, positive_points.shape[1], 6, device=points.device)
            rotation_6d = irreps2rot(self.grasp_sampler.sample_data(noise, context, self.decoder_rot))
            score = self.decoder_grasp_qual(torch.cat((positive_points, rotation_6d), dim=-1), features)
            if best_score is None:
                best_rotation, best_score = rotation_6d, score
            else:
                select = score > best_score
                best_rotation = torch.where(select.unsqueeze(-1), rotation_6d, best_rotation)
                best_score = torch.maximum(best_score, score)
        rotations = torch.randn(1, points.shape[1], 4, device=points.device)
        rotations[mask] = matrix_to_quaternion(rotation_6d_to_matrix(best_rotation.reshape(-1, 6)))
        grasp_quality = torch.zeros_like(quality)
        grasp_quality[mask] = best_score.sigmoid().reshape(-1)
        return (quality * grasp_quality).sqrt(), rotations, self.decoder_width(points, features)

    def forward(self, inputs, points, target=None, p_tsdf=None):
        if target is None:
            return self.inference(inputs, points)
        self.batch_size, self.sample_num = points.shape[:2]
        features = self.backbone(inputs)
        quality = self.decoder_qual(points, features)
        width = self.decoder_width(points, features)
        noise = torch.rand(self.batch_size, self.sample_num, 6, device=points.device)
        sampled = self.grasp_sampler.sample_data(noise, self.decoder_rot.query_feature(points, features), self.decoder_rot)
        rotation = matrix_to_quaternion(rotation_6d_to_matrix(irreps2rot(sampled)))
        if p_tsdf is None:
            return quality, rotation, width
        return quality, rotation, width, self.decoder_tsdf(p_tsdf, features)

    def export(self):
        self.backbone = self.backbone.export()
        self.decoder_grasp_qual = self.decoder_grasp_qual.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        self.decoder_width = self.decoder_width.export()
        return self
