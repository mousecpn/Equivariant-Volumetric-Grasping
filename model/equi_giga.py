import torch
import torch.nn as nn
import sys
import os
cur_file = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_file)
parent_dir = os.path.dirname(cur_dir)
sys.path.insert(0, parent_dir)
from model.backbone import CyclicUnet3d, Tri_UNet
from model.triunet import JointTriUNet
from model.decoder import EquiLocalDecoder, LocalDecoder, TriEquiLocalDecoder,BilinearSampler, TriFullEquiDecoder
import escnn.nn as enn
from escnn.group import CyclicGroup, DihedralGroup
from escnn.nn import FieldType, GeometricTensor
from escnn.gspaces import no_base_space, rot2dOnR2
import torch.nn.functional as F
from utils.transform import matrix_to_quaternion,rotation_6d_to_matrix, irreps2rot, padding, negative_sampling, Rotation
import time
from scipy import stats
import numpy as np
from utils.loss import *
from model.triunet_induced import JointTriUNetInduced, InducedBilinearSampler


class EquiGIGA3d(nn.Module):
    def __init__(self):
        super(EquiGIGA3d, self).__init__()
        self._init = 'he'
        self.N = 4
        self.gs = no_base_space(CyclicGroup(self.N))
        self.backbone = CyclicUnet3d(in_channel=1, out_channel=96, hidden_dim=32, depth=3, N=self.N)
        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr]*(self.backbone.out_type.size//self.N))

        irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
        irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)]*3)
        self.decoder_qual = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        self.decoder_width = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        self.decoder_rot = TriFullEquiDecoder(decoder_in_type, irrep_3x2, dim=3, N=self.N, hidden_size=32)
        self.decoder_tsdf = TriFullEquiDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        # self.device = device

        # self.decoder_qual = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)
        # self.decoder_width = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)
        # self.decoder_rot = LocalDecoder(dim=3, c_dim=96, out_dim=4,  hidden_size=32)
        # self.decoder_tsdf = LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32)

    def export(self):
        """Export the equivariant model to a standard PyTorch CNN.

        The backbone (``CyclicUnet3d``) and each decoder (``TriFullEquiDecoder``)
        already implement ``export()``, which strips the ``GeometricTensor``
        wrappers and replaces equivariant layers with plain convolutions /
        linear layers.  This method composes those exports so the resulting
        model can be profiled with standard FLOPs tools (e.g. ``thop``).
        """
        self.backbone = self.backbone.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_width = self.decoder_width.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        return self


    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        c = self.backbone(inputs.unsqueeze(1))
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        # rot = nn.functional.normalize(rot, dim=-1)

        if p_tsdf is not None:
            # if self.detach_tsdf:
            #     for k, v in c.items():
            #         c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width


        # if self.training:
        #     label, rot_gt, width_gt, occ_value = target
        #     loss_qual = self._qual_loss_fn(qual.reshape(-1), label.reshape(-1)).mean()
        #     loss_rot = self._quat_loss_fn(rot, rot_gt)


class TriEquiGIGA(nn.Module):
    def __init__(self, N=4, flip=False):
        super(TriEquiGIGA, self).__init__()
        self.N = N
        self.flip = flip
        hidden_dim = 36 if self.N % 3 ==0 else 32
        # self.backbone = Tri_UNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N)
        self.backbone = JointTriUNet(in_channel=1, hidden_dim=hidden_dim, depth=3, N=self.N, flip=flip, plane_resolution=40)
        if self.flip:
            self.gs = no_base_space(DihedralGroup(self.N))
        else:
            self.gs = no_base_space(CyclicGroup(self.N))

        decoder_in_type = FieldType(self.gs, [self.gs.regular_repr]*(hidden_dim//self.gs.regular_repr.size) + [self.gs.trivial_repr]*hidden_dim*2)

        if self.flip:
            irrep_0 = FieldType(self.gs, [self.gs.irrep(0,0)])
            irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1,1)]*3)
        else:
            irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
            irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)]*3)


        # decoder_hidden_dim = 96
        # mixed_repr = [self.gs.regular_repr] + [self.gs.trivial_repr]*8
        # hidden_type = FieldType(self.gs, mixed_repr*(decoder_hidden_dim//16))
        # self.decoder_qual = EquiLocalDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        # self.decoder_width = EquiLocalDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        # self.decoder_rot = EquiLocalDecoder(decoder_in_type, irrep_3x2, dim=3, N=self.N, hidden_size=32)
        # self.decoder_tsdf = EquiLocalDecoder(decoder_in_type, irrep_0, dim=3, N=self.N, hidden_size=32)
        # self.device = device

        # self.decoder_qual = TriEquiLocalDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64,inv_map=True)
        # self.decoder_width = TriEquiLocalDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64)
        # self.decoder_rot = TriEquiLocalDecoder(in_type=decoder_in_type, out_type=irrep_3x2, N=self.N, hidden_size=64)
        # self.decoder_tsdf = TriEquiLocalDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=64)

        self.decoder_qual = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_width = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_rot = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_3x2, N=self.N, hidden_size=32, feature_sampler=BilinearSampler())
        self.decoder_tsdf = TriFullEquiDecoder(in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32, feature_sampler=BilinearSampler())

        # self.decoder_qual =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())
        # self.decoder_width =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())
        # self.decoder_rot =  LocalDecoder(dim=3, c_dim=96, out_dim=6, hidden_size=32, feature_sampler=BilinearSampler())
        # self.decoder_tsdf =  LocalDecoder(dim=3, c_dim=96, out_dim=1, hidden_size=32, feature_sampler=BilinearSampler())

    def export(self, ):
        self.backbone = self.backbone.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        self.decoder_width = self.decoder_width.export()
        return self

    def scene_compute_loss(self, inputs, p, target=None, p_tsdf=None, validate=False):
        """
        p (num_grasps, 3)
        inputs (bs, 1, 40, 40, 40)
        """
        self.batch_size = inputs.size(0)
        label, rot_gt, width_gt, num_grasps, occ_value = target

        c = self.backbone(inputs)
        padded_p = padding(p.squeeze(1), num_grasps=num_grasps)
        valid_mask = ~torch.isnan(padded_p).all(-1)


        qual = self.decoder_qual(padded_p, c)[valid_mask]
        width = self.decoder_width(padded_p, c)[valid_mask]
        tsdf = self.decoder_tsdf(p_tsdf, c)

        loss_qual = qual_loss_fn(qual.flatten(), label.flatten(), 0.1).mean()

        # neg_p, neg_label = negative_sampling(padded_p, None, rotation=False)
        # neg_qual = self.decoder_qual(neg_p, c)

        # all_qual = torch.cat((qual.flatten(), neg_qual.flatten()), dim=0)
        # all_label = torch.cat((label.flatten(), neg_label.flatten()), dim=0)
        # loss_qual = sigmoid_focal_loss(all_qual, all_label,  sigmoid=False, reduction='mean',smoothing=0.1)

        irreps = self.decoder_rot(padded_p, c)
        rot = irreps2rot(irreps)[valid_mask]
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        loss_rot = rot_loss_fn(rot, rot_gt)[label==1].mean()


        # loss_qual = qual_loss_fn(qual, label, 0.1).mean()
        # loss_grasp_qual = _qual_loss_fn(grasp_qual.squeeze(1), label).mean()
        loss_width = width_loss_fn(width[label==1].reshape(-1), width_gt[label==1]).mean()
        loss_occ = occ_loss_fn(tsdf, occ_value, 0.1).mean()

        loss = loss_qual +  loss_rot + 0.01 * loss_width + loss_occ
        loss_dict = {'loss_qual': loss_qual,
                    'loss_rot': loss_rot,
                    'loss_width': loss_width,
                    'loss_occ': loss_occ,
                    'loss_all': loss}

        # y_pred = ((qual.sigmoid()*grasp_qual.sigmoid()).sqrt(), rot_gt[:,:1], width, tsdf)
        if validate:
            y_pred = (qual.sigmoid(), rot, width, tsdf)
        else:
            y_pred = (qual.sigmoid(), rot_gt[:,:1], width, tsdf)

        return loss, loss_dict, y_pred


    def scene_forward(self, inputs, p, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)


        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        # rot = nn.functional.normalize(rot, dim=-1)

        if p_tsdf is not None:
            # if self.detach_tsdf:
            #     for k, v in c.items():
            #         c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width




    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)#.sigmoid()
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        # rot = irreps
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        # rot = nn.functional.normalize(rot, dim=-1)

        if p_tsdf is not None:
            # if self.detach_tsdf:
            #     for k, v in c.items():
            #         c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width


    def test_eq(self,):
        p = torch.rand(1, 1, 3)-0.5
        input_ = torch.randn(1, 1, 40, 40, 40)
        for g in self.gs.testing_elements:
            # if g.value[1] == 0:
            #     continue
            input = self.backbone.voxel2repr(input_)
            input = GeometricTensor(input, self.backbone.in_type)
            input_transformed = input.transform(g)
            if self.flip == True:
                rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value[1]*np.pi / 2.0])
                ## reflection
                rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)
                reflection_map = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float32))
                if g.value[0] == 1:
                    rot_mat = rot_mat @ reflection_map

                if g.value[0] == 1:
                    p_trans = p.clone()
                    p_trans[:,:,0] = p_trans[:,:,0]
                    p_trans[:,:,1] = -p_trans[:,:,1]
                else:
                    p_trans = p.clone()
                p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1,3))).reshape(1,1,3)
                p_trans = torch.tensor(p_trans, dtype=torch.float32)
            else:
                rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value * 2.0 * np.pi / self.N])
                rot_mat = torch.tensor(rot.as_matrix()[:2,:2], dtype=torch.float32)
                p_trans = torch.tensor(rot.apply(p.numpy().reshape(-1,3))).reshape(1,1,3)


            qual, rot, width = self.forward(input, p)
            qual_trans, rot_trans, width_trans = self.forward(input_transformed, p_trans)
            # print('rot_trans:', rot_trans.detach().numpy().reshape(-1))

            rot_trans_after = (rot_mat @ rot.reshape(2,3)).reshape(1,1,6)
            # rot_trans_after = GeometricTensor(rot.reshape(1,6), self.decoder_rot.out_type).transform(g).tensor.reshape(1,1,6)
            # print('rot_trans_after:', rot_trans_after.detach().numpy().reshape(-1))


            print('rot equi test:  ' + ('YES' if torch.allclose(rot_trans, rot_trans_after, atol=1e-4, rtol=1e-4) else 'NO'))
            print('width inv test:  ' + ('YES' if torch.allclose(width, width_trans, atol=1e-4, rtol=1e-4) else 'NO'))
            print('qual inv test:  ' + ('YES' if torch.allclose(qual, qual_trans, atol=1e-4, rtol=1e-4) else 'NO'))
        return

class TriEquiGIGAInduce(nn.Module):
    """Tri-plane GIGA whose backbone is the induced-representation UNet.

    Unlike :class:`TriEquiGIGA` (which collapses the vertical planes to
    invariant scalars), :class:`JointTriUNetInduced` keeps the XZ/YZ pair as
    an ``Ind_{C2}^{C4}`` field.  The induced backbone therefore returns three
    regular-C4 planes (XY + the packed side pair), so every decoder consumes
    a purely regular input type and uses :class:`InducedBilinearSampler`.
    """

    def __init__(self, N=4, flip=False, s2tp=True, share_plane=False):
        super(TriEquiGIGAInduce, self).__init__()
        self.N = N
        self.flip = flip
        hidden_dim = 36 if self.N % 3 == 0 else 32
        # Induced backbone: all three planes are regular C4 fields.
        self.backbone = JointTriUNetInduced(
            in_channel=1,
            hidden_dim=hidden_dim,
            depth=3,
            N=self.N,
            plane_resolution=40,
            s2tp=s2tp,
            share_plane_unet=share_plane
        )
        if self.flip:
            self.gs = no_base_space(DihedralGroup(self.N))
        else:
            self.gs = no_base_space(CyclicGroup(self.N))

        # XY (hidden_dim) + induced side pair (2*hidden_dim) = 3*hidden_dim
        # regular C4 channels.
        decoder_in_type = FieldType(
            self.gs,
            [self.gs.regular_repr] * (3 * hidden_dim // self.gs.regular_repr.size),
        )

        if self.flip:
            irrep_0 = FieldType(self.gs, [self.gs.irrep(0, 0)])
            irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1, 1)] * 3)
        else:
            irrep_0 = FieldType(self.gs, [self.gs.irrep(0)])
            irrep_3x2 = FieldType(self.gs, [self.gs.irrep(1)] * 3)

        self.decoder_qual = TriFullEquiDecoder(
            in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_width = TriFullEquiDecoder(
            in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_rot = TriFullEquiDecoder(
            in_type=decoder_in_type, out_type=irrep_3x2, N=self.N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )
        self.decoder_tsdf = TriFullEquiDecoder(
            in_type=decoder_in_type, out_type=irrep_0, N=self.N, hidden_size=32,
            feature_sampler=InducedBilinearSampler(),
        )

    def export(self):
        self.backbone = self.backbone.export()
        self.decoder_qual = self.decoder_qual.export()
        self.decoder_rot = self.decoder_rot.export()
        self.decoder_tsdf = self.decoder_tsdf.export()
        self.decoder_width = self.decoder_width.export()
        return self

    def scene_compute_loss(self, inputs, p, target=None, p_tsdf=None, validate=False):
        """
        p (num_grasps, 3)
        inputs (bs, 1, 40, 40, 40)
        """
        self.batch_size = inputs.size(0)
        label, rot_gt, width_gt, num_grasps, occ_value = target

        c = self.backbone(inputs)
        padded_p = padding(p.squeeze(1), num_grasps=num_grasps)
        valid_mask = ~torch.isnan(padded_p).all(-1)

        qual = self.decoder_qual(padded_p, c)[valid_mask]
        width = self.decoder_width(padded_p, c)[valid_mask]
        tsdf = self.decoder_tsdf(p_tsdf, c)

        loss_qual = qual_loss_fn(qual.flatten(), label.flatten(), 0.1).mean()

        irreps = self.decoder_rot(padded_p, c)
        rot = irreps2rot(irreps)[valid_mask]
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))
        loss_rot = rot_loss_fn(rot, rot_gt)[label == 1].mean()

        loss_width = width_loss_fn(width[label == 1].reshape(-1), width_gt[label == 1]).mean()
        loss_occ = occ_loss_fn(tsdf, occ_value, 0.1).mean()

        loss = loss_qual + loss_rot + 0.01 * loss_width + loss_occ
        loss_dict = {
            'loss_qual': loss_qual,
            'loss_rot': loss_rot,
            'loss_width': loss_width,
            'loss_occ': loss_occ,
            'loss_all': loss,
        }

        if validate:
            y_pred = (qual.sigmoid(), rot, width, tsdf)
        else:
            y_pred = (qual.sigmoid(), rot_gt[:, :1], width, tsdf)

        return loss, loss_dict, y_pred

    def scene_forward(self, inputs, p, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))

        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

    def forward(self, inputs, p, target=None, p_tsdf=None):
        if isinstance(p, dict):
            self.batch_size = p['p'].size(0)
            self.sample_num = p['p'].size(1)
        else:
            self.batch_size = p.size(0)
            self.sample_num = p.size(1)

        c = self.backbone(inputs)
        qual = self.decoder_qual(p, c)
        width = self.decoder_width(p, c)
        irreps = self.decoder_rot(p, c)
        rot = irreps2rot(irreps)
        rot = matrix_to_quaternion(rotation_6d_to_matrix(rot))

        if p_tsdf is not None:
            tsdf = self.decoder_tsdf(p_tsdf, c)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

    def test_eq(self):
        """Numerically verify C4 equivariance of the induced tri-plane model.

        For every ``g in C4`` it checks that rotating the input voxel grid and
        the query points by ``g`` transforms the outputs as ``qual``/``width``
        (invariant) and the raw 6D rotation irreps as a 3x2 vector field
        (``rot_trans = Rg @ rot.reshape(2, 3)``).

        Note: the quaternion returned by :meth:`forward` is *not* tested here,
        because ``rotation_6d_to_matrix`` + ``matrix_to_quaternion`` is not an
        equivariant map for the 3x2 irrep; the equivariant quantity is the raw
        irreps output of ``decoder_rot``.
        """
        p = torch.rand(1, 1, 3) - 0.5
        input_ = torch.randn(1, 1, 40, 40, 40)
        for g in self.gs.testing_elements:
            input = self.backbone.voxel2repr(input_)
            input = GeometricTensor(input, self.backbone.in_type)
            input_transformed = input.transform(g)
            if self.flip:
                rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value[1] * np.pi / 2.0])
                reflection_map = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float32))
                if g.value[0] == 1:
                    rot = rot * Rotation.from_matrix(reflection_map.numpy())
                if g.value[0] == 1:
                    p_trans = p.clone()
                    p_trans[:, :, 1] = -p_trans[:, :, 1]
                else:
                    p_trans = p.clone()
                p_trans = torch.tensor(rot.apply(p_trans.numpy().reshape(-1, 3))).reshape(1, 1, 3)
                p_trans = torch.tensor(p_trans, dtype=torch.float32)
            else:
                rot = Rotation.from_rotvec(np.r_[0.0, 0.0, g.value * 2.0 * np.pi / self.N])
                p_trans = torch.tensor(rot.apply(p.numpy().reshape(-1, 3))).reshape(1, 1, 3)

            c = self.backbone(input)
            c_trans = self.backbone(input_transformed)

            qual = self.decoder_qual(p, c)
            width = self.decoder_width(p, c)
            irreps = self.decoder_rot(p, c)
            qual_trans = self.decoder_qual(p_trans, c_trans)
            width_trans = self.decoder_width(p_trans, c_trans)
            irreps_trans = self.decoder_rot(p_trans, c_trans)

            # 3x2 irrep: the 6D output is three 2D vectors
            # [cos1, sin1, cos2, sin2, cos3, sin3], each rotating by the 2x2
            # action of g.
            rot_mat = torch.tensor(rot.as_matrix()[:2, :2], dtype=torch.float32)
            irreps_after = (irreps.reshape(3, 2) @ rot_mat.T).reshape(1, 1, 6)

            print('rot equi test:  ' + ('YES' if torch.allclose(irreps_trans, irreps_after, atol=1e-4, rtol=1e-4) else 'NO'))
            print('width inv test:  ' + ('YES' if torch.allclose(width, width_trans, atol=1e-4, rtol=1e-4) else 'NO'))
            print('qual inv test:  ' + ('YES' if torch.allclose(qual, qual_trans, atol=1e-4, rtol=1e-4) else 'NO'))
        return


def apply_rot(rot_mat, vector):
    shape = vector.shape
    vector = vector.reshape(-1,3).permute(1,0)
    vector[:2] = rot_mat @ vector[:2]
    vector = vector.permute(1,0).reshape(shape)
    return vector


if __name__=="__main__":
    model = EquiGIGA3d()


    # model = TriEquiGIGAInduce(s2tp=False, share_plane=True)#.cuda()
    model.export()
    ### calculate params
    total_params = sum(p.numel() for p in model.parameters())
    print(f'The model has {total_params} parameters')
    # p = torch.rand(128, 1, 3).cuda()
    # input = torch.randn(128, 1, 40, 40, 40).cuda()
    # model(input, p)

    # model.test_eq()
    # p = torch.rand(4, 5, 3).cuda() - 0.5
    # input = torch.randn(4, 1, 40, 40, 40).cuda()
    # p[:,3:] = float('nan')

    # model.scene_forward(input, p)

