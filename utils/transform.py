import numpy as np
import scipy.spatial.transform
import torch
import torch.nn.functional as F

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.
    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.
    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])

    return torch.stack((o1, o2, o3, o0), -1)


# def matrix_to_quaternion(matrix: torch.Tensor, format='ijkr') -> torch.Tensor:
#     """
#     Convert rotations given as rotation matrices to quaternions.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).

#     Returns:
#         quaternions with real part first, as tensor of shape (..., 4).
#     """
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

#     batch_dim = matrix.shape[:-2]
#     m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
#         matrix.reshape(batch_dim + (9,)), dim=-1
#     )

#     q_abs = _sqrt_positive_part(
#         torch.stack(
#             [
#                 1.0 + m00 + m11 + m22,
#                 1.0 + m00 - m11 - m22,
#                 1.0 - m00 + m11 - m22,
#                 1.0 - m00 - m11 + m22,
#             ],
#             dim=-1,
#         )
#     )

#     # we produce the desired quaternion multiplied by each of r, i, j, k
#     quat_by_rijk = torch.stack(
#         [
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
#             # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
#             #  `int`.
#             torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
#         ],
#         dim=-2,
#     )

#     # We floor here at 0.1 but the exact level is not important; if q_abs is small,
#     # the candidate won't be picked.
#     flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
#     quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))


#     # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
#     # forall i; we pick the best-conditioned one (with the largest denominator)

#     quat = quat_candidates[
#         F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
#     ].reshape(batch_dim + (4,))

#     if format == 'ijkr':
#         quat = torch.stack((quat[...,1], quat[...,2], quat[...,3], quat[...,1]), dim=-1)

#     return quat

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()



def irreps2rot(irrep):
        cos1 = irrep[..., 0:1]
        sin1 = irrep[..., 1:2]
        cos2 = irrep[..., 2:3]
        sin2 = irrep[..., 3:4]
        cos3 = irrep[..., 4:5]
        sin3 = irrep[..., 5:6]
        return torch.cat((cos1, cos2, cos3, sin1, sin2, sin3), dim=-1)

def rot2irreps(rot):
    cos1 = rot[..., 0:1]
    cos2 = rot[..., 1:2]
    cos3 = rot[..., 2:3]
    sin1 = rot[..., 3:4]
    sin2 = rot[..., 4:5]
    sin3 = rot[..., 5:6]
    return torch.cat((cos1, sin1, cos2, sin2, cos3, sin3), dim=-1)

def padding(data, num_grasps):
    """
    data: torch.tensor(n_grasps, dim)
    num_grasps: torch.tensor(bs, )
    """
    if len(data.shape) == 2:
        padded_data = torch.full((len(num_grasps), max(num_grasps), data.shape[-1]), float('nan'), device=data.device, dtype=torch.float32)
    elif len(data.shape) == 3:
        padded_data = torch.full((len(num_grasps), max(num_grasps), data.shape[-2], data.shape[-1]), float('nan'), device=data.device, dtype=torch.float32)
    count = 0
    for i in range(len(num_grasps)):
        padded_data[i, :num_grasps[i]] = data[count:count+num_grasps[i]]
        count += num_grasps[i]
    return padded_data

def negative_sampling(grasp, label=None, rotation=True):
    """
    grasp: (bs, ns ,dim)
    """
    neg_samples = torch.zeros_like(grasp[:,:0,:]) # (bs, 0, dim)
    # neg_sample = grasp.clone()
    bs, ns = grasp.shape[0], grasp.shape[1]
    sample_type = np.random.choice([0,1])
    # if rotation is False:
    #     trans_perturb_level = 0.3
    # else:
    trans_perturb_level = 0.1
    rot_perturb_level = 0.5
    num_trans_samples = 10
    num_rotations = 5
    # neg_label = label.clone()

    if rotation is True:
        yaws = np.linspace(0.0, np.pi, num_rotations)
        for yaw in yaws[1:-1]:
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    for i in range(num_trans_samples):
        neg_sample = grasp.clone()
        noise = torch.randn_like(grasp[...,:3]) * trans_perturb_level
        neg_sample[..., :3] += noise
        neg_samples = torch.cat((neg_samples, neg_sample), dim=1)
        if rotation is True:
            yaws = np.linspace(0.0, np.pi, num_rotations)
            yaw = np.random.choice(yaws[1:-1])
            neg_sample = grasp.clone()
            z_rot = Rotation.from_euler("z", yaw)
            R = Rotation.from_matrix(rotation_6d_to_matrix(neg_sample[..., 3:]).reshape(-1,3,3).detach().cpu().numpy())
            # R = Rotation.from_quat(neg_sample[..., 3:].reshape(-1,4).detach().cpu().numpy())

            neg_rot = (R*z_rot).as_matrix()
            neg_rot = torch.from_numpy(neg_rot.astype('float32')).to(grasp.device)

            # noise = torch.randn_like(grasp[...,3:]) * rot_perturb_level
            # neg_sample[..., 3:] += noise
            neg_sample[..., 3:] = matrix_to_rotation_6d(neg_rot.reshape(bs,ns,3,3))
            neg_samples = torch.cat((neg_samples, neg_sample), dim=1)

    return neg_samples, torch.zeros_like(neg_samples[...,0])