import numpy as np
from scipy import ndimage
import torch.utils.data
from pathlib import Path
import sys
sys.path.append("/home/pinhao/IGDv2")
from utils.io import *
from utils.perception import *
from utils.transform import Rotation, Transform
from utils.implicit import get_scene_from_mesh_pose_list
import trimesh
from urchin import URDF, Mesh
import open3d as o3d
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import fpsample
from utils.grasp import Grasp
from utils.visual import grasp2mesh
from utils.noise import apply_noise


class DatasetVoxelOccFile(torch.utils.data.Dataset):
    def __init__(self, root, raw_root, num_point=2048, num_point_occ=2048, augment=False, load_occ=True, load_pcl=False):
        self.root = root
        self.augment = augment
        self.num_point = num_point
        self.num_point_occ = num_point_occ

        self.raw_root = raw_root
        self.num_th = 32
        self.df = read_df(raw_root)
        self.size, _, _, _ = read_setup(raw_root)
        self.bias = np.array([0.0, -0.0047, 0.033])

        self.load_occ = load_occ
        self.load_pcl = load_pcl
        self.pcl_num = 2048
        self.occ_cache = {}

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        ori = Rotation.from_quat(self.df.loc[i, "qx":"qw"].to_numpy(np.single))
        pos = self.df.loc[i, "x":"z"].to_numpy(np.single)
        width = self.df.loc[i, "width"].astype(np.single)
        label = self.df.loc[i, "label"]#.astype(np.int64)
        if label > 0:
            label = 1
        else:
            label = 0
        t1 = time.time()
        if self.load_pcl:
            # t1 = time.time()
            pcl = read_point_cloud(self.root, scene_id)
            # t2 = time.time()
            # print('read_pcl:', t2-t1)
            while pcl.shape[0] < self.pcl_num:
                pcl = np.concatenate([pcl,pcl], axis=0)
            pcl_index = fpsample.bucket_fps_kdline_sampling(pcl, self.pcl_num, h=3)
            pcl = pcl[pcl_index]
        else:
            voxel_grid = read_voxel_grid(self.root, scene_id)
        t2 = time.time()
        if self.load_occ:
            occ_points, occ = self.read_occ(scene_id, self.num_point_occ)

        if self.augment:
            voxel_grid, ori, pos, occ_points = apply_transform(voxel_grid, ori, pos, occ_points, self.size)
        
        if self.load_pcl:
            pcl = pcl/self.size - 0.5
        pos = pos / self.size - 0.5
        width = width / self.size

        rotations = np.empty((2, 4), dtype=np.single)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[0] = ori.as_quat()
        rotations[1] = (ori * R).as_quat()

        if self.load_pcl:
            x, y = pcl, (label, rotations, width)
        else:
            x, y = voxel_grid[0], (label, rotations, width)
        t3 = time.time()

        if self.load_occ:
            occ_points = occ_points / self.size - 0.5
        
            # print("load occ:", t2-t1)
            # print("process:", t3-t2)

            return x, y, pos, occ_points, occ
        else:
            return x, y, pos


    

    def get_item_in_one_scene(self, i):
        scene_id = self.df.loc[i, "scene_id"]
        mask = self.df.loc[:, "scene_id"] == scene_id

        ori_list = []
        ori_array = self.df.loc[mask, "qx":"qw"].to_numpy(np.single)
        for term in ori_array:
            ori_list.append(Rotation.from_quat(term))
        pos_array = self.df.loc[mask, "x":"z"].to_numpy(np.single)
        width_array = self.df.loc[mask, "width"].to_numpy(np.single)
        label_array = self.df.loc[mask, "label"].to_numpy(np.single).astype(np.int64)
        voxel_grid = read_voxel_grid(self.root, scene_id)

        mesh_list = []
        for j in range(ori_array.shape[0]):
            ori = ori_list[j]
            pos = pos_array[j]
            width = width_array[j]
            label = label_array[j]
            if label == 1:
                scene_mesh, gripper_mesh = self.visualize_gripper(i, width, pos, ori, label, viz=False)
                mesh_list.append(gripper_mesh)
        
                o3d.visualization.draw_geometries([scene_mesh,gripper_mesh], window_name="franka hand", width=800,height=600, left=50, top=50, point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
        


    def read_occ(self, scene_id, num_point):
        occ_paths = list((self.raw_root / 'occ' / scene_id).glob('*.npz'))
        path_idx = torch.randint(high=len(occ_paths), size=(1,), dtype=int).item()
        occ_path = occ_paths[path_idx]
        if occ_path in self.occ_cache.keys():
            occ_data = self.occ_cache[occ_path]
            # print('cache hit')
        else:
            occ_data = np.load(occ_path)
            self.occ_cache[occ_path] = occ_data
            # print('cache miss')
        points = occ_data['points']
        occ = occ_data['occ'] 
        points, idxs = sample_point_cloud(points, num_point, return_idx=True)
        occ = occ[idxs]
        return points, occ

    def get_mesh(self, idx):
        scene_id = self.df.loc[idx, "scene_id"]
        mesh_pose_list_path = self.raw_root / 'mesh_pose_list' / (scene_id + '.npz')
        mesh_pose_list = np.load(mesh_pose_list_path, allow_pickle=True)['pc']
        scene = get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False)
        return scene
    


def visualize_point_cloud_with_normals(points, normals=None, grasps=None):
    """
    Visualizes a point cloud with normals using Open3D.

    Args:
        points (torch.Tensor): Tensor of shape (n, 3) representing point cloud.
        normals (torch.Tensor, optional): Tensor of shape (n, 3) representing normals. Defaults to None.
    """
    # Convert points to Open3D format
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcl.normals = o3d.utility.Vector3dVector(normals)

    # Visualize the point cloud
    if grasps is None:
        o3d.visualization.draw_geometries([pcl],
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)
    else:
        o3d.visualization.draw_geometries([pcl, ]+ grasps,
                                        window_name="Point Cloud with Normals",
                                        point_show_normal=True)

def apply_transform(voxel_grid, orientation, position, occ_points, size):
    resolution = voxel_grid.shape[-1]
    position = position / size * resolution
    occ_points = occ_points / size * resolution
    # angle = np.pi / 2.0 * np.random.choice(4)
    # angle = 0
    angle = np.pi * 2 * np.random.rand()
    R_augment = Rotation.from_rotvec(np.r_[0.0, 0.0, angle])

    # z_offset = np.random.uniform(6, 34) - position[2]
    # z_offset = 0
    z_offset = np.random.uniform(-3, 3) 

    t_augment = np.r_[0.0, 0.0, z_offset]
    T_augment = Transform(R_augment, t_augment)

    T_center = Transform(Rotation.identity(), np.r_[20.0, 20.0, 20.0])
    T = T_center * T_augment * T_center.inverse()

    # transform voxel grid
    T_inv = T.inverse()
    matrix, offset = T_inv.rotation.as_matrix(), T_inv.translation
    voxel_grid[0] = ndimage.affine_transform(voxel_grid[0], matrix, offset, order=0)

    # transform grasp pose
    position = T.transform_point(position)
    occ_points = T.transform_point(occ_points)
    orientation = T.rotation * orientation
    position = position * size / resolution
    occ_points = occ_points * size / resolution
    return voxel_grid, orientation, position, occ_points



def sample_point_cloud(pc, num_point, return_idx=False):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    if return_idx:
        return pc[idxs], idxs
    else:
        return pc[idxs]

def trimesh_to_open3d(trimesh_mesh):
    """
    Converts a Trimesh object to an Open3D object.

    Args:
        trimesh_mesh (trimesh.Trimesh): A Trimesh object.

    Returns:
        open3d.geometry.TriangleMesh: The corresponding Open3D mesh.
    """
    # Extract vertices and faces from Trimesh
    vertices = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    # Create Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Optionally compute normals
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh


if __name__=="__main__":
    dataset = DatasetVoxelOccFile(Path("/home/pinhao/IGDv2/data/data_pile_train_processed_dex_noise"), Path("/home/pinhao/IGDv2/data/data_pile_train_raw"), load_occ=True, augment=False)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=0, #collate_fn=SceneBasedDatasetVoxelOccFile.collate
    )
    from tqdm import tqdm
    # t1 = time.time()
    # for data in train_loader:
    #     t2 = time.time()
    #     print("latency:", t2-t1)
    #     t1 = time.time()
        # break
    print()
    for data in tqdm(train_loader):
        t2 = time.time()
        # print("latency:", t2-t1)
        t1 = time.time()