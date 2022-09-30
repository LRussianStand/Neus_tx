# -*- coding: utf-8 -*-
""" 
@Time    : 2022/9/19 14:38
@Author  : HCF
@FileName: process_depth.py
@SoftWare: PyCharm
"""

import os
import cv2
import numpy as np
from glob import glob
import sys
import matplotlib.image as mpimg
from utils import read_array
# from wys.read_write_dense import read_array,read_consistency_graph
from colmap_read_model import read_model, qvec2rotmat
from matplotlib import pyplot as plt
import cv2
import open3d as o3d


colmap_root = '/mnt/data2/lzc/tx/data/scene06fo/dense/0/'
output_root = '/mnt/data2/lzc/tx/neus/scene06fo/'


class colmap_depth():
    def __init__(self, colmap_root):
        self.pic_lis = sorted(glob(os.path.join(colmap_root, 'images/*')))
        self.consistency_lis = sorted(glob(os.path.join(colmap_root, 'stereo/consistency_graphs/*.geometric.bin')))
        self.depth_lis_geo = sorted(glob(os.path.join(colmap_root, 'stereo/depth_maps/*.geometric.bin')))
        self.depth_lis_photo = sorted(glob(os.path.join(colmap_root, 'stereo/depth_maps/*.photometric.bin')))
        assert self.depth_lis_geo.__len__() == self.depth_lis_photo.__len__(), 'inputs do not match'
        assert self.depth_lis_geo.__len__() == self.pic_lis.__len__(), 'inputs do not match'
        self.pic_num = self.pic_lis.__len__()
        self.depth_geo = []
        self.depth_photo = []
        self.mask = []
        self.depth = []
        self.img_shape=(1024,1024)
    def read_depth(self):
        for file in self.depth_lis_geo:
            img = read_array(file)
            self.depth_geo.append(img)
        for file in self.depth_lis_photo:
            img = read_array(file)
            self.depth_photo.append(img)
        self.img_shape = img.shape

    def depth_filter(self, err_thres=0.05, errode=False):
        for i in range(self.pic_num):
            depth_geo = self.depth_geo[i]
            depth_photo = self.depth_photo[i]
            dmin, dmax = depth_geo.min(), depth_geo.max()
            if dmin ==0 and dmax==0:
                mask = np.ones_like(depth_photo)
            else :
                depth_err = np.abs(depth_photo.clip(dmin, dmax) - depth_geo)
                thres = np.quantile(depth_geo[depth_geo != 0], [0.1, 0.9])
                thres = np.abs(thres[1] - thres[0]) * err_thres
                mask = depth_err <= thres
            if errode:
                mask = cv2.erode(np.asarray(mask, np.uint8), kernel=np.ones((5, 5), np.uint8))
                mask = np.asarray(mask, np.bool)
            self.mask.append(mask)
            # print(i)

    # def write_consistency(self, output_root):
    #     if not os.path.exists(os.path.join(output_root, 'consistency')):
    #         os.mkdir(os.path.join(output_root, 'consistency'))
    #     for i in range(self.consistency_lis.__len__()):
    #         img = read_array(self.consistency_lis[i])
    #         mpimg.imsave(os.path.join(output_root, 'consistency', f'{i.__str__().zfill(6)}_consistency.jpg'), img)

    def write_consistency_fakenpy(self, output_root):
        h,w = self.img_shape[:2]
        if not os.path.exists(os.path.join(output_root)):
            os.mkdir(os.path.join(output_root))
        lis = []
        for i in range(self.depth_geo.__len__()):
            img = np.ones((h,w))
            lis.append(img)
            # img = np.expand_dims(img, 2)
        np.save(os.path.join(output_root, 'consistency.npy'), lis)

    def write_depth_npy(self, output_root, filter=True):
        '''
        save depth geo as npy file
        '''
        if not os.path.exists(os.path.join(output_root)):
            os.mkdir(os.path.join(output_root))
        if filter:
            img = []
            for i in range(self.pic_num):
                img.append(self.depth_geo[i] * self.mask[i])
            np.save(os.path.join(output_root, 'depth.npy'), img)
        else:
            np.save(os.path.join(output_root, 'depth.npy'), self.depth_geo)

class colmap_sfm():
    def __init__(self, dense_folder):
        self.img_root = os.path.join(dense_folder,'images')
        self.dense_folder = dense_folder
        self.model_dir = os.path.join(dense_folder, 'sparse')
        self.cameras, self.images, self.points3d = read_model(self.model_dir, '.bin')
        self.num_images = len(list(self.images.items()))
        self.name_list = []
        self.image_file_list = [(int(imgname.split('.')[0]), imgname) for i, imgname in enumerate(os.listdir(os.path.join(dense_folder,'images')))]
        self.image_file_list.sort(key=lambda record: record[0])

        # intrinsic
        param_type = {
            'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
            'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
            'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
            'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
            'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
            'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
            'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
            'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
            'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
            'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
            'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
        }
        self.intrinsic = {}
        for camera_id, cam in self.cameras.items():
            params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
            if 'f' in param_type[cam.model]:
                params_dict['fx'] = params_dict['f']
                params_dict['fy'] = params_dict['f']
            i = np.array([
                [params_dict['fx'], 0, params_dict['cx']],
                [0, params_dict['fy'], params_dict['cy']],
                [0, 0, 1]
            ])
            self.intrinsic[camera_id] = i
        print('intrinsic[1]\n', self.intrinsic[1], end='\n\n')

        # extrinsic
        self.extrinsic = {}
        # for image_id, image in self.images.items():
        #     e = np.zeros((4, 4))
        #     e[:3, :3] = qvec2rotmat(image.qvec)
        #     e[:3, 3] = image.tvec
        #     e[3, 3] = 1
        #     self.extrinsic[image_id] = e
        # print('extrinsic[1]\n', self.extrinsic[1], end='\n\n')
        for image_id, image in self.images.items():
            e = np.zeros((4, 4))
            img_name = image.name
            self.name_list.append(img_name)
            e[:3, :3] = qvec2rotmat(image.qvec)
            e[:3, 3] = image.tvec
            e[3, 3] = 1
            self.extrinsic[img_name] = e
        print('extrinsic[1]\n', self.extrinsic[img_name], end='\n\n')

        # depth range and interval
        self.intrinsic_img_match = {}
        max_d = 0
        interval_scale = 1
        self.depth_ranges = {}
        for image_id, image in self.images.items():  # 此处i改为从dic keys中得到的索引
            name = image.name
            zs = []
            for p3d_id in image.point3D_ids:
                if p3d_id == -1:
                    continue
                transformed = np.matmul(self.extrinsic[name],
                                        [self.points3d[p3d_id].xyz[0], self.points3d[p3d_id].xyz[1], self.points3d[p3d_id].xyz[2], 1])
                zs.append(np.asscalar(transformed[2]))
            zs_sorted = sorted(zs)
            # relaxed depth range
            depth_min = zs_sorted[int(len(zs) * .01)]
            depth_max = zs_sorted[int(len(zs) * .99)]
            # determine depth number by inverse depth setting, see supplementary material
            if max_d == 0:
                image_int = self.intrinsic[image.camera_id]
                self.intrinsic_img_match[name] = image.camera_id
                image_ext = self.extrinsic[name]
                image_r = image_ext[0:3, 0:3]
                image_t = image_ext[0:3, 3]
                p1 = [image_int[0, 2], image_int[1, 2], 1]
                p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
                P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
                P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
                P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
                P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
                depth_num = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
            else:
                depth_num = max_d
            depth_interval = (depth_max - depth_min) / (depth_num - 1) / interval_scale
            self.depth_ranges[name] = (depth_min, depth_interval, depth_num, depth_max)
        print('depth_ranges[name]\n', self.depth_ranges[name], end='\n\n')

    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 256
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        return cam

    def write_cam(self, save_dir):
        # write
        try:
            os.makedirs(save_dir)
        except os.error:
            print(save_dir + ' already exist.')

        for i, (id, img_file) in enumerate(self.image_file_list):

            with open(os.path.join(save_dir, '%08d_cam.txt' % i), 'w') as f:
                f.write('extrinsic\n')
                for j in range(4):
                    for k in range(4):
                        f.write(str(self.extrinsic[img_file][j, k]) + ' ')
                    f.write('\n')
                f.write('\nintrinsic\n')
                for j in range(3):
                    for k in range(3):
                        f.write(str(self.intrinsic[self.intrinsic_img_match[img_file]][j, k]) + ' ')
                    f.write('\n')
                f.write('\n%f %f %f %f\n' % (
                self.depth_ranges[img_file][0], self.depth_ranges[img_file][1], self.depth_ranges[img_file][2], self.depth_ranges[img_file][3]))

    def write_cam_npz(self, cam_folder, normalization_mat, cam_num, save_path, cameras_filename):
        '''
        :param cam_folder: which contains split cam file
        :param normalization_mat:
        :param cam_num: int
        :param save_path:
        :param cameras_filename: save name
        :return:
        '''
        cameras_new = {}
        new_idx = 0
        for idx in range(cam_num):
            cam = self.load_cam(os.path.join(cam_folder, f'{idx.__str__().zfill(8)}_cam.txt'))
            proj_mat = cam[1][:3, :3] @ cam[0][:3, :]
            cameras_new['scale_mat_%d' % new_idx] = normalization_mat
            cameras_new['world_mat_%d' % new_idx] = np.concatenate((proj_mat, np.array([[0, 0, 0, 1.0]])),
                                                                   axis=0).astype(np.float32)
            new_idx += 1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez('{0}/{1}.npz'.format(save_path, cameras_filename), **cameras_new)

    def get_image_mask(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, 'image')):
            os.mkdir(os.path.join(save_path, 'image'))
        if not os.path.exists(os.path.join(save_path, 'mask')):
            os.mkdir(os.path.join(save_path, 'mask'))

        img_list = [i[1] for i in self.image_file_list]
        print(img_list)
        if len(glob(os.path.join(save_path, 'image', "*.jpg"))) == len(self.image_file_list)\
                and len(glob(os.path.join(save_path, 'mask', "*.jpg"))) == len(self.image_file_list):
            print('bigscene2idr input finished')
            return
        for idx in range(len(img_list)):
            pic = mpimg.imread(os.path.join(self.img_root, img_list[idx]))
            mask = np.ones(pic.shape)
            mpimg.imsave(os.path.join(save_path, 'image', f'{idx.__str__().zfill(6)}.jpg'), pic)
            mpimg.imsave(os.path.join(save_path, 'mask', f'{idx.__str__().zfill(6)}.jpg'), mask)
            print(idx)
        print('bigscene2idr input finished')



scannet_dense = colmap_depth(colmap_root)
scannet_dense.read_depth()
scannet_dense.depth_filter()
os.makedirs(output_root, exist_ok=True)
scannet_dense.write_depth_npy(output_root, filter=True)
scannet_dense.write_consistency_fakenpy(output_root)


scannetmodel = colmap_sfm(colmap_root)
if not os.path.exists(os.path.join(output_root,'cams')):
    scannetmodel.write_cam(os.path.join(output_root,'cams'))
scannetmodel.get_image_mask(output_root)
#sfm重建的点云
def scale1(work_dir):
    points = np.asarray([np.asarray(scannetmodel.points3d[point].xyz, dtype=np.float) for point in scannetmodel.points3d])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10,
                                                        std_ratio=0.3)
    # o3d.visualization.draw_geometries([cl])
    points = np.asarray(cl.points)
    cam_position = np.asarray([image.tvec for i,image in scannetmodel.images.items()])
    pcd_cam = o3d.geometry.PointCloud()
    pcd_cam.points = o3d.utility.Vector3dVector(cam_position)
    # o3d.visualization.draw_geometries([cl,pcd_cam])

    # get normalization mat
    s_scale = 2  # 继续缩小两倍
    x_quantile_up = np.quantile(points[:, 0], 0.9)
    y_quantile_up = np.quantile(points[:, 1], 0.9)
    z_quantile_up = np.quantile(points[:, 2], 0.9)
    x_quantile_down = np.quantile(points[:, 0], 0.1)
    y_quantile_down = np.quantile(points[:, 1], 0.1)
    z_quantile_down = np.quantile(points[:, 2], 0.1)
    x_set = points[np.logical_and(points[:, 0] <= x_quantile_up, points[:, 0] >= x_quantile_down), 0]
    y_set = points[np.logical_and(points[:, 1] <= y_quantile_up, points[:, 1] >= y_quantile_down), 1]
    z_set = points[np.logical_and(points[:, 2] <= z_quantile_up, points[:, 2] >= z_quantile_down), 2]
    x_mean = x_set.mean()
    y_mean = y_set.mean()
    z_mean = z_set.mean()
    scale = np.concatenate((x_set, y_set, z_set)).std()
    normalization = np.eye(4).astype(np.float32)
    normalization[0, 3] = x_mean
    normalization[1, 3] = y_mean
    normalization[2, 3] = z_mean
    normalization[0, 0] = scale * s_scale
    normalization[1, 1] = scale * s_scale
    normalization[2, 2] = scale * s_scale
    return normalization


def scale2(work_dir):
    import trimesh
    pcd = trimesh.load(os.path.join(work_dir, 'inter_point.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center
    return scale_mat

# normalization = scale2(colmap_root)
normalization = scale1(colmap_root)
scannetmodel.write_cam_npz(os.path.join(output_root,'cams'), normalization, scannetmodel.num_images, output_root, 'cameras_sphere')
