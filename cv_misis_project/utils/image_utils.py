from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


def load_image(img_path: Path, convert_to_rgb: bool = True) -> np.ndarray:
    target_img = cv2.imread(str(img_path))
    if convert_to_rgb:
        return cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    else:
        return target_img


def get_density_map(img_path: Path, ground_truth_path: Path, sigma: int = 15, truncate: int = 5 * 5) -> np.ndarray:
    loaded_img = load_image(img_path)
    image_base = np.zeros_like(loaded_img, dtype=np.float32)[:, :, 0]
    ground_truth_data = loadmat(str(ground_truth_path))['image_info']
    ground_truth_coordinates = ground_truth_data[0][0][0][0][0].astype(int)
    ground_truth_coordinates = ground_truth_coordinates[
        (ground_truth_coordinates[:, 0] < image_base.shape[1]) & (ground_truth_coordinates[:, 1] < image_base.shape[0])
        ]
    x_cords, y_cords = ground_truth_coordinates[:, 1], ground_truth_coordinates[:, 0]
    image_base[x_cords, y_cords] = 1
    gaussian_data = gaussian_filter(image_base, sigma=sigma, mode='constant')
    gaussian_data = gaussian_data / np.sum(gaussian_data) * len(ground_truth_coordinates)
    return gaussian_data


# def get_gaussian_filter_density(img_path: Path, ground_truth_path: Path):
#     # Логика из оригинальной работы
#     gt =
#
#     print gt.shape
#     density = np.zeros(gt.shape, dtype=np.float32)
#     gt_count = np.count_nonzero(gt)
#     if gt_count == 0:
#         return density
#
#     pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
#     leafsize = 2048
#     # build kdtree
#     tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
#     # query kdtree
#     distances, locations = tree.query(pts, k=4)
#
#     print 'generate density...'
#     for i, pt in enumerate(pts):
#         pt2d = np.zeros(gt.shape, dtype=np.float32)
#         pt2d[pt[1],pt[0]] = 1.
#         if gt_count > 1:
#             sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
#         else:
#             sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
#         density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
#     return density
