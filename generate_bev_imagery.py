#!/usr/bin/python3

import argparse
import logging
import os
import pdb
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import pyntcloud

from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t
from argoverse.data_loading.synchronization_database import get_timestamps_from_sensor_folder
from argoverse.utils.se2 import SE2
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class BEVParams:
    def __init__(
        self, img_h: int = 1000, img_w: int = 1000, meters_per_px: float = 0.1, accumulate_sweeps: bool = True
    ) -> None:
        """ meters_per_px is resolution """
        self.img_h = img_h
        self.img_w = img_w
        self.meters_per_px = meters_per_px
        self.accumulate_sweeps = accumulate_sweeps

        # num px in horizontal direction
        h_px = img_w / 2

        # num px in vertical direction
        v_px = img_h / 2

        # get grid boundaries in meters
        xmin_m = -int(h_px * meters_per_px)
        xmax_m = int(h_px * meters_per_px)
        ymin_m = -int(v_px * meters_per_px)
        ymax_m = int(v_px * meters_per_px)

        xlims = [xmin_m, xmax_m]
        ylims = [ymin_m, ymax_m]

        self.xlims = xlims
        self.ylims = ylims

    def __str__(self):
        """ String representation of class for stdout """
        params_repr = "BEVParams:"
        for k, v in self.__dict__.items():
            if "lims" in k:
                params_repr += f" ({k}: [{v[0]}_{v[1]}])"
            else:
                params_repr += f" ({k}: {v})"
        return params_repr


def prune_to_2d_bbox(pts: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float) -> np.ndarray:
    """"""
    x = pts[:, 0]
    y = pts[:, 1]
    is_valid = np.logical_and.reduce([xmin <= x, x <= xmax, ymin <= y, y <= ymax])
    return pts[is_valid]


def test_prune_to_2d_bbox():
    """ """
    pts = np.array([[-2, 2], [2, 0], [1, 2], [0, 1]])  # will be discarded  # will be discarded
    xmin = -1
    ymin = -1
    xmax = 1
    ymax = 2

    pts = prune_to_2d_bbox(pts, xmin, ymin, xmax, ymax)
    gt_pts = np.array([[1, 2], [0, 1]])
    assert np.allclose(pts, gt_pts)


def load_ply(ply_fpath: str) -> np.ndarray:
    """Load a point cloud file from a filepath.
    Args:
        ply_fpath: Path to a PLY file

    Returns:
        arr: Array of shape (N, 3)
    """
    data = pyntcloud.PyntCloud.from_file(os.fspath(ply_fpath))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    intensity = np.array(data.points.intensity)[:, np.newaxis]

    return np.concatenate((x, y, z, intensity), axis=1)


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Bring image values to [0,1] range
    Args:
        img: (H,W,C) or (H,W) image
    """
    img -= img.min() # shift min val to 0
    img /= img.max() # shrink max val to 1
    return img

def equalize_distribution(reflectance: np.ndarray) -> np.ndarray:
    """Add one to reflectance to map 0 values to 0 under logarithm
    """
    log_reflectance = np.log(reflectance + 1)
    log_reflectance = normalize_img(log_reflectance) * 255
    log_reflectance = np.round(log_reflectance).astype(np.uint8)
    return log_reflectance


def render_bev_img(bev_params: BEVParams, lidar_pts: np.ndarray) -> np.ndarray:
    """
    Args:
        bev_params: parameters for rendering
        lidar_pts: accumulated points in the egovehicle frame.
    """
    grid_xmin, grid_xmax = bev_params.xlims
    grid_ymin, grid_ymax = bev_params.ylims

    lidar_pts = prune_to_2d_bbox(lidar_pts, grid_xmin, grid_ymin, grid_xmax, grid_ymax)

    reflectance = lidar_pts[:, 3]

    num_lidar_pts = lidar_pts.shape[0]
    print(f"Rendering {num_lidar_pts/1e6} million LiDAR points")

    img_SE2_ego = SE2(rotation=np.eye(2), translation=np.array([-grid_xmin, -grid_ymin]))

    lidar_xy = lidar_pts[:, :2]
    img_xy = img_SE2_ego.transform_point_cloud(lidar_xy)
    img_xy *= 1 / bev_params.meters_per_px  # m/px -> px/m
    img_xy = np.round(img_xy).astype(np.int64)
    xmax, ymax = np.amax(img_xy, axis=0)
    img_h = ymax + 1
    img_w = xmax + 1
    x = img_xy[:, 0]
    y = img_xy[:, 1]

    bev_refl_img = np.zeros((img_h, img_w), dtype=np.uint8)
    bev_refl_img[y, x] = reflectance

    return bev_refl_img


def accumulate_all_frames(log_id: str, dataset_dir: str) -> np.ndarray:
    """ Obtain (N,4) array representing (x,y,z) in city coordinate system, and reflectance """
    all_city_pts = np.zeros((0, 4))
    sensor_folder_wildcard = f"{dataset_dir}/{log_id}/lidar/PC_*.ply"
    lidar_timestamps = get_timestamps_from_sensor_folder(sensor_folder_wildcard)
    for lidar_timestamp in lidar_timestamps:

        city_SE3_egovehicle = get_city_SE3_egovehicle_at_sensor_t(lidar_timestamp, dataset_dir, log_id)
        ply_fpath = f"{dataset_dir}/{log_id}/lidar/PC_{lidar_timestamp}.ply"
        lidar_pts = load_ply(ply_fpath)
        ego_xyz = lidar_pts[:, :3]
        ego_reflectance = lidar_pts[:, 3].reshape(-1, 1)
        city_xyz = city_SE3_egovehicle.transform_point_cloud(ego_xyz)
        city_pts = np.hstack([city_xyz, ego_reflectance])
        all_city_pts = np.vstack([all_city_pts, city_pts])

    return all_city_pts


def write_out_log_train_data(
    args: argparse.Namespace,
    bev_params: BEVParams,
    log_id: str,
    dl: SimpleArgoverseTrackingDataLoader,
    split_data_dir: str,
) -> None:
    """"""
    log_city_pts = accumulate_all_frames(log_id, split_data_dir)

    ply_fpaths = dl.get_ordered_log_ply_fpaths(log_id)
    all_lidar_timestamps = [Path(path).stem.split("_")[-1] for path in ply_fpaths]
    all_city_SE3_ego = [dl.get_city_SE3_egovehicle(log_id, t) for t in all_lidar_timestamps]

    dataset_name = ""
    for k, v in bev_params.__dict__.items():
        if "lims" in k:
            dataset_name += f"__{k}_{v[0]}_{v[1]}"
        else:
            dataset_name += f"__{k}_{v}"

    dataset_save_dir = f"{args.dump_dir}/{dataset_name}/{log_id}"
    os.makedirs(dataset_save_dir, exist_ok=True)

    # render at each of the ego-poses
    for fr_idx, city_SE3_egovehicle in enumerate(all_city_SE3_ego):

        egovehicle_SE3_city = city_SE3_egovehicle.inverse()
        # whether or not to accumulate multiple sweeps
        if bev_params.accumulate_sweeps:
            ego_pts = log_city_pts.copy()
            ego_pts[:, :3] = egovehicle_SE3_city.transform_point_cloud(log_city_pts.copy()[:, :3])
        else:
            ego_pts = load_ply(ply_fpaths[fr_idx])

        ego_pts[:, 3] = equalize_distribution(ego_pts[:, 3])

        bev_refl_img = render_bev_img(bev_params, ego_pts)
        refl_img_fpath = f"{dataset_save_dir}/refl__fr_{fr_idx}.png"
        imageio.imwrite(refl_img_fpath, bev_refl_img)


def generate_imagery_all_splits(args: argparse.Namespace):
    """ """
    bev_params = BEVParams()

    print(f"Generate imagery w/ params:")
    print(bev_params)

    for split in ["train1", "train2", "train3", "train4", "val", "test"]:
        split_data_dir = f"{args.argoverse_data_root}/{split}"
        dl = SimpleArgoverseTrackingDataLoader(data_dir=split_data_dir, labels_dir=split_data_dir)

        log_ids = list(dl.sdb.get_valid_logs())

        fn_args = [(args, bev_params, log_id, dl, split_data_dir) for log_id in log_ids]
        with Pool(args.num_processes) as p:
            accum = p.starmap(write_out_log_train_data, fn_args)

        #write_out_log_train_data(args, bev_params, log_ids[0], dl, split_data_dir)


if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--argoverse_data_root",
        type=str,
        default="/srv/share/cliu324/argoverse-tracking-readonly",
        help="directory data where argoverse data subsets were downloaded to",
    )
    parser.add_argument("--dump_dir", type=str, required=True, help="directory where to dump the imagery")
    parser.add_argument(
        "--num_processes", type=int, required=True, help="number of processes to launch (uses multiprocessing if >1)"
    )
    args = parser.parse_args()
    logging.info(args)

    generate_imagery_all_splits(args)
