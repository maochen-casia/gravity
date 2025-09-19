import os
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import csv
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple
from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage
import time
import tqdm
import shutil
import re
import random

from transform import quat2R, R2quat, compose_transformation, image2world, world2image, cam2world


K = torch.tensor([[610.17784, 0, 512], [0, 610.17784, 512], [0, 0, 1]], dtype=torch.float32)
quat_left2rov = torch.tensor([0.579, 0.406, 0.406, 0.579])
R_left2rov = quat2R(quat_left2rov)
t_left2rov = torch.tensor([1.000, -0.155, -1.500])
baseline = torch.ones([]) * 0.31

def pfm2tensor(file_path, size):
    """Convert a PFM file to a torch.Tensor."""

    with open(file_path, 'rb') as f:
        header = f.readline().decode().rstrip()
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        shape = (height, width)
        depth = np.reshape(data, shape) # shape (height, width)
        depth = torch.from_numpy(depth)

    depth = F.interpolate(depth[None,None], (size, size), mode='bilinear', align_corners=False)
    depth = depth.squeeze()
        
    return depth

# --- Helper functions used by the interpolation process ---

def read_pose_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads pose data from a CSV file, removes duplicates, and sorts by timestamp.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Pose file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, dtype={'timestamp': 'int64'})
        
        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        return df.sort_values(by='timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

def extract_timestamps_from_filenames(directory_path: str) -> List[int]:
    """
    Extracts and sorts integer timestamps from image filenames in a directory.
    """
    if not os.path.isdir(directory_path):
        print(f"Warning: Image directory not found at {directory_path}")
        return []
    try:
        timestamps = [
            int(os.path.splitext(f)[0])
            for f in os.listdir(directory_path)
            if f.endswith('.png') and os.path.splitext(f)[0].isdigit()
        ]
        timestamps.sort()
        return timestamps
    except ValueError as e:
        print(f"Error parsing timestamps in {directory_path}: {e}")
        return []

def write_poses_to_csv(filepath: str, poses: List[Dict], header: List[str]):
    """
    Writes a list of pose dictionaries to a CSV file.
    This replaces the row-by-row writing in the original function.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for pose in poses:
            # Ensures the values are written in the correct order as defined by the header
            writer.writerow([pose.get(key) for key in header])

def interpolate_pose(pose_df: pd.DataFrame, timestamp: int) -> Optional[Dict]:
    """
    Interpolates rover pose for a given timestamp using linear and spherical interpolation (Slerp).
    """
    timestamps = pose_df['timestamp'].values
    if not (timestamps[0] <= timestamp <= timestamps[-1]):
        # This check is important to avoid interpolation errors
        return None

    # Linear interpolation for position (x, y, z)
    position_interpolator = interp1d(
        timestamps,
        pose_df[['x', 'y', 'z']].values,
        axis=0,
        fill_value="extrapolate"
    )
    interpolated_position = position_interpolator(timestamp)

    # Spherical Linear Interpolation (Slerp) for orientation (quaternions)
    # The 'from_quat' method initializes a 'Rotation' object from quaternions. [2, 3]
    rotations = Rotation.from_quat(pose_df[['qx', 'qy', 'qz', 'qw']].values)
    slerp = Slerp(timestamps, rotations)
    interpolated_rotation = slerp(timestamp)
    # The 'as_quat()' method represents the rotation as a quaternion. [2]
    interpolated_quat = interpolated_rotation.as_quat()  # Returns in [x, y, z, w] order

    return {
        'timestamp': timestamp,
        'x': interpolated_position[0], 'y': interpolated_position[1], 'z': interpolated_position[2],
        'qw': interpolated_quat[3], 'qx': interpolated_quat[0], 'qy': interpolated_quat[1], 'qz': interpolated_quat[2]
    }

# --- Refactored Main Interpolation Function ---

def process_and_interpolate_rover_poses(base_path: str = './raw_data'):
    """
    This function replaces `interpolate_and_save_rover_poses`.

    It reads rover poses, interpolates them to match image timestamps for each scene,
    and saves the complete result to a new CSV file.
    """
    print("--- Starting Rover Pose Interpolation Process ---")
    for scene_id in range(1, 10):
        scene_name = f"Moon_{scene_id}"
        scene_path = os.path.join(base_path, scene_name)
        print(f"Processing Scene: {scene_name}")
        
        rover_poses_df = read_pose_data(os.path.join(scene_path, 'rover_poses.txt'))
        if rover_poses_df is None or rover_poses_df.empty:
            print(f"  Warning: No rover pose data found or readable for scene {scene_id}. Skipping.")
            continue
            
        image_timestamps = extract_timestamps_from_filenames(os.path.join(scene_path, 'left_images'))
        if not image_timestamps:
            print(f"  Warning: No image timestamps found for scene {scene_id}. Skipping.")
            continue
        
        # Collect all valid interpolated poses in a list first
        interpolated_poses = []
        for ts in image_timestamps:
            pose = interpolate_pose(rover_poses_df, ts)
            if pose:
                interpolated_poses.append(pose)

        if not interpolated_poses:
            print(f"  Warning: No poses could be interpolated for scene {scene_id}.")
            continue

        # Write the entire list of poses to the CSV file at once
        output_file = os.path.join(scene_path, 'rover_poses_interpolated.txt')
        header = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        write_poses_to_csv(output_file, interpolated_poses, header)
        print(f"  Successfully interpolated and saved {len(interpolated_poses)} poses to {output_file}")
    
    print("--- Finished Rover Pose Interpolation Process ---\n")

def _process_batch_by_visibility(
    batch_rover_poses: pd.DataFrame,
    scene_path: str,
    sat_data: Dict,
    pixel_grid: torch.Tensor,
    H_orig: int, W_orig: int,
    downsample_size: int,
    dev: torch.device
) -> List[Dict]:
    """
    Processes a single batch of rover images to find their satellite pairs.

    Args:
        batch_rover_poses: DataFrame slice containing poses for the current batch.
        scene_path: Path to the current scene directory.
        sat_data: Dict with satellite poses ('R', 't') and timestamps.
        pixel_grid: Pre-computed grid of pixel coordinates for projection.
        H_orig, W_orig: Original height and width of the images.
        downsample_size: The size of the downsampled depth maps.
        dev: The torch device for computation.

    Returns:
        A list of dictionaries, where each dictionary is a rover-satellite pair.
    """
    current_batch_size = len(batch_rover_poses)
    batch_rover_ts = batch_rover_poses['timestamp'].values
    
    # --- Load and prepare depth data for the batch ---
    batch_depths = []
    valid_ts_mask = []
    for ts in batch_rover_ts:
        depth_path = os.path.join(scene_path, 'depths', f"{ts}.pfm")
        try:
            depth_tensor = pfm2tensor(depth_path, size=downsample_size).to(dev)
            batch_depths.append(depth_tensor.reshape(1, -1))
            valid_ts_mask.append(True)
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not load or process depth for timestamp {ts}. Skipping. Error: {e}")
            valid_ts_mask.append(False)
    
    if not batch_depths:
        return []

    depth_values = torch.cat(batch_depths, dim=0)
    batch_rover_poses = batch_rover_poses[valid_ts_mask]
    current_batch_size = len(batch_rover_poses)
    
    # --- Get rover poses for the valid items in the batch ---
    quat_rov_batch = torch.from_numpy(batch_rover_poses[['qw', 'qx', 'qy', 'qz']].values).float()
    R_rov2world_batch = torch.stack([quat2R(q) for q in quat_rov_batch], dim=0).to(dev)
    t_rov2world_batch = torch.from_numpy(batch_rover_poses[['x', 'y', 'z']].values).float().to(dev)
    
    R_left2world, t_left2world = compose_transformation(
        R_left2rov.unsqueeze(0).expand(current_batch_size, -1, -1).to(dev),
        t_left2rov.unsqueeze(0).expand(current_batch_size, -1).to(dev),
        R_rov2world_batch,
        t_rov2world_batch
    )

    # --- Project rover pixels to world frame ---
    points_world = image2world(
        points=pixel_grid.expand(current_batch_size, -1, -1),
        K=K.unsqueeze(0).expand(current_batch_size, -1, -1).to(dev),
        R_c2w=R_left2world,
        t_c2w=t_left2world,
        depth=depth_values
    )

    # --- Project world points to all satellite image frames using tensor expansion ---
    num_sat_images = len(sat_data['timestamps'])
    # Expand dims for broadcasting: (B, 1, N, 3) and (1, S, 3, 3), (1, S, 3)
    points_world_expanded = points_world.unsqueeze(1).expand(-1, num_sat_images, -1, -1)
    R_sat_expanded = sat_data['R'].unsqueeze(0).expand(current_batch_size, -1, -1, -1)
    t_sat_expanded = sat_data['t'].unsqueeze(0).expand(current_batch_size, -1, -1)
    
    B, S, N, _ = points_world_expanded.shape
    # Reshape for batched world2image call
    projected_pixels = world2image(
        points=points_world_expanded.reshape(B * S, N, 3),
        K=K.unsqueeze(0).expand(B * S, -1, -1).to(dev),
        R_c2w=R_sat_expanded.reshape(B * S, 3, 3),
        t_c2w=t_sat_expanded.reshape(B * S, 3)
    ).reshape(B, S, N, 2)

    # --- Count visible pixels ---
    in_bounds = (projected_pixels[..., 0] >= 0) & (projected_pixels[..., 0] < W_orig) & \
                (projected_pixels[..., 1] >= 0) & (projected_pixels[..., 1] < H_orig)
    
    visibility_counts = in_bounds.sum(dim=-1)

    # --- Find best satellite image for each rover image and format output ---
    best_sat_indices = torch.argmax(visibility_counts, dim=1)
    best_sat_timestamps = sat_data['timestamps'][best_sat_indices]
    
    batch_pairs = []
    for i, rov_ts in enumerate(batch_rover_poses['timestamp'].values):
        batch_pairs.append({
            'rover_timestamp': int(rov_ts),
            'sat_timestamp': int(best_sat_timestamps[i].item()),
            'x_offset_ratio': random.random() * 2 - 1,
            'y_offset_ratio': random.random() * 2 - 1
        })
        
    return batch_pairs

def _process_batch_by_distance(
    batch_rover_poses: pd.DataFrame,
    sat_data: Dict,
    dev: torch.device
) -> List[Dict]:
    """
    Processes a single batch of rover poses to find the closest satellite image pair by distance.

    This method calculates the Euclidean distance between each rover camera's 3D position
    and all satellite camera positions for the scene. It then pairs each rover image
    with the satellite image corresponding to the minimum distance.

    Args:
        batch_rover_poses (pd.DataFrame): DataFrame slice containing poses for the current batch.
        sat_data (Dict): Dictionary containing satellite data, including poses ('R', 't')
                         and 'timestamps', all as torch.Tensors on the target device.
        dev (torch.device): The torch device for computation ('cuda' or 'cpu').

    Returns:
        A list of dictionaries, where each dictionary represents a rover-satellite pair
        in the format {'rover_timestamp': int, 'sat_timestamp': int}.
    """
    current_batch_size = len(batch_rover_poses)
    
    # --- Get rover poses for the valid items in the batch ---
    quat_rov_batch = torch.from_numpy(batch_rover_poses[['qw', 'qx', 'qy', 'qz']].values).float()
    R_rov2world_batch = torch.stack([quat2R(q) for q in quat_rov_batch], dim=0).to(dev)
    t_rov2world_batch = torch.from_numpy(batch_rover_poses[['x', 'y', 'z']].values).float().to(dev)
    
    # --- Calculate the 3D position of the rover's left camera in the world frame ---
    # This composes the transformation from the left camera to the rover's body,
    # and then from the rover's body to the world.
    _, t_left2world = compose_transformation(
        R_left2rov.unsqueeze(0).expand(current_batch_size, -1, -1).to(dev),
        t_left2rov.unsqueeze(0).expand(current_batch_size, -1).to(dev),
        R_rov2world_batch,
        t_rov2world_batch
    ) # We only need the translation component (t_left2world) for distance calculation.

    # --- Calculate Euclidean distance between each rover pose and all satellite poses ---
    # Get satellite positions from the pre-loaded satellite data dictionary
    t_sat_all = sat_data['t'] # Shape: (S, 3), where S is the number of satellite images

    # Use broadcasting to efficiently compute all pairwise distances.
    # Expand rover positions to (B, 1, 3) and satellite positions to (1, S, 3).
    # The subtraction results in a tensor of shape (B, S, 3).
    # torch.norm computes the L2 norm (Euclidean distance) along the last dimension.
    distances = torch.norm(t_left2world.unsqueeze(1) - t_sat_all.unsqueeze(0), dim=-1) # Shape: (B, S)

    # --- Find the closest satellite image for each rover image ---
    # torch.argmin finds the index of the minimum distance for each rover in the batch.
    best_sat_indices = torch.argmin(distances, dim=1) # Shape: (B,)
    
    # Retrieve the timestamps of the closest satellite images using the indices.
    best_sat_timestamps = sat_data['timestamps'][best_sat_indices]
    
    # --- Format the output ---
    batch_pairs = []
    for i, rov_ts in enumerate(batch_rover_poses['timestamp'].values):
        batch_pairs.append({
            'rover_timestamp': int(rov_ts),
            'sat_timestamp': int(best_sat_timestamps[i].item()),
            'x_offset_ratio': random.random() * 2 - 1,
            'y_offset_ratio': random.random() * 2 - 1
        })
        
    return batch_pairs

def generate_pairs(base_path: str = './raw_data', batch_size: int = 32, device: str = 'cuda', downsample_scale: int = 4):
    """
    Generates rover-satellite image pairs based on 3D projection and visibility.

    Args:
        base_path (str): The root directory where the raw scene data is stored.
        batch_size (int): The number of rover images to process in a single batch.
        device (str): The device to use for computation ('cuda' or 'cpu').
        downsample_scale (int): Factor to downsample images for faster processing.
    """
    print("--- Starting Pair Generation Process ---")
    
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU.")

    H, W = 1024, 1024
    H_down, W_down = H // downsample_scale, W // downsample_scale
    yy, xx = torch.meshgrid(torch.arange(H_down, device=dev), torch.arange(W_down, device=dev), indexing='ij')
    pixel_grid = torch.stack([xx * downsample_scale, yy * downsample_scale], dim=-1).float().view(1, -1, 2)

    for scene_id in range(1, 10):
        scene_name = f"Moon_{scene_id}"
        scene_path = os.path.join(base_path, scene_name)
        print(f"Processing Scene: {scene_name}")

        rover_poses_df = read_pose_data(os.path.join(scene_path, 'rover_poses_interpolated.txt'))
        sat_poses_df = read_pose_data(os.path.join(scene_path, 'sat_poses.txt'))

        if rover_poses_df is None or sat_poses_df is None or rover_poses_df.empty or sat_poses_df.empty:
            print(f"  Warning: Skipping scene {scene_id} due to missing pose data.")
            continue

        # Prepare satellite data once per scene and move to device
        sat_data = {
            'timestamps': torch.tensor(sat_poses_df['timestamp'].values, device=dev),
            'R': torch.stack([quat2R(torch.tensor(q)) for q in sat_poses_df[['qw', 'qx', 'qy', 'qz']].values]).to(dev),
            't': torch.from_numpy(sat_poses_df[['x', 'y', 'z']].values).float().to(dev)
        }
        
        found_pairs = []
        num_rover_images = len(rover_poses_df)
        
        # Process rover images in batches
        for i in tqdm.tqdm(range(0, num_rover_images, batch_size), desc=f"  Pairing for Scene {scene_id}"):
            batch_df = rover_poses_df.iloc[i:i + batch_size]
            
            # batch_pairs = _process_batch_by_visibility(
            #     batch_rover_poses=batch_df,
            #     scene_path=scene_path,
            #     sat_data=sat_data,
            #     pixel_grid=pixel_grid,
            #     H_orig=H, W_orig=W,
            #     downsample_size=W_down,
            #     dev=dev
            # )
            batch_pairs = _process_batch_by_distance(
                batch_rover_poses=batch_df,
                sat_data=sat_data,
                dev=dev
            )
            found_pairs.extend(batch_pairs)
        
        if not found_pairs:
            print(f"  Warning: No matching pairs found for scene {scene_id}.")
            continue

        # Save pairs to file
        output_file = os.path.join(scene_path, 'visible_pairs.txt')
        header = ['rover_timestamp', 'sat_timestamp', 'x_offset_ratio', 'y_offset_ratio']
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(found_pairs)
            print(f"  Successfully found and saved {len(found_pairs)} pairs to {output_file}")
        except IOError as e:
            print(f"  Error writing pairs to file {output_file}: {e}")

    print("--- Finished Pair Generation Process ---")

def read_pairs_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads rover-satellite timestamp pairs from the 'pairs.txt' file.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Pairs file not found at {file_path}")
        return None
    try:
        # The file has a simple structure: rover_timestamp,sat_timestamp
        df = pd.read_csv(file_path, dtype={'rover_timestamp': 'int64', 'sat_timestamp': 'int64'})
        return df
    except Exception as e:
        print(f"Error reading or processing file {file_path}: {e}")
        return None

def create_pair_splits(
    train_scenes: List[int],
    val_scenes: List[int],
    test_scenes: List[int],
    base_path: str = './raw_data',
    output_path: str = './'
):
    """
    Processes the generated data and creates training, validation, and test splits.

    Args:
        train_scenes: A list of scene IDs (e.g., [1, 2, 3, 4]) for the training set.
        val_scenes: A list of scene IDs for the validation set.
        test_scenes: A list of scene IDs for the test set.
        base_path: The root directory where the raw_data is stored.
        output_path: The directory where the final .pth files will be saved.
    """
    print("\n--- Starting Data Split Creation ---")
    os.makedirs(output_path, exist_ok=True)

    # Process each split
    for split_name, scene_ids in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        if not scene_ids:
            print(f"Skipping {split_name} set (no scenes provided).")
            continue

        print(f"Processing {split_name} set for scenes: {scene_ids}...")
        
        all_data_pairs = []
        for scene_id in scene_ids:
            scene_name = f"Moon_{scene_id}"
            scene_path = os.path.join(base_path, scene_name)
            
            # Load the pose data for rover (left) and satellite images
            pairs_df = read_pairs_data(os.path.join(scene_path, 'visible_pairs.txt'))
            rover_poses_df = read_pose_data(os.path.join(scene_path, 'rover_poses_interpolated.txt'))
            sat_poses_df = read_pose_data(os.path.join(scene_path, 'sat_poses.txt'))

            # Check if all required data is present
            if pairs_df is None or rover_poses_df is None or sat_poses_df is None:
                print(f"  Warning: Skipping scene {scene_id} for {split_name} split due to missing data file(s).")
                continue

            # --- Use timestamp as index for fast lookups ---
            rover_poses_df.set_index('timestamp', inplace=True)
            sat_poses_df.set_index('timestamp', inplace=True)
            
            for _, pair in pairs_df.iterrows():
                rover_timestamp = int(pair['rover_timestamp'])
                sat_timestamp = int(pair['sat_timestamp'])

                try:
                    rover_row = rover_poses_df.loc[rover_timestamp]
                    sat_row = sat_poses_df.loc[sat_timestamp]
                except KeyError as e:
                    print(f"  Warning: Timestamp {e} not found in pose files for scene {scene_id}. Skipping pair.")
                    continue

                quat_rov = torch.tensor([rover_row.qw, rover_row.qx, rover_row.qy, rover_row.qz])
                t_rov2world = torch.tensor([rover_row.x, rover_row.y, rover_row.z], dtype=torch.float32)
                R_rov2world = quat2R(quat_rov)
                
                global R_left2rov, t_left2rov, K
                R_left2world, t_left2world = compose_transformation(R_left2rov.unsqueeze(0), 
                                                                    t_left2rov.unsqueeze(0), 
                                                                    R_rov2world.unsqueeze(0), 
                                                                    t_rov2world.unsqueeze(0))
                R_left2world, t_left2world = R_left2world.squeeze(0), t_left2world.squeeze(0)
                
                # --- Create Satellite Data ---
                quat_sat = torch.tensor([sat_row.qw, sat_row.qx, sat_row.qy, sat_row.qz])
                t_sat = torch.tensor([sat_row.x, sat_row.y, sat_row.z], dtype=torch.float32)

                baseline = torch.ones([]) * 0.31

                data_dict = {
                    "left_image_path": os.path.join(scene_path, 'left_images', f"{rover_timestamp}.png"),
                    "right_image_path": os.path.join(scene_path, 'right_images', f"{rover_timestamp}.png"),
                    "K_left": K,
                    "R_left2world": R_left2world,
                    "t_left2world": t_left2world,
                    "x_offset_ratio": pair['x_offset_ratio'],
                    "y_offset_ratio": pair['y_offset_ratio'],

                    "sat_image_path": os.path.join(scene_path, 'sat_images', f"{sat_timestamp}.png"),
                    "K_sat": K, # Assuming same intrinsics
                    "R_sat2world": quat2R(quat_sat),
                    "t_sat2world": t_sat,

                    'baseline': baseline,
                    "left_depth_path": os.path.join(scene_path, 'depths', f"{rover_timestamp}.pfm"),
                }
                all_data_pairs.append(data_dict)
        
        # Save the final list to a .pth file
        if all_data_pairs:
            output_file = os.path.join(output_path, f"{split_name}_data.pth")
            torch.save(all_data_pairs, output_file)
            print(f"Successfully saved {split_name} set with {len(all_data_pairs)} samples to {output_file}")

    print("--- Finished Data Split Creation ---")


                           
        

if __name__ == '__main__':
    
    # process_and_interpolate_rover_poses()

    # generate_pairs(pair_offset_thresh=50, max_random_offset=20, max_rotate_deg=180)
    # create_pair_splits(train_scenes=[1,2,3,4,5,6,9],
    #                    val_scenes=[7],
    #                    test_scenes=[8])

    #generate_pairs(downsample_scale=1, batch_size=64)

    create_pair_splits(train_scenes=[1,2,3,4,5,6,9],
                       val_scenes=[7],
                       test_scenes=[8])