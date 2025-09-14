import torch
import torch.nn.functional as F

@torch.jit.script
def cam2world(points: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of points from camera frame to world frame.

    Args:
        points (torch.Tensor): Points in camera frame, shape (B, N, 3).
        R_c2w (torch.Tensor): Rotation matrices from camera to world, shape (B, 3, 3).
        t_c2w (torch.Tensor): Translation vectors from camera to world, shape (B, 3).

    Returns:
        torch.Tensor: Points in world frame, shape (B, N, 3).
    """
    return points @ R_c2w.transpose(-2, -1) + t_c2w.unsqueeze(1)

@torch.jit.script
def world2cam(points: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of points from world frame to camera frame.

    Args:
        points (torch.Tensor): Points in world frame, shape (B, N, 3).
        R_c2w (torch.Tensor): Rotation matrices from camera to world, shape (B, 3, 3).
        t_c2w (torch.Tensor): Translation vectors from camera to world, shape (B, 3).

    Returns:
        torch.Tensor: Points in camera frame, shape (B, N, 3).
    """
    R_w2c = R_c2w.transpose(-2, -1)
    t_w2c = -R_w2c @ t_c2w.unsqueeze(-1)
    
    return points @ R_w2c.transpose(-2, -1) + t_w2c.transpose(-2, -1)

@torch.jit.script
def cam2image(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Batched projection of points from camera frame to image frame.

    Args:
        points (torch.Tensor): Points in camera frame, shape (B, N, 3).
        K (torch.Tensor): Camera intrinsic matrices, shape (B, 3, 3).

    Returns:
        torch.Tensor: Points in image frame (pixels), shape (B, N, 2).
    """
    # Transpose points to (B, 3, N) for matmul, then transpose back to (B, N, 3).
    image_points_hom = (K @ points.transpose(-2, -1)).transpose(-2, -1)
    
    # Perform perspective divide with stability guard.
    depth = image_points_hom[..., 2:3]
    # Prevent division by zero or near-zero depths.
    depth = depth + 1e-8
    
    return image_points_hom[..., :2] / depth

@torch.jit.script
def image2cam(points: torch.Tensor, K: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of points from image frame to camera frame using depth.

    Args:
        points (torch.Tensor): Points in image frame (pixels), shape (B, N, 2).
        K (torch.Tensor): Camera intrinsic matrices, shape (B, 3, 3).
        depth (torch.Tensor): Depth values for each point, shape (B, N).

    Returns:
        torch.Tensor: Points in camera frame, shape (B, N, 3).
    """
    # Add homogeneous coordinate of 1.
    points_homogeneous = F.pad(points, (0, 1), "constant", 1.0) # (B, N, 3)
    
    # `torch.linalg.inv` is batch-aware.
    inv_K = torch.linalg.inv(K)
    
    # Unproject to get normalized direction vectors in camera frame.
    cam_points_unscaled = (inv_K @ points_homogeneous.transpose(-2, -1)).transpose(-2, -1)
    
    # Scale by depth.
    return cam_points_unscaled * depth.unsqueeze(-1)

@torch.jit.script
def world2image(points: torch.Tensor, K: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of points from world frame to image frame.

    Args:
        points (torch.Tensor): Points in world frame, shape (B, N, 3).
        K (torch.Tensor): Camera intrinsic matrices, shape (B, 3, 3).
        R_c2w (torch.Tensor): Rotation matrices from camera to world, shape (B, 3, 3).
        t_c2w (torch.Tensor): Translation vectors from camera to world, shape (B, 3).

    Returns:
        torch.Tensor: Points in image frame (pixels), shape (B, N, 2).
    """
    points_cam = world2cam(points, R_c2w, t_c2w)
    return cam2image(points_cam, K)

@torch.jit.script
def image2world(points: torch.Tensor, K: torch.Tensor, R_c2w: torch.Tensor, t_c2w: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of points from image frame to world frame using depth.

    Args:
        points (torch.Tensor): Points in image frame (pixels), shape (B, N, 2).
        K (torch.Tensor): Camera intrinsic matrices, shape (B, 3, 3).
        R_c2w (torch.Tensor): Rotation matrices from camera to world, shape (B, 3, 3).
        t_c2w (torch.Tensor): Translation vectors from camera to world, shape (B, 3).
        depth (torch.Tensor): Depth values for each point, shape (B, N).

    Returns:
        torch.Tensor: Points in world frame, shape (B, N, 3).
    """
    points_cam = image2cam(points, K, depth)
    return cam2world(points_cam, R_c2w, t_c2w)

@torch.jit.script
def compose_transformation(R1: torch.Tensor, t1: torch.Tensor, R2: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched composition of two transformations: R1, t1 followed by R2, t2.
    The resulting transformation takes a point p and computes: R2 @ (R1 @ p + t1) + t2

    Args:
        R1 (torch.Tensor): First rotation matrix, shape (B, 3, 3).
        t1 (torch.Tensor): First translation vector, shape (B, 3).
        R2 (torch.Tensor): Second rotation matrix, shape (B, 3, 3).
        t2 (torch.Tensor): Second translation vector, shape (B, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Composed rotation and translation.
    """
    R_composed = R2 @ R1
    # Unsqueeze t1 for matmul, then add t2.
    t_composed = (R2 @ t1.unsqueeze(-1)).squeeze(-1) + t2
    return R_composed, t_composed
