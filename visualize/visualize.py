import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from transform import image2world, world2image
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _visualize_model_output(
    left_image,
    sat_image,
    node_coords,
    node_depths,
    node_scores,
    match_sat_coords,
    K_left,
    R_left2world,
    t_left2world_pred,
    t_left2world_label,
    K_sat,
    R_sat2world,
    t_sat2world,
    save_path
):
    """
    Visualizes the output of the model with orientation arrows.

    Args:
        left_image (torch.Tensor): Left image in shape (3, H1, W1).
        sat_image (torch.Tensor): Satellite image in shape (3, H2, W2).
        left_image_score_map (torch.Tensor): Left image score map in shape (H1, W1).
        node_coords (torch.Tensor): N keypoint coordinates in the left image, shape (N, 2).
        node_depths (torch.Tensor): Depth of N keypoints in the left image, shape (N).
        match_sat_coords (torch.Tensor): N corresponding matching points in the satellite image, shape (N, 2).
        K_left (torch.Tensor): Left camera intrinsics, shape (3, 3).
        R_left2world (torch.Tensor): Left camera rotation to world, shape (3, 3).
        t_left2world_pred (torch.Tensor): Predicted left camera translation to world, shape (3).
        t_left2world_label (torch.Tensor): Ground truth left camera translation to world, shape (3).
        K_sat (torch.Tensor): Satellite camera intrinsics, shape (3, 3).
        R_sat2world (torch.Tensor): Satellite camera rotation to world, shape (3, 3).
        t_sat2world (torch.Tensor): Satellite camera translation to world, shape (3).
    """

    # Convert tensors to numpy arrays for visualization
    left_image_np = left_image.permute(1, 2, 0).cpu().numpy()
    sat_image_np = sat_image.permute(1, 2, 0).cpu().numpy()
    node_coords_np = node_coords.cpu().numpy()
    node_scores_np = node_scores.cpu().numpy()
    match_sat_coords_np = match_sat_coords.cpu().numpy()

    # Create the figure and subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax6 = fig.add_subplot(gs[1, 2:4])

    # 1. Left image
    ax1.imshow(left_image_np)
    ax1.set_title("Left Image")
    ax1.axis("off")

    # 2. Satellite image
    ax2.imshow(sat_image_np)
    ax2.set_title("Satellite Image")
    ax2.axis("off")

    # 3. Left image covered by score map
    ax3.imshow(left_image_np)
    scatter = ax3.scatter(node_coords_np[:, 0], node_coords_np[:, 1], c=node_scores_np, cmap='jet', s=20)
    ax3.set_title("Left Image with Keypoints")
    ax3.axis("off")
    # Add a colorbar to ax3
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(scatter, cax=cax, orientation="vertical")

    # 4. Satellite image with ground truth and predicted positions and orientations
    ax4.imshow(sat_image_np)
    
    # Project ground truth and predicted camera positions to satellite image
    # Unsqueeze to add batch and N dimensions for the world2image function
    gt_pos_world = t_left2world_label.unsqueeze(0).unsqueeze(0)
    pred_pos_world = t_left2world_pred.unsqueeze(0).unsqueeze(0)

    # Prepare batch dimensions for camera parameters
    K_sat_b = K_sat.unsqueeze(0)
    R_sat2world_b = R_sat2world.unsqueeze(0)
    t_sat2world_b = t_sat2world.unsqueeze(0)

    gt_pos_image = world2image(gt_pos_world, K_sat_b, R_sat2world_b, t_sat2world_b).squeeze().cpu().numpy()
    pred_pos_image = world2image(pred_pos_world, K_sat_b, R_sat2world_b, t_sat2world_b).squeeze().cpu().numpy()

    # Define camera forward vector and project its endpoint to get orientation
    arrow_length_world = 10.0  # Length of arrow in world units (e.g., meters)
    cam_forward_vec = torch.tensor([0.0, 0.0, 1.0], device=R_left2world.device) * arrow_length_world
    world_forward_vec = R_left2world @ cam_forward_vec
    
    # Project arrow endpoints
    gt_arrow_end_world = (t_left2world_label + world_forward_vec).unsqueeze(0).unsqueeze(0)
    pred_arrow_end_world = (t_left2world_pred + world_forward_vec).unsqueeze(0).unsqueeze(0)
    
    gt_arrow_end_image = world2image(gt_arrow_end_world, K_sat_b, R_sat2world_b, t_sat2world_b).squeeze().cpu().numpy()
    pred_arrow_end_image = world2image(pred_arrow_end_world, K_sat_b, R_sat2world_b, t_sat2world_b).squeeze().cpu().numpy()

    # Draw ground truth position and orientation
    ax4.scatter(gt_pos_image[0], gt_pos_image[1], s=100, c="yellow", marker="^", label="Ground Truth")
    ax4.arrow(gt_pos_image[0], gt_pos_image[1], 
              gt_arrow_end_image[0] - gt_pos_image[0], 
              gt_arrow_end_image[1] - gt_pos_image[1],
              color='yellow', head_width=10, length_includes_head=True)

    # Draw predicted position and orientation
    ax4.scatter(pred_pos_image[0], pred_pos_image[1], s=100, c="red", marker=".", label="Predicted")
    ax4.arrow(pred_pos_image[0], pred_pos_image[1],
              pred_arrow_end_image[0] - pred_pos_image[0],
              pred_arrow_end_image[1] - pred_pos_image[1],
              color='red', head_width=10, length_includes_head=True)

    ax4.set_title("Satellite View of Camera Position")
    ax4.legend()
    ax4.axis("off")

    # 5. Predicted matches
    # Convert images to BGR for OpenCV
    left_image_bgr = cv2.cvtColor((left_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    sat_image_bgr = cv2.cvtColor((sat_image_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    kp1 = [cv2.KeyPoint(p[0], p[1], 1) for p in node_coords_np]
    kp2 = [cv2.KeyPoint(p[0], p[1], 1) for p in match_sat_coords_np]
    matches = [cv2.DMatch(_i, _i, 0) for _i in range(len(kp1))]
    match_img_gt = cv2.drawMatches(
        left_image_bgr,
        kp1,
        sat_image_bgr,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ax5.imshow(cv2.cvtColor(match_img_gt, cv2.COLOR_BGR2RGB))
    ax5.set_title("Predicted Matches")
    ax5.axis("off")

    # 6. Ground truth matches
    # Project node coordinates from the left image to the satellite image using the predicted pose
    world_coords_gt = image2world(
        node_coords.unsqueeze(0),
        K_left.unsqueeze(0),
        R_left2world.unsqueeze(0),
        t_left2world_label.unsqueeze(0),
        node_depths.unsqueeze(0),
    )
    sat_coords_gt = world2image(
        world_coords_gt,
        K_sat.unsqueeze(0),
        R_sat2world.unsqueeze(0),
        t_sat2world.unsqueeze(0),
    )
    sat_coords_gt_np = sat_coords_gt.squeeze(0).cpu().numpy()
    
    kp2_gt = [cv2.KeyPoint(p[0], p[1], 1) for p in sat_coords_gt_np]
    match_img_gt = cv2.drawMatches(
        left_image_bgr,
        kp1,
        sat_image_bgr,
        kp2_gt,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    ax6.imshow(cv2.cvtColor(match_img_gt, cv2.COLOR_BGR2RGB))
    ax6.set_title("Ground Truth Matches")
    ax6.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)

    print(f"Visualization saved to {save_path}")

def visualize_model_output(model, data):

    batch_size = data['input']['left_image'].shape[0]
    assert batch_size == 1, "Currently only support batch size 1 for visualization."

    model.eval()
    with torch.no_grad():
        pred = model(data['input'])
    
    save_dir = './figures/model_output'
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(data['input']['left_image_path'][0])
    save_path = os.path.join(save_dir, file_name)

    _visualize_model_output(
        left_image=data['input']['left_image'][0],
        sat_image=data['input']['sat_image'][0],
        node_coords=pred['node_coords'][0].cpu(),
        node_depths=pred['node_depths'][0].cpu(),
        node_scores=pred['node_scores'][0].cpu(),
        match_sat_coords=pred['match_sat_coords'][0].cpu(),
        K_left=data['input']['K_left'][0],
        R_left2world=data['input']['R_left2world'][0],
        t_left2world_pred=pred['t_left2world'][0].cpu(),
        t_left2world_label=data['label']['t_left2world'][0],
        K_sat=data['input']['K_sat'][0],
        R_sat2world=data['input']['R_sat2world'][0],
        t_sat2world=data['input']['t_sat2world'][0],
        save_path=save_path
    )