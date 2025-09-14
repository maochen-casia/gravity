from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
import re
import torch
from typing import Optional
import matplotlib.pyplot as plt
import cv2

def read_image(file_path, size):
    transform = Compose([Resize(size), ToTensor()])
    with Image.open(file_path).convert('RGB') as image:
        image = transform(image)
    return image

def read_depth(file_path, size):
    with open(file_path, 'rb') as f:
        header = f.readline().decode().rstrip()

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        # 处理缩放因子和字节序
        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        # 读取二进制数据并转换为NumPy数组
        data = np.fromfile(f, endian + 'f')
        shape = (height, width)
        img = np.reshape(data, shape)
        #img = np.flipud(img)  # 翻转图像方向

    depth = torch.tensor(img).reshape(1, 1, height, width)
    depth = torch.nn.functional.interpolate(depth, (size, size), mode='bilinear', align_corners=False)
    return depth.squeeze()

def visualize_image(image: torch.Tensor, save_path: str):
    image = image.permute(1,2,0).detach().cpu().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_path)



def tensor_to_cv2_img(tensor):
    """
    Converts a PyTorch tensor to an OpenCV image (NumPy array).

    Args:
        tensor (torch.Tensor): The input tensor. Assumes the tensor is in (C, H, W) format and RGB color space.

    Returns:
        numpy.ndarray: The image as a NumPy array in (H, W, C) format and BGR color space.
    """
    # Move tensor to CPU if it's on a GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # Convert tensor to a NumPy array
    numpy_image = tensor.numpy()

    # If the tensor has a batch dimension, remove it
    if numpy_image.ndim == 4:
        numpy_image = np.squeeze(numpy_image, axis=0)

    # Transpose the array from (C, H, W) to (H, W, C)
    numpy_image = np.transpose(numpy_image, (1, 2, 0))

    # Convert from RGB to BGR color space for OpenCV
    # Also, scale the pixel values to the 0-255 range if they are floats (0.0-1.0)
    if numpy_image.dtype == np.float32 or numpy_image.dtype == np.float64:
        numpy_image = (numpy_image * 255).astype(np.uint8)

    image_bgr = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return image_bgr

def visualize_tensors_with_arrow(ground_tensor, satellite_tensor, start_point, end_point, save_path=None):
    """
    Visualizes a ground-level and satellite image from PyTorch tensors side-by-side,
    with a red arrow drawn on the satellite image.

    Args:
        ground_tensor (torch.Tensor): The tensor for the ground-level image.
        satellite_tensor (torch.Tensor): The tensor for the satellite image.
        start_point (tuple): The (x, y) coordinates for the start of the arrow.
        end_point (tuple): The (x, y) coordinates for the end of the arrow.
    """
    # Convert tensors to OpenCV images
    ground_img = tensor_to_cv2_img(ground_tensor)
    satellite_img = tensor_to_cv2_img(satellite_tensor)

    h_sat, w_sat = satellite_img.shape[:2]
    center_coordinates = (w_sat // 2, h_sat // 2)
    point_radius = 5  # The radius of the center point
    point_color = (0, 0, 255)  # BGR color for red
    point_thickness = -1  # Thickness of -1 fills the circle

    # Draw a red point (filled circle) in the center of the satellite image
    cv2.circle(satellite_img, center_coordinates, point_radius, point_color, point_thickness)

    # Draw a red arrow on the satellite image
    satellite_img_with_arrow = cv2.arrowedLine(satellite_img, start_point, end_point, (0, 0, 255), thickness=2, tipLength=0.3)

    # Resize images to have the same height for side-by-side display
    h1, w1 = ground_img.shape[:2]
    h2, w2 = satellite_img_with_arrow.shape[:2]

    if h1 != h2:
        if h1 > h2:
            new_w1 = int(w1 * (h2 / h1))
            ground_img_resized = cv2.resize(ground_img, (new_w1, h2))
            satellite_img_resized = satellite_img_with_arrow
        else:
            new_w2 = int(w2 * (h1 / h2))
            satellite_img_resized = cv2.resize(satellite_img_with_arrow, (new_w2, h1))
            ground_img_resized = ground_img
    else:
        ground_img_resized = ground_img
        satellite_img_resized = satellite_img_with_arrow

    # Combine the two images horizontally
    combined_image = cv2.hconcat([ground_img_resized, satellite_img_resized])

    # Display the final image
    if save_path is not None:
        cv2.imwrite(save_path, combined_image)
    else:
        cv2.imshow('Ground and Satellite View', combined_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
