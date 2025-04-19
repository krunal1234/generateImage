import torch
import numpy as np
import cv2

def preprocess_image(image: np.ndarray, size: list) -> torch.Tensor:
    image_resized = cv2.resize(image, (size[1], size[0]))
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor

def postprocess_image(mask_tensor: torch.Tensor, shape: tuple) -> np.ndarray:
    mask_np = mask_tensor.detach().cpu().numpy()
    mask_resized = cv2.resize(mask_np, (shape[1], shape[0]))
    mask_resized = (mask_resized * 255).astype(np.uint8)
    return mask_resized