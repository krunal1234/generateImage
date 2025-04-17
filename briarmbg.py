# briarmbg.py

from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

class BriaRMBG:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained DeepLabV3 model for background removal
        self.model = deeplabv3_resnet101(pretrained=True)  # This is a pre-trained model for segmentation
        self.model.to(self.device)
        self.model.eval()

    def remove_background(self, image: Image.Image) -> Image.Image:
        input_image = self.preprocess_image(image)
        
        with torch.no_grad():
            output = self.model(input_image)['out'][0]  # Get the segmentation output
        
        output_image = self.postprocess_image(output)
        return output_image

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def postprocess_image(self, output_tensor: torch.Tensor) -> Image.Image:
        # Convert the output tensor to a binary mask and apply it to the original image
        output_mask = torch.argmax(output_tensor, dim=0).cpu().numpy()
        output_mask = (output_mask == 15).astype(np.uint8) * 255  # Class 15 is for "person" in COCO dataset
        output_image = Image.fromarray(output_mask)
        return output_image
