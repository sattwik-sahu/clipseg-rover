from torchvision import transforms
from models.clipseg_gpu import CLIPDensePredT
import torch
import numpy as np

def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device('cuda')), strict=False)
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize((352, 352)),
    ])

def get_square_image(image):
    height, width, _ = image.shape
    image = image[:, int(abs(height - width)/2):int(abs(height + width)/2)]
    return image

def apply_threshold(image, threshold):
    binary_image = np.where(image >= threshold, 1, 0).astype(np.uint8)
    return binary_image
