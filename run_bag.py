import os
import rosbag
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.clipseg_old import CLIPDensePredT
import torch
import numpy as np

def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)
    return model

# Image transformation pipeline
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
# Specify the path to the ROS bag file and the output directory
bag_file_path = 'example_filtered.bag'


# Initialize CvBridge
bridge = CvBridge()
prompts = ['navigable pathway']
model = load_model()
transform = get_transform()
i = 0
# Open the ROS bag
with rosbag.Bag(bag_file_path, 'r') as bag:
    # Iterate through the messages in the bag
    for topic, msg, t in bag.read_messages():
        if topic == '/nerian/right/image_raw':
            cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            img = transform(input_image).unsqueeze(0)
            
            img = img.half()
            img = img.to('cuda')
            with torch.no_grad():
                preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
            # for i in range(1):
            #     print(np.transpose(torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy(), (1, 2, 0)).shape)
            
            # visualize prediction
            _, ax = plt.subplots(1, len(prompts)+1, figsize=(15, 4))
            [a.axis('off') for a in ax.flatten()]
            ax[0].imshow(input_image)
            [ax[i+1].imshow(cv2.resize(np.float32(np.transpose(torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy(), (1, 2, 0))), (msg.width, msg.height), interpolation = cv2.INTER_CUBIC)) for i in range(len(prompts))];
            [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))];
            plt.savefig(f'images/{str(i).zfill(5)}.png')
            plt.close()
            i+= 1
            # if i == 20:
            #     break

print("Extraction complete.")
