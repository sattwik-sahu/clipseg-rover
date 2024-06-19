import os
import rosbag
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from models.clipseg import CLIPDensePredT
import torch
import numpy as np
from run_utils import load_model, get_transform


bag_file_path = 'example_filtered.bag'

bridge = CvBridge()
prompts = ['navigable pathway', 'dirt road']
model = load_model()
transform = get_transform()
i = 0

with rosbag.Bag(bag_file_path, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == '/nerian/right/image_raw':
            PREDS = []
            cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cv_image[:592, int(400-592/2):int(400+592/2), :]
            img = transform(cv_image).unsqueeze(0)
                
            # img = img.half()
            # img = img.to('cuda')
            with torch.no_grad():
                preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
            _, ax = plt.subplots(1, len(prompts)+1, figsize=(15, 4))
            [a.axis('off') for a in ax.flatten()]
            ax[0].imshow(cv_image)
            [ax[i+1].imshow(cv2.resize(np.float32(np.transpose(torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy(), (1, 2, 0))), (592, 592), interpolation = cv2.INTER_CUBIC)) for i in range(len(prompts))]
            [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))]
            plt.savefig(f'images/{str(i).zfill(5)}.png')
            plt.close()
            i+= 1

