import cv2
import os
from natsort import natsorted
from run_utils import load_model, get_transform, get_square_image, apply_threshold
import matplotlib.pyplot as plt
import torch
import numpy as np

image_folder = '/workspace/RUGD_frames/creek'

model = load_model()
transform = get_transform()
prompts = ['navigable pathway', 'stone pathway']

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files = natsorted(image_files)


for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = get_square_image(frame)

    img = transform(frame).unsqueeze(0)
    img = img.half()
    img = img.to('cuda')
    with torch.no_grad():
        preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]
    preds = [torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy() for i in range(len(prompts))]
    _, ax = plt.subplots(1, len(prompts)+1, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(frame)
    [ax[i+1].imshow(frame) for i in range(len(prompts))]
    [ax[i+1].imshow(np.float32(np.transpose(preds[i], (1, 2, 0))), alpha=0.5) for i in range(len(prompts))]
    [ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))]
    plt.savefig(f'/workspace/rugd_predictions/creek/{image_file}')
    print(image_file)
    plt.close()


    
