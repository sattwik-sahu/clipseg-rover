import argparse
import cv2
import os
from natsort import natsorted
from run_utils import load_model, get_transform, get_square_image, apply_threshold
import matplotlib.pyplot as plt
import torch
import numpy as np

def process_images(image_folder, prompts, output_folder):
    model = load_model()
    transform = get_transform()

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = get_square_image(frame)

        img = transform(frame).unsqueeze(0)
        img = img.half()
        img = img.to('cuda')
        with torch.no_grad():
            preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]
        preds = [torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy() for i in range(len(prompts))]
        
        _, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, 4))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(frame)
        [ax[i + 1].imshow(frame) for i in range(len(prompts))]
        [ax[i + 1].imshow(np.float32(np.transpose(preds[i], (1, 2, 0))), alpha=0.5) for i in range(len(prompts))]
        [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))]
        
        output_path = os.path.join(output_folder, image_file)
        plt.savefig(output_path)
        print(f'Saved: {output_path}')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and overlay predictions.')
    parser.add_argument('--image_folder', type=str, default='/workspace/RUGD_frames/creek', help='Folder containing the images.')
    parser.add_argument('--prompts', nargs='+', default=['navigable pathway', 'stone pathway'], help='List of prompts for the model.')
    parser.add_argument('--output_folder', type=str, default='/workspace/rugd_predictions/creek', help='Folder to save the output images.')

    args = parser.parse_args()
    process_images(args.image_folder, args.prompts, args.output_folder)
