import argparse
import rosbag
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from models.clipseg_gpu import CLIPDensePredT
import torch
import numpy as np
from run_utils import load_model, get_transform, get_square_image

def main(bag_file_path, prompts, topic):
    bridge = CvBridge()
    model = load_model()
    transform = get_transform()

    with rosbag.Bag(bag_file_path, 'r') as bag:
        for i, (topic_msg, msg, t) in enumerate(bag.read_messages()):
            if topic_msg == topic:
                cv_image = bridge.imgmsg_to_cv2(msg, "rgb8")
                cv_image = get_square_image(cv_image)
                cv_image = cv2.resize(cv_image, (int(cv_image.shape[0]/2), int(cv_image.shape[0]/2)))
                img = transform(cv_image).unsqueeze(0)

                img = img.half()
                img = img.to('cuda')
                with torch.no_grad():
                    preds = model(img.repeat(len(prompts), 1, 1, 1), prompts)[0]

                _, ax = plt.subplots(1, len(prompts) + 1, figsize=(15, 4))
                [a.axis('off') for a in ax.flatten()]
                ax[0].imshow(cv_image)
                [ax[i + 1].imshow(cv2.resize(np.float32(np.transpose(torch.sigmoid(preds[i][0]).cpu().unsqueeze(0).numpy(), (1, 2, 0))), (592, 592), interpolation=cv2.INTER_CUBIC)) for i in range(len(prompts))]
                [ax[i + 1].text(0, -15, prompts[i]) for i in range(len(prompts))]
                plt.savefig(f'images/{str(i).zfill(5)}.png')
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a ROS bag file and overlay image masks.')
    parser.add_argument('--bag_file_path', type=str, default='example_filtered.bag', help='Path to the ROS bag file.')
    parser.add_argument('--prompts', nargs='+', default=['navigable pathway', 'dirt road'], help='List of prompts for the model.')
    parser.add_argument('--topic', type=str, default='/nerian/right/image_raw', help='The topic to filter from the ROS bag.')
    parser.add_argument('--resize_scale', type=float, default=1, help='The topic to filter from the ROS bag.')

    args = parser.parse_args()
    main(args.bag_file_path, args.prompts, args.topic)