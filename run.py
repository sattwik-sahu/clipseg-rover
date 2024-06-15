import torch
import rospy
import cv2
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

class ROSImageProcessor:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('image_processor', anonymous=True)
        
        # Subscribe to the image topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", RosImage, self.image_callback)
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Publisher for the predicted images
        self.image_pub = rospy.Publisher("/predicted_images", RosImage, queue_size=10)
        
        # Load model
        self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
        self.model.eval()
        self.model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
        
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)),
        ])
        
        # Prompts
        self.prompts = ['straight line', 'something to fill', 'wood', 'a jar']
    
    def image_callback(self, ros_image):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            
            # Convert OpenCV image (BGR) to PIL image (RGB)
            input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Process the image
            self.process_image(input_image)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def process_image(self, input_image):
        # Normalize and resize image
        img = self.transform(input_image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            preds = self.model(img.repeat(4, 1, 1, 1), self.prompts)[0]
        
        # Visualize prediction
        _, ax = plt.subplots(1, 5, figsize=(15, 4))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(input_image)
        [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
        [ax[i + 1].text(0, -15, self.prompts[i]) for i in range(4)]
        plt.show()

        # Publish the predicted images
        self.publish_predictions(preds)

    def publish_predictions(self, preds):
        for i, pred in enumerate(preds):
            # Convert the prediction to a numpy array
            pred_np = torch.sigmoid(pred[0]).cpu().numpy() * 255
            pred_np = pred_np.astype(np.uint8)
            
            # Convert the numpy array to an OpenCV image
            pred_img = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
            
            try:
                # Convert OpenCV image to ROS Image message
                ros_pred_img = self.bridge.cv2_to_imgmsg(pred_img, encoding="bgr8")
                ros_pred_img.header.stamp = rospy.Time.now()
                
                # Publish the image
                self.image_pub.publish(ros_pred_img)
            except CvBridgeError as e:
                rospy.logerr(f"CvBridge Error: {e}")

if __name__ == '__main__':
    try:
        processor = ROSImageProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
