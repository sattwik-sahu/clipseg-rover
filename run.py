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

# Initialize the model
def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
    return model

# Image transformation pipeline
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])

# Process the image and predict
def process_image(input_image, model, transform, prompts):
    # Normalize and resize image
    img = transform(input_image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        preds = model(img.repeat(4, 1, 1, 1), prompts)[0]
    
    # Visualize prediction
    _, ax = plt.subplots(1, 5, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(input_image)
    [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
    [ax[i + 1].text(0, -15, prompts[i]) for i in range(4)]
    plt.show()

    return preds

# Publish predictions
def publish_predictions(preds, bridge, image_pub):
    for i, pred in enumerate(preds):
        # Convert the prediction to a numpy array
        pred_np = torch.sigmoid(pred[0]).cpu().numpy() * 255
        pred_np = pred_np.astype(np.uint8)
        
        # Convert the numpy array to an OpenCV image
        pred_img = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
        
        try:
            # Convert OpenCV image to ROS Image message
            ros_pred_img = bridge.cv2_to_imgmsg(pred_img, encoding="bgr8")
            ros_pred_img.header.stamp = rospy.Time.now()
            
            # Publish the image
            image_pub.publish(ros_pred_img)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

# Callback function for image topic
def image_callback(ros_image, model, transform, prompts, bridge, image_pub):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        
        # Convert OpenCV image (BGR) to PIL image (RGB)
        input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Process the image and get predictions
        preds = process_image(input_image, model, transform, prompts)
        
        # Publish the predictions
        publish_predictions(preds, bridge, image_pub)
        
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

def main():
    rospy.init_node('image_processor', anonymous=True)
    
    # Initialize CvBridge
    bridge = CvBridge()
    
    # Prompts
    prompts = ['straight line', 'something to fill', 'wood', 'a jar']
    
    # Load model and transformation pipeline
    model = load_model()
    transform = get_transform()
    
    # Publisher for the predicted images
    image_pub = rospy.Publisher("/predicted_images", RosImage, queue_size=10)
    
    # Subscribe to the image topic
    rospy.Subscriber("/camera/rgb/image_raw", RosImage, image_callback, (model, transform, prompts, bridge, image_pub))
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
