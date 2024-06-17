import torch
import rospy
import cv2
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import pyrealsense2 

def convert_depth_to_phys_coord_using_realsense(x, y, depth, cameraInfo):
    _intrinsics = pyrealsense2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = pyrealsense2.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    result = pyrealsense2.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    #result[0]: right, result[1]: down, result[2]: forward
    return result[2], -result[0], -result[1]
# Initialize the model
def load_model():
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=16)
    model.eval()
    model.load_state_dict(torch.load('weights/rd16-uni.pth', map_location=torch.device('cpu')), strict=False)
    return model

# Export the model to ONNX
def export_model_to_onnx(model, dummy_input, onnx_file_path):
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(f"Model exported to {onnx_file_path}")

# Quantize the ONNX model
def quantize_model(onnx_file_path, quantized_model_path):
    quantize_dynamic(onnx_file_path, quantized_model_path, weight_type=QuantType.QUInt8)
    print(f"Model quantized and saved to {quantized_model_path}")

# Load the ONNX model
def load_onnx_model(onnx_file_path):
    ort_session = ort.InferenceSession(onnx_file_path)
    return ort_session

# Image transformation pipeline
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])

# Process the image and predict
def process_image(input_image, ort_session, transform, prompts):
    # Normalize and resize image
    img = transform(input_image).unsqueeze(0).numpy()
    
    # Predict
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = torch.tensor(ort_outs[0])
    
    # Visualize prediction
    # _, ax = plt.subplots(1, 5, figsize=(15, 4))
    # [a.axis('off') for a in ax.flatten()]
    # ax[0].imshow(input_image)
    # [ax[i + 1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
    # [ax[i + 1].text(0, -15, prompts[i]) for i in range(4)]
    # plt.show()

    return preds

# Publish predictions
def publish_predictions(preds, bridge, image_pub):
    # Convert the prediction to a numpy array
    pred_np = torch.sigmoid(preds[0][0]).cpu().numpy() * 255
    pred_np = pred_np.astype(np.uint8)
    
    # Convert the numpy array to an OpenCV image
    pred_img = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
    
    try:
        # Convert OpenCV image to ROS Image message
        ros_pred_img = bridge.cv2_to_imgmsg(pred_img, encoding="bgr8")
        ros_pred_img.header.stamp = rospy.Time.now()
        
        # Publish the image
        # image_pub.publish(ros_pred_img)
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
    msg = String()
    msg.data = "msg"
    image_pub.publish(msg)

def depth_callback(msg, args):
    bridge = args[0]
    cv_image = bridge.imgmsg_to_cv2(msg, "passthrough")
    

# Callback function for image topic
def image_callback(ros_image, args):
    ort_session = args[0]
    transform = args[1]
    prompts = args[2]
    bridge = args[3]
    image_pub = args[4]

    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
        
        # Convert OpenCV image (BGR) to PIL image (RGB)
        input_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Process the image and get predictions
        preds = process_image(input_image, ort_session, transform, prompts)
        
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
    
    # Dummy input for exporting the model
    dummy_input = torch.randn(1, 3, 352, 352)
    
    # Export model to ONNX
    onnx_file_path = 'model.onnx'
    export_model_to_onnx(model, dummy_input, onnx_file_path)
    # Quantize the ONNX model
    quantized_model_path = 'model_quantized.onnx'
    quantize_model(onnx_file_path, quantized_model_path)

    # Load quantized ONNX model for inference
    ort_session = load_onnx_model(quantized_model_path)
    
    # Load ONNX model for inference
    # ort_session = load_onnx_model(onnx_file_path)
    
    # Publisher for the predicted images
    image_pub = rospy.Publisher("/predicted_images", String, queue_size=1)
    
    # Subscribe to the image topic
    rospy.Subscriber("/camera/color/image_raw", RosImage, image_callback, (ort_session, transform, prompts, bridge, image_pub),  queue_size=1, buff_size=2**24)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", RosImage, depth_callback, (bridge, ),  queue_size=1, buff_size=2**24)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
