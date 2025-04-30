import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def overlay_gradcam_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays a Grad-CAM heatmap onto the original image.
    
    Args:
        image (np.ndarray): Original image in RGB, shape (H, W, 3), values in [0, 1] or [0, 255].
        heatmap (np.ndarray): Heatmap array, shape (H, W), values in [0, 1].
        alpha (float): Transparency for the overlay.
        colormap (int): OpenCV colormap to apply.

    Returns:
        np.ndarray: Image with heatmap overlay.
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)

    # Ensure image is in 0–255 and uint8
    if image.max() <= 1.0:
        image = np.uint8(255 * image)
    else:
        image = np.uint8(image)

    # Resize heatmap to image size (if necessary)
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def load_frozen_graph(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def grad_cam_frozen_graph(graph, input_tensor_name, output_tensor_name, conv_layer_name, image, class_index=None):
    # Access the required tensors
    input_tensor = graph.get_tensor_by_name(input_tensor_name + ":0")
    output_tensor = graph.get_tensor_by_name(output_tensor_name + ":0")
    conv_tensor = graph.get_tensor_by_name(conv_layer_name + ":0")

    with tf.compat.v1.Session(graph=graph) as sess:
        # Forward pass to get predictions and feature maps
        conv_output, predictions = sess.run([conv_tensor, output_tensor], feed_dict={input_tensor: image})

        if class_index is None:
            class_index = np.argmax(predictions)

        # Gradient of output w.r.t. conv features
        y_c = output_tensor[0, class_index]
        grads = tf.gradients(y_c, conv_tensor)[0]

        output, grads_val = sess.run([conv_tensor, grads], feed_dict={input_tensor: image})

        # Global average pooling of gradients
        weights = np.mean(grads_val[0], axis=(0, 1))

        # Compute weighted sum of feature maps
        cam = np.zeros(output.shape[1:3], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[0, :, :, i]

        # ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        return cam

graph = load_frozen_graph("frozen_graph_224.pb")
for op in graph.get_operations():
    print(op.name)

# Replace these names based on your model
input_name = "x"         # without :0
output_name = "Identity"  # classification output
conv_name = "model/last_conv/Conv2D"     # last convolutional layer

# I = cv2.imread("data_3dsar_pass1_az001_VV.bmp")
# I = cv2.imread("data_3dsar_pass1_az003_HV.bmp")
# I = cv2.imread("data_3dsar_pass5_az030_HV_style_1_numPulses_117.bmp")
I = cv2.imread("data_3dsar_pass5_az030_VV_style_1_numPulses_117.bmp")

I = I.astype(np.float32)
preprocessed_image = I / 255.0
preprocessed_image = cv2.resize(preprocessed_image, (224,224))
image = np.expand_dims(preprocessed_image, axis=0)  # Shape: (1, H, W, 3)
heatmap = grad_cam_frozen_graph(graph, input_name, output_name, conv_name, image)

# Assume `image` is in shape (1, H, W, 3), and `heatmap` is output from Grad-CAM
overlay = overlay_gradcam_on_image(image[0], heatmap)

# Display with matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.axis('off')
plt.title("Grad-CAM Overlay")
plt.show()
