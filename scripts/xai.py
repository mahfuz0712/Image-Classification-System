import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import os  # Importing the os module for handling file system and directory operations

# Setting TensorFlow environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def ensure_model_built(model, dummy_input_shape=(1, 200, 200, 3)):
    """Ensures the model is built by calling it with dummy input."""
    if not model.built:
        dummy_input = np.zeros(dummy_input_shape, dtype=np.float32)
        model(dummy_input)  # Call the model to define input/output tensors

def load_image(image_path, target_size):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_size)
        img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize
        return img, img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None

def grad_cam(model, image, layer_name):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        return heatmap.numpy()
    except AttributeError as e:
        if "The layer sequential has never been called" in str(e):
            print("Model not built. Please ensure the model is built before using Grad-CAM.")
        else:
            raise e  # Re-raise the error for other unexpected exceptions
    except Exception as e:
        print(f"Error calculating Grad-CAM: {e}")
        return None

def main():
    try:
        # Load the trained model
        model = load_model('../models/best.model.h5')

        ensure_model_built(model)

        # Path to the input image
        image_path = "../datasets/test_data/congenital_disorder/101.jpg"
        original_img, processed_img = load_image(image_path, target_size=(200, 200))

        if processed_img is None:
            print("\033[91mXAI cannot process on this system. Please use a higher-end configuration.\033[0m")
            return

        # Generate Grad-CAM heatmap
        layer_name = "conv2d_2"  # Update this based on your model's summary
        heatmap = grad_cam(model, processed_img, layer_name)

        if heatmap is None:
            print("\033[91mXAI cannot process on this system. Please use a a device with higher-end configuration.\033[0m")
            return

        # ... (rest of your code for overlaying heatmap, SHAP, LIME, etc.)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("\033[91mXAI cannot process on this system or Google Colab. Please use a device with higher-end configuration.\033[0m")

if __name__ == "__main__":
    main()