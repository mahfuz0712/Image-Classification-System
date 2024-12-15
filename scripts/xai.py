import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries





def ensure_model_built(model, dummy_input_shape=(1, 200, 200, 3)):
    """Ensures the model is built by calling it with dummy input."""
    if not model.built:
        dummy_input = np.zeros(dummy_input_shape, dtype=np.float32)
        model(dummy_input)  # Call the model to define input/output tensors
        model.summary()  # Print model summary to confirm build

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

        # Compile the model with a dummy optimizer and loss if not already compiled
        if not model.compiled_loss:
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        ensure_model_built(model)

        # Path to the input image
        image_path = "../datasets/test_data/congenital_disorder/101.jpg"
        original_img, processed_img = load_image(image_path, target_size=(200, 200))

        if processed_img is None:
            print("\033[91mXAI cannot process on this system or google colab. Please use a higher-end configuration.\033[0m")
            return

        # Generate Grad-CAM heatmap
        layer_name = "conv2d_1"  # Update this based on your model's summary
        heatmap = grad_cam(model, processed_img, layer_name)

        if heatmap is None:
            print("\033[91mXAI cannot process on this system or google colab. Please use a a device with higher-end configuration.\033[0m")
            return

        # Overlay the heatmap on the original image
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

        # Display the original image, heatmap, and superimposed image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        plt.title('Superimposed Image')
        plt.axis('off')

        plt.show()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("\033[91mXAI cannot process on this system or Google Colab. Please use a device with higher-end configuration.\033[0m")

if __name__ == "__main__":
    main()