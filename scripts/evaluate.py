import os  # Importing the os module for handling environment variables and file operations

# Disabling TensorFlow's oneDNN optimizations for potentially better performance on some systems
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Importing necessary libraries
import numpy as np  # For numerical operations
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc  # For evaluation metrics
from tensorflow.keras.models import load_model  # type: ignore # For loading the saved model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # For image preprocessing

def evaluate_model():
    """
    Function to evaluate the saved model using validation data and calculate various performance metrics.
    """
    # Load the pre-trained model from the specified path
    model = load_model('../models/best.model.h5')  # Ensure the correct model file path

    # Prepare an ImageDataGenerator instance for rescaling validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values to the range [0, 1]
    
    # Define the directory containing validation data
    validation_dir = '../datasets/validation'  # Replace with the correct validation data directory path
    
    # Create a data generator for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # Path to validation dataset
        target_size=(200, 200),  # Resize images to match the input size expected by the model
        batch_size=32,  # Number of images in each batch
        class_mode='binary',  # Specify 'binary' for binary classification tasks
        shuffle=False  # Disable shuffling to ensure predictions align with true labels
    )

    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(validation_generator)  # Get the loss and accuracy metrics
    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")  # Print the evaluation results

    # Obtain true labels from the validation generator
    y_val = validation_generator.classes  # True labels for the validation data
    
    # Generate predictions from the model
    y_pred = model.predict(validation_generator, verbose=1).flatten()  # Get predicted probabilities, flatten to a 1D array for binary classification
    
    # Convert predicted probabilities to binary class labels
    y_pred_class = (y_pred > 0.5).astype(int)  # Use a threshold of 0.5 for binary classification

    # Print a detailed classification report
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred_class, zero_division=1))  # Include precision, recall, F1-score, and support

    # Calculate and display ROC-AUC and Precision-Recall AUC (for binary classification only)
    if validation_generator.num_classes == 2:  # Ensure this is a binary classification task
        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y_val, y_pred)  # Use probabilities for calculating ROC-AUC
        print(f"ROC-AUC Score: {roc_auc}")

        # Calculate Precision-Recall curve and its AUC
        precision, recall, _ = precision_recall_curve(y_val, y_pred)  # Get precision and recall values
        pr_auc = auc(recall, precision)  # Calculate the area under the precision-recall curve
        print(f"Precision-Recall AUC: {pr_auc}")
        print("Done evaluation now you can run xai.py for the AI Explainability")  # Additional user prompt for next steps

# Call the evaluation function to execute the evaluation process
evaluate_model()
