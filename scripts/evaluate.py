import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def evaluate_model():
    # Load the model
    model = load_model('../models/best.model.h5')  # Ensure you're loading the correct model

    # Prepare ImageDataGenerator for validation data
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalize pixel values to [0, 1]
    
    # Load validation data from directory (replace with the correct path)
    validation_dir = '../datasets/validation'  # Replace with your validation folder path
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),  # Resize images to the model's input size
        batch_size=32,
        class_mode='binary',  # Use 'binary' for binary classification
        shuffle=False  # Don't shuffle the validation data to keep the predictions aligned with the true labels
    )

    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")

    # Get predictions for classification report and other metrics
    y_val = validation_generator.classes  # True labels
    y_pred = model.predict(validation_generator, verbose=1).flatten()  # Predictions from the model (flattened for binary classification)
    y_pred_class = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class labels

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_val, y_pred_class, zero_division=1))

    # ROC-AUC (Only applicable for binary classification)
    if validation_generator.num_classes == 2:  # Binary classification
        roc_auc = roc_auc_score(y_val, y_pred)  # Use the probabilities for the positive class
        print(f"ROC-AUC Score: {roc_auc}")

        # Precision-Recall Curve (Only applicable for binary classification)
        precision, recall, _ = precision_recall_curve(y_val, y_pred)  # Use the probabilities for the positive class
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc}")
        print("Done evaluation now you can run xai.py for the AI Explainability ")

# Call the function to evaluate the model
evaluate_model()
