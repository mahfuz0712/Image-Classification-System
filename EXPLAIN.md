
# Explainable AI Project: Binary Image Classification with Grad-CAM and LIME

## Project Overview
This project involves training a binary image classification model using convolutional neural networks (CNNs) and implementing Explainable AI (XAI) techniques to interpret the model's predictions. The trained model identifies images belonging to two classes and provides visual explanations using Grad-CAM and LIME.

---

## Project Structure
```
├── datasets/
│   ├── augmented_data/          # Training dataset
│   ├── validation/              # Validation dataset
├── models/
│   ├── best.model.h5            # Trained Keras model
├── scripts/
│   ├── train.py                 # Training the model
│   ├── evaluate.py              # Model evaluation
│   ├── xai_gradcam.py           # Grad-CAM implementation
│   ├── xai_lime.py              # LIME implementation
├── outputs/
│   ├── gradcam_heatmap.png      # Grad-CAM output
│   ├── lime_explanation.png     # LIME output
├── EXPLAIN.md                   # Project documentation
```

---

## Requirements
Ensure the following libraries are installed before running the scripts:
- TensorFlow
- NumPy
- Matplotlib
- scikit-image
- LIME (for local interpretable model explanations)
- scikit-learn

Install all dependencies with:
```bash
pip install tensorflow numpy matplotlib scikit-learn lime
```

---

## Scripts Explanation

### `train.py`
- **Purpose:** Trains a binary image classification model using a CNN.
- **Key Steps:**
  1. Loads and preprocesses training and validation datasets.
  2. Defines the CNN architecture with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout` layers.
  3. Compiles the model with the Adam optimizer and binary cross-entropy loss.
  4. Implements early stopping to avoid overfitting.
  5. Saves the best-performing model as `best.model.h5`.

### `evaluate.py`
- **Purpose:** Evaluates the model on validation data.
- **Key Steps:**
  1. Loads the trained model.
  2. Prepares validation data for evaluation.
  3. Calculates loss, accuracy, precision, recall, F1-score, ROC-AUC, and Precision-Recall AUC.
  4. Outputs a classification report and metrics.

### `xai_gradcam.py`
- **Purpose:** Generates visual explanations using Grad-CAM.
- **Key Steps:**
  1. Identifies the last convolutional layer in the model.
  2. Computes a heatmap based on gradients for the predicted class.
  3. Superimposes the heatmap on the original image to highlight important regions.
  4. Saves the Grad-CAM visualization as `gradcam_heatmap.png`.

### `xai_lime.py`
- **Purpose:** Generates explanations using the LIME library.
- **Key Steps:**
  1. Creates a LIME explainer instance.
  2. Segments the input image into superpixels.
  3. Highlights superpixels that contribute most to the model's decision.
  4. Saves the LIME visualization as `lime_explanation.png`.

---

## How to Run

### 1. Train the Model
Run the training script to create a binary image classifier:
```bash
python scripts/train.py
```
This will save the trained model in the `models/` directory.

### 2. Evaluate the Model
Evaluate the model's performance on the validation dataset:
```bash
python scripts/evaluate.py
```
This will print evaluation metrics and save a classification report.

### 3. Explain Predictions
#### Grad-CAM Explanation
Run the Grad-CAM script to generate heatmaps for predictions:
```bash
python scripts/xai_gradcam.py
```
The output heatmap will be saved as `outputs/gradcam_heatmap.png`.

#### LIME Explanation
Run the LIME script to generate visual explanations for predictions:
```bash
python scripts/xai_lime.py
```
The output explanation will be saved as `outputs/lime_explanation.png`.

---

## Outputs
- **Grad-CAM Heatmap:** Highlights image regions critical for the model's predictions.
- **LIME Explanation:** Explains model predictions using superpixel segmentation.

---

## Grad-CAM Integration Details
Ensure that the `layer_name` in the Grad-CAM script matches the name of a convolutional layer in the model. Use `model.summary()` to inspect the layer names and update the `grad_cam()` function accordingly.

---

## Conclusion
This project combines CNN-based binary classification with interpretability techniques to provide insights into model decisions. Grad-CAM and LIME visualizations enable users to trust and understand the model's predictions.
