# Image Classification System
=====================Project Directory Struncture=====================
#### Step-1: Organize your project directory like this:
```
image_classification_system/
│
├── datasets/
│   ├── test_data/
│   │   ├── congenital_disorder/
│   │   │   ├── sick_child1.jpg
│   │   │   ├── sick_child2.jpg
│   │   │   └── ...
│   │   └── congenital_disorder_part_two/
│   ├── augmented_data/
│   │   ├── congenital_disorder_part_one/
│   │   │   ├── sick_child1.jpg
│   │   │   ├── sick_child2.jpg
│   │   │   └── ...
│   │   └── congenital_disorder_part_two/
│   ├── validation/
│       ├── congenital_disorder_part_one/
│       │   ├── sick_child1.jpg
│       │   ├── sick_child2.jpg
│       │   └── ...
│       |── congenital_disorder_part_two/
│
├── models/
│   └── best.model.h5 // this will be generated after you run train.py
│
├── scripts/
│   ├── augment.py
|   |── evaluate.py
│   ├── summary.py
│   ├── train.py
│   └── xai.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
#### Step-2: Set Up Your Virtual Environment
##### 1.  Create a virtual environment (optional but recommended):
```
python -m venv venv
```
###### Activate the environment:

* Windows: .\venv\Scripts\activate
* Mac/Linux: source venv/bin/activate

##### 2. Install the required packages:
```
pip install -r requirements.txt
```

#### Step-3: augment the test_data to create augmented_data
##### 1. Run the augment.py script to create augmented_data
```
cd scripts
python augment.py
```
This script will train the model using the training data and save the model to the models directory.

#### Step-4: train the Model
##### 1. Run the train.py script to evaluate the model:
```
python train.py
```
This script will train the model using the augmented_data and validation data and generate  the best.mode.h5 with accuracy of 90% and above 

#### Step-5: Generate Classification Report
##### 1. Run the evaluate.py script to see the report
```
python evaluate.py
```
This script will load the trained model and make classification repornt on the on the images present at the validation path.



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
│   ├── xai.py                   # Grad-CAM implementation
│   ├── xai.py                   # LIME implementation
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

### `xai.py`
- **Purpose:** Generates visual explanations using Grad-CAM.
- **Key Steps:**
  1. Identifies the last convolutional layer in the model.
  2. Computes a heatmap based on gradients for the predicted class.
  3. Superimposes the heatmap on the original image to highlight important regions.
  4. Saves the Grad-CAM visualization as `gradcam_heatmap.png`.


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
python scripts/xai.py
```
The output heatmap will be saved as `outputs/gradcam_heatmap.png`.


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

## Author
[Mohammad Mahfuz Rahman](https://github.com/mahfuz0712)

## License

This project is not open-source and is only available under a paid License.

---

**Happy Coding!**