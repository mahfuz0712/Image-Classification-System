import os  # Importing the os module for handling file system and directory operations

# Setting TensorFlow environment variable to disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf  # Importing TensorFlow for deep learning tasks
# Importing necessary classes and functions from TensorFlow Keras for building and training models
from tensorflow.keras.models import Sequential  # To define a sequential model
from tensorflow.keras import layers, models  # For layers and model building
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layers for CNN architecture
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image preprocessing and augmentation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # For saving the model and stopping training early

# Paths
base_dir = '../datasets'  # Base directory containing dataset folders
train_dir = os.path.join(base_dir, 'augmented_data')  # Path to the training dataset directory
validation_dir = os.path.join(base_dir, 'validation')  # Path to the validation dataset directory

# Image dimensions and parameters
img_width, img_height = 200, 200  # Dimensions to which input images will be resized
batch_size = 32  # Number of images processed in a batch
num_classes = 2  # Number of output classes (update if more classes are added)

# Data generators for loading and preprocessing images
# Training data generator with image rescaling to normalize pixel values between 0 and 1
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
# Validation data generator with the same rescaling
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Generating training data batches from the training directory
train_generator = train_datagen.flow_from_directory(
    train_dir,  # Path to the training directory
    target_size=(img_width, img_height),  # Resize images to specified dimensions
    batch_size=batch_size,  # Set batch size
    class_mode='binary'  # Classification type; 'binary' for two classes
)

# Generating validation data batches from the validation directory
val_generator = val_datagen.flow_from_directory(
    validation_dir,  # Path to the validation directory
    target_size=(img_width, img_height),  # Resize images to specified dimensions
    batch_size=batch_size,  # Set batch size
    class_mode='binary'  # Classification type; 'binary' for two classes
)

# Defining the Convolutional Neural Network (CNN) model
model = models.Sequential([  # Creating a sequential model
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),  # 1st convolutional layer with 32 filters
    layers.MaxPooling2D(2, 2),  # 1st max pooling layer to reduce spatial dimensions
    layers.Conv2D(64, (3, 3), activation='relu'),  # 2nd convolutional layer with 64 filters
    layers.MaxPooling2D(2, 2),  # 2nd max pooling layer
    layers.Conv2D(128, (3, 3), activation='relu'),  # 3rd convolutional layer with 128 filters
    layers.MaxPooling2D(2, 2),  # 3rd max pooling layer
    layers.Flatten(),  # Flatten the feature maps into a 1D vector for the dense layers
    layers.Dense(512, activation='relu'),  # Fully connected dense layer with 512 neurons
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compiling the model with loss function, optimizer, and evaluation metrics
model.compile(
    loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
    optimizer=tf.keras.optimizers.Adam(),  # Adam optimizer for efficient training
    metrics=['accuracy']  # Evaluate model performance using accuracy
)

# Setting up callbacks
# Early stopping callback to stop training if validation loss doesn't improve for 5 epochs
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training the model using the training and validation generators
history = model.fit(
    train_generator,  # Training data generator
    steps_per_epoch=train_generator.samples // batch_size,  # Number of steps per epoch
    validation_data=val_generator,  # Validation data generator
    validation_steps=val_generator.samples // batch_size,  # Number of validation steps per epoch
    epochs=100,  # Maximum number of epochs for training
    callbacks=[early_stop]  # List of callbacks; early stopping in this case
)

# Save the final model to the specified path
model.save('../models/best.model.h5')  # Save the best model as an HDF5 file
print("Training complete. Best model saved as 'best.model.h5' in models directory. Run evaluate.py to see the classification report")  # Print completion message
