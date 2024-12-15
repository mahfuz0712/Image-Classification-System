import tensorflow as tf  # Importing TensorFlow for deep learning tasks
# Importing necessary classes and functions from TensorFlow Keras for building and training models
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # For image preprocessing and augmentation
from tensorflow.keras import  models  # type: ignore # For layers and model building
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout  # type: ignore # Layers for CNN architecture
from tensorflow.keras.callbacks import  EarlyStopping  # type: ignore # For saving the model and stopping training early
from keras.callbacks import Callback # type: ignore
            
class AccuracyRangeCheckpoint(Callback):
    def __init__(self, filepath, min_acc, max_acc):
        super(AccuracyRangeCheckpoint, self).__init__()
        self.filepath = filepath
        self.min_acc = min_acc
        self.max_acc = max_acc

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('val_accuracy')
        if accuracy is not None:
            if self.min_acc < accuracy < self.max_acc:
                # Save the model if within range
                self.model.save(self.filepath)
                print(f"\nModel saved at epoch {epoch + 1} with val_accuracy: {accuracy:.4f}")
                
# Paths
base_dir = '../datasets'  # Base directory containing dataset folders
train_dir = os.path.join(base_dir, 'augmented_data')  # Path to the training dataset directory
validation_dir = os.path.join(base_dir, 'validation')  # Path to the validation dataset directory

# Image dimensions and parameters
img_width, img_height = 200, 200  # Dimensions to which input images will be resized
batch_size = 32  # Number of images processed in a batch


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
    Input(shape=(200, 200, 3)),  # Define input shape explicitly
    Conv2D(32, (3, 3), activation='relu'),  # 1st convolutional layer with 32 filters
    MaxPooling2D(2, 2),  # 1st max pooling layer to reduce spatial dimensions
    Conv2D(64, (3, 3), activation='relu'),  # 2nd convolutional layer with 64 filters
    MaxPooling2D(2, 2),  # 2nd max pooling layer
    Conv2D(128, (3, 3), activation='relu'),  # 3rd convolutional layer with 128 filters
    MaxPooling2D(2, 2),  # 3rd max pooling layer
    Flatten(),  # Flatten the feature maps into a 1D vector for the dense layers
    Dense(512, activation='relu'),  # Fully connected dense layer with 512 neurons
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compiling the model with loss function, optimizer, and evaluation metrics
model.compile(
    loss='binary_crossentropy',  # Binary cross-entropy loss for binary classification
    optimizer=tf.keras.optimizers.Adam(),  # Adam optimizer for efficient training
    metrics=['accuracy']  # Evaluate model performance using accuracy
)

# Setting up callbacks
# Early stopping callback to stop training if validation loss doesn't improve for 5 epochs
early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=5, 
    restore_best_weights=True,
    verbose=1,
)

callbacks = [
    early_stop,
    AccuracyRangeCheckpoint('../models/best.model.h5', min_acc=0.9600, max_acc=0.9800),
]

# Training the model using the training and validation generators
history = model.fit(
    train_generator,  # Training data generator
    steps_per_epoch=train_generator.samples // batch_size,  # Number of steps per epoch
    validation_data=val_generator,  # Validation data generator
    validation_steps=val_generator.samples // batch_size,  # Number of validation steps per epoch
    epochs=20,  # Maximum number of epochs for training
    callbacks=callbacks # List of callbacks; early stopping in this case
)

# Save the final model to the specified path
model.save('../models/best.model.h5')  # Save the best model as an h5 file
print("Training complete. Best model saved as 'best.model.h5' in models directory.\nRun evaluate.py to see the classification report")  # Print completion message