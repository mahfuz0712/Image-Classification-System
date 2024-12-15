import os  # Importing the os module for directory and environment variable management

# Importing necessary functions for image processing and augmentation from TensorFlow Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array  # type: ignore

# Defining a function to augment images
def augment_images(fileName, input_dir, output_dir, validation_dir, target_size=(200, 200), num_augmented_images=2000):
    # Ensure the output directory exists; create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    
    # Define data augmentation parameters using ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=40,  # Randomly rotate images by up to 40 degrees
        shear_range=0.2,  # Randomly apply shearing transformations
        zoom_range=0.2,  # Randomly zoom into images
        horizontal_flip=True,  # Randomly flip images horizontally
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        fill_mode='reflect'  # Fill missing pixels using reflection
    )

    # Get a list of image files in the input directory with specific extensions
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)  # Count the total number of images found

    # Print the total number of images found and the target number of augmented images
    print(f"Found {total_images} images in {input_dir}. Augmenting to {num_augmented_images} total images...")

    # Calculate how many augmented images should be generated per original image
    images_per_original = num_augmented_images // total_images
    count = 0  # Initialize a counter for the total number of augmented images generated

    for img_file in image_files:
        # Construct the full path to the image
        img_path = os.path.join(input_dir, img_file)
        # Load the image with the specified target size
        img = load_img(img_path, target_size=target_size)
        # Convert the image to a NumPy array
        img_array = img_to_array(img)
        # Reshape the image array to add a batch dimension
        img_array = img_array.reshape((1,) + img_array.shape)

        # Generate augmented images in batches
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix=fileName, save_format='jpg'):
            count += 1  # Increment the counter
            # Stop generating images once the target number is reached
            if count >= num_augmented_images:
                print("Augmentation complete.")  # Notify when augmentation is complete
                return
            # Break after generating the desired number of images per original image
            if count % images_per_original == 0:
                break
        # Generate augmented images in batches
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=validation_dir, save_prefix=fileName, save_format='jpg'):
            count += 1  # Increment the counter
            # Stop generating images once the target number is reached
            if count >= num_augmented_images:
                print("Augmentation complete.")  # Notify when augmentation is complete
                return
            # Break after generating the desired number of images per original image
            if count % images_per_original == 0:
                break

    # Print the total number of images augmented
    print(f"Augmented {count} images and saved to {output_dir}. & {validation_dir}")

# Paths to the input, output, and validation directories for class 1 and class 2
input_dir1 = "../datasets/test_data/congenital_disorder"  # Input directory for class 1
input_dir2 = "../datasets/test_data/healthy"  # Input directory for class 2
output_dir1 = "../datasets/augmented_data/congenital_disorder"  # Output directory for augmented images of class 1
output_dir2 = "../datasets/augmented_data/healthy"  # Output directory for augmented images of class 2
validation_dir1 = "../datasets/validation/congenital_disorder"  # Validation directory for class 1
validation_dir2 = "../datasets/validation/healthy"  # Validation directory for class 2

# Run the augmentation for class 1
augment_images("sick_child", input_dir1, output_dir1, validation_dir1, target_size=(200, 200), num_augmented_images=2000)
# Run the augmentation for class 2
augment_images("sick_child", input_dir2, output_dir2, validation_dir2, target_size=(200, 200), num_augmented_images=2000)

# Notify that the augmentation process is complete and prompt to run the training script
print("run train.py")