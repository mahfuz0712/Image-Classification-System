# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore

# def augment_images(input_dir, output_dir, target_size=(200, 200), num_augmented_images=2000):
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Data augmentation parameters
#     datagen = ImageDataGenerator(
#         rotation_range=40,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         fill_mode='reflect'
#     )

#     # Load all images from the input directory
#     image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     total_images = len(image_files)

#     print(f"Found {total_images} images in {input_dir}. Augmenting to {num_augmented_images} total images...")

#     # Calculate how many augmented images to generate per original image
#     images_per_original = num_augmented_images // total_images
#     count = 0  # Counter for generated images

#     for img_file in image_files:
#         # Load the image
#         img_path = os.path.join(input_dir, img_file)
#         img = load_img(img_path, target_size=target_size)
#         img_array = img_to_array(img)
#         img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

#         # Generate augmented images
#         for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix='sick', save_format='jpg'):
#             count += 1
#             if count >= num_augmented_images:
#                 print("Augmentation complete.")
#                 return
#             if count % images_per_original == 0:
#                 break
    
#     print(f"Augmented {count} images and saved to {output_dir}.")
#     print("copying augmented data to validation directory")
#     # copy augmented data to validation directory
#     for filename in os.listdir(output_dir):
#         file_path = os.path.join(output_dir, filename
#             shutil.copy(file_path, os.path.join(validation_dir, filename))
#             print("Copied Successfully.")
#             return
                                 



# # Paths to the input and output directories
# input_dir = "../datasets/test_data/sick"
# output_dir = "../datasets/augmented_data/sick"

# # Run the augmentation
# augment_images(input_dir, output_dir, target_size=(200, 200), num_augmented_images=2500)



import os
import shutil
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array # type: ignore

def augment_images(input_dir, output_dir, validation_dir, target_size=(200, 200), num_augmented_images=2000, file_name):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Data augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='reflect'
    )

    # Load all images from the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    print(f"Found {total_images} images in {input_dir}. Augmenting to {num_augmented_images} total images...")

    # Calculate how many augmented images to generate per original image
    images_per_original = num_augmented_images // total_images
    count = 0  # Counter for generated images

    for img_file in image_files:
        # Load the image
        img_path = os.path.join(input_dir, img_file)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension

        # Generate augmented images
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix=file_name, save_format='jpg'):
            count += 1
            if count >= num_augmented_images:
                print("Augmentation complete.")
                return
            if count % images_per_original == 0:
                break

    print(f"Augmented {count} images and saved to {output_dir}.")
    print("copying augmented data to validation directory...")
    # copy augmented data to validation directory
    for file_name in os.listdir(output_dir):
        source_file = os.path.join(output_dir, file_name)
        destination_file = os.path.join(validation_dir, file_name)
        # Copy only if it's a file
        if os.path.isfile(source_file):
            shutil.copy(source_file, destination_file)
            print(f"Copied: {file_name}")


# Paths to the input, output and validation directories
input_dir = "../datasets/test_data/sick"
output_dir = "../datasets/augmented_data/sick"
validation_dir = "../datasets/validation/sick"

# Run the augmentation
augment_images(input_dir, output_dir, validation_dir, target_size=(200, 200), num_augmented_images=2000, "sick_child")

augment_images("../datasets/test_data/healthy", "../datasets/augmented_data/healthy", "../datasets/validation/healthy", target_size=(200,200), num_augmented_images=2000, "healthy_child")
print("Augmentation & Transferation complete now run train.py to train the model")