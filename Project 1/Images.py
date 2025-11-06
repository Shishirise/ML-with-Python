```python
# Import TensorFlow, the main deep learning library for training neural networks.
import tensorflow as tf

# Import matplotlib for plotting training accuracy and loss graphs.
import matplotlib.pyplot as plt

# Import os to work with folders and file paths on your computer.
import os

# Import Pillow tools for opening, resizing, and checking image files.
from PIL import Image, UnidentifiedImageError

# Import warnings to hide unnecessary TensorFlow or Python alerts.
import warnings


# Ignore all warning messages so the console output looks clean and readable.
warnings.filterwarnings('ignore')

# Suppress low-level TensorFlow logs that are not useful for normal users.
tf.get_logger().setLevel('ERROR')


# Define the main folder that holds your Cat and Dog image data.
data_dir = '/Users/shishiradhikari/Desktop/ImageClassification'

# List the two subfolders that represent the classes in your dataset.
folders = ["Cat", "Dog"]

# Notify the user that a data-cleaning scan is starting.
print("Checking for corrupted images (deep scan)...")

# Initialize a counter for the number of corrupted or unreadable images.
bad_files = 0

# Loop through each class folder (Cat and Dog).
for folder in folders:
    # Build the complete path for each folder.
    folder_path = os.path.join(data_dir, folder)
    
    # Loop through every file inside the folder.
    for filename in os.listdir(folder_path):
        # Create a full path for each individual image file.
        file_path = os.path.join(folder_path, filename)
        
        # Skip files that are not images based on their file extensions.
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        # Try to open and process the image.
        try:
            with Image.open(file_path) as img:
                # Convert the image to RGB and resize it to 150x150 pixels.
                img.convert("RGB").resize((150,150))
                # Save the image back as a clean JPEG file.
                img.save(file_path, "JPEG", quality=95)
        
        # If the image cannot be opened or decoded, remove it.
        except (OSError, UnidentifiedImageError, ValueError):
            print(f"Removing unreadable file: {file_path}")
            os.remove(file_path)
            bad_files += 1

# After scanning, print the number of cleaned or removed files.
if bad_files == 0:
    print("No corrupted files found.\n")
else:
    print(f"Cleaned {bad_files} corrupted files.\n")


# Inform that the dataset is now being loaded into TensorFlow.
print("Loading dataset...")

# Load the training portion of data (80%) from the folder structure.
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# Load the remaining 20% of data as validation images.
val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

# Display the detected class names, which are derived from the folder names.
classes = train_data.class_names
print("Classes detected:", classes)


# Create a normalization layer to scale pixel values from 0–255 to 0–1.
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization and parallel loading to the training dataset.
train_data = train_data.map(lambda x, y: (normalization_layer(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

# Apply normalization and prefetching to the validation dataset as well.
val_data = val_data.map(lambda x, y: (normalization_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

# Enable caching, shuffling, and background prefetching to speed up training.
train_data = train_data.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_data = val_data.cache().prefetch(tf.data.AUTOTUNE)


# Build the CNN architecture that will learn to classify cats and dogs.
model = tf.keras.Sequential([
    # First convolutional layer to extract basic features.
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    # Pooling layer to reduce image dimensions.
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Second convolutional layer to detect more complex shapes.
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # Pooling again to reduce computation.
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Third convolutional layer for deeper pattern learning.
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    # Third pooling layer to simplify feature maps.
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten layer to convert the 2D features into a 1D vector.
    tf.keras.layers.Flatten(),
    
    # Fully connected layer that combines learned features.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer that prevents overfitting by ignoring some neurons randomly.
    tf.keras.layers.Dropout(0.3),
    
    # Output layer with sigmoid activation for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model with the Adam optimizer and binary loss function.
# Accuracy is tracked to monitor how well the model performs.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print a summary of all layers and their parameters for reference.
print("\nModel Summary:")
model.summary()


# Start training the CNN using the training and validation datasets.
# Each epoch means one full pass through the entire training set.
print("\nTraining started...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=1
)
print("\nTraining complete!\n")


# Create two side-by-side graphs to visualize accuracy and loss changes.
plt.figure(figsize=(10,4))

# Plot accuracy for both training and validation.
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss for both training and validation datasets.
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Display sample predictions on a few images from the validation set.
# Each image will show the true and predicted labels with color coding.
print("Displaying predictions...")
for images, labels in val_data.take(1):
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype("int32")

    plt.figure(figsize=(10,10))
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy(), vmin=0, vmax=1)
        true_label = classes[labels[i]]
        pred_label = classes[predictions[i][0]]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis("off")
    plt.show()
    ```
