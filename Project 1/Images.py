import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
import warnings
import logging
from PIL import ImageFile

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ====== Step 1: Clean corrupt images ======
src_dir = '/Users/shishiradhikari/Documents/Images'
clean_dir = '/Users/shishiradhikari/Documents/Images_clean'

if not os.path.exists(clean_dir):
    os.makedirs(clean_dir)

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)
    if os.path.isdir(class_path):
        clean_class_path = os.path.join(clean_dir, class_name)
        os.makedirs(clean_class_path, exist_ok=True)
        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg')):
                src_path = os.path.join(class_path, file)
                dst_path = os.path.join(clean_class_path, file)
                try:
                    img_bytes = tf.io.read_file(src_path)
                    img = tf.image.decode_image(img_bytes)
                    img.numpy()  # Force decode
                    shutil.copy2(src_path, dst_path)
                except:
                    pass

# ====== Step 2: Load and Normalize Dataset ======
print(" Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    clean_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    clean_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

class_names = train_ds.class_names

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ====== Step 3: Build the CNN Model ======
print(" Building model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ====== Step 4: Compile and Train ======
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(" Training the model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# ====== Step 5: Show Predictions ======
print("\n Showing predictions on sample validation images...")

for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    predictions = (predictions > 0.5).astype("int32")

    plt.figure(figsize=(12, 12))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((images[i].numpy() * 255).astype("uint8"))
        true_label = class_names[labels[i]]
        predicted_label = class_names[predictions[i][0]]
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
