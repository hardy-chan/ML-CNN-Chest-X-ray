import os
import tensorflow as tf

# 1. Define dataset directory paths
BASE_DIR = "D:/Programming-large-dataset/ML-imaging-pneumonia/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- LOADING DATASETS ---")

# 2. Load directories explicitly using pre-split folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

# Optimize data loading pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Define Convolutional Neural Network (CNN) architecture
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification layer
])

# 4. Compile model with standard evaluation metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train model on training partition and validate on validation partition
print("\n--- STARTING TRAINING ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 6. Run final evaluation on unseen testing partition
print("\n--- FINAL TEST EVALUATION ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
