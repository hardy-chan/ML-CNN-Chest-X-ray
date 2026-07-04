import os
import numpy as np
import tensorflow as tf

BASE_DIR = "D:/Programming-large-dataset/ML-imaging-pneumonia/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- LOADING DATASETS ---")
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Calculate class distribution to handle dataset imbalance
labels = np.concatenate([y for x, y in train_ds], axis=0)
total_samples = len(labels)
num_normal = np.sum(labels == 0)
num_pneumonia = np.sum(labels == 1)

# Apply standard inverse-frequency weighting formula
weight_for_0 = (1 / num_normal) * (total_samples / 2.0)
weight_for_1 = (1 / num_pneumonia) * (total_samples / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}
print(f"Calculated Class Weights - Normal (0): {weight_for_0:.2f}, Pneumonia (1): {weight_for_1:.2f}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 1. Initialize base backbone
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False 

# 2. Construct network layers
model = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Stage 1: Initial warm-up training
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- STAGE 1: WARM-UP TRAINING ---")
model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weights)

# 3. Stage 2: Unfreeze deep layers for Fine-Tuning
base_model.trainable = True

# Refreeze all layers except the last 40 layers
fine_tune_at = len(base_model.layers) - 40
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a significantly lower learning rate to prevent gradient explosion
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
]

print("\n--- STAGE 2: DEEP FINE-TUNING ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights
)

print("\n--- FINAL TEST EVALUATION ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nFine-Tuned Test Accuracy: {test_accuracy * 100:.2f}%")
