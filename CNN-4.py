import os
import numpy as np
import tensorflow as tf

BASE_DIR = "/mnt/d/Programming-large-dataset/ML-imaging-pneumonia/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("--- CLUSTER EXECUTION DETECTED: LOADING DATA ---")
train_ds = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
test_ds = tf.keras.utils.image_dataset_from_directory(TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Handle class imbalance with inverse-frequency weighting
labels = np.concatenate([y for x, y in train_ds], axis=0)
total_samples = len(labels)
num_normal = np.sum(labels == 0)
num_pneumonia = np.sum(labels == 1)

weight_for_0 = (1 / num_normal) * (total_samples / 2.0)
weight_for_1 = (1 / num_pneumonia) * (total_samples / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}

# Optimize data loading pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
# Cache and prefetch datasets for improved performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Initialize pre-trained backbone for transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False 

# Construct the CNN architecture with data augmentation and regularization
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Stage 1: Warm-up
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5, class_weight=class_weights)

# Stage 2: Deep Fine-Tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 40
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Fine-tune with a lower learning rate for stability
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Add ModelCheckpoint to automatically save weights directly on the cluster storage
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_pneumonia_model.keras', monitor='val_loss', save_best_only=True)
]

# Start fine-tuning with callbacks for optimization
print("\n--- INITIATING HIERARCHICAL FINE-TUNING ---")
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, class_weight=class_weights)

# Evaluate the final model on the test dataset
print("\n--- FINAL TEST EVALUATION ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nFine-Tuned Cluster Test Accuracy: {test_accuracy * 100:.2f}%")
