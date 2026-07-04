import os
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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 1. Establish pre-trained backbone (Transfer Learning)
# Exclude the default top classification layer to use custom binary logic
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze weights to preserve pre-trained features

# 2. Construct the production network pipeline
model = tf.keras.Sequential([
    # Data Augmentation Section
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    
    # Pre-trained feature extractor expects inputs scaled between [-1, 1]
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    
    base_model,
    
    # Global Pooling instead of Flatten to minimize parameter explosion
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Regularization dense block
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 3. Add automated optimization callbacks
callbacks = [
    # Automatically drops learning rate by a factor of 0.2 if val_loss stalls for 2 epochs
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6),
    # Stops training early if validation accuracy fails to improve for 4 straight epochs
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
]

print("\n--- STARTING OPTIMIZED TRAINING ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20, # Higher epoch allowance due to EarlyStopping callback control
    callbacks=callbacks
)

print("\n--- FINAL TEST EVALUATION ---")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nOptimized Test Accuracy: {test_accuracy * 100:.2f}%")
