import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Model architecture definition
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Data generators for training and validation
def get_data_generators(train_dir, val_dir, target_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(rescale=0.25)
    val_datagen = ImageDataGenerator(rescale=0.25)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, val_generator

# Train the model
def train_model(model, train_generator, val_generator, epochs=10):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    return history

# Save the trained model
def save_model(model, model_path):
    model.save(model_path)

# Load the trained model if exists, otherwise train and save it
def load_or_train_model(train_dir, val_dir, model_path, input_shape, epochs=10):
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = load_trained_model(model_path)
    else:
        print("Training new model...")
        model = build_model(input_shape)
        model = compile_model(model)
        train_generator, val_generator = get_data_generators(train_dir, val_dir)
        train_model(model, train_generator, val_generator, epochs)
        save_model(model, model_path)
    
    return model

# Load the trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Test the model with a single image
def test_model(model, img_path, target_size=(150, 150)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_array = tf.image.convert_image_dtype(img_array, tf.float32)
    
    classes = model.predict(img_array, batch_size=1)
    prediction = "cancerous" if classes[0][0] > 0.5 else "non-cancerous"
    
    plt.imshow(np.array(Image.open(img_path)))
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()
    
    print(f"The image is predicted to be {prediction}.")

# Paths to directories and model
train_dir = r'C:\Users\opsup\Downloads\Leukemia-detection\cancer\train'
val_dir = r'C:\Users\opsup\Downloads\Leukemia-detection\cancer\val'
model_path = 'mymodel.h5'
test_img_path = r"C:\Users\opsup\Downloads\Leukemia-detection\cancer\test\Cancer\_0_994.jpeg"

# Define, compile and train or load the model
input_shape = (150, 150, 3)
model = load_or_train_model(train_dir, val_dir, model_path, input_shape, epochs=10)

# Test the model
test_model(model, test_img_path)
