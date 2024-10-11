import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for: {gpus}")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
else:
    print("No GPU detected.")

data_path = "Face Mask Dataset"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (128, 128)) / 255.0  # Normalize the image
    return img_resized

def load_images_labels(data_path, folder):
    images = []
    labels = []
    folder_path = os.path.join(data_path, folder)

    for label in ['WithMask', 'WithoutMask']:
        label_folder_path = os.path.join(folder_path, label)
        for img_file in os.listdir(label_folder_path):
            img_path = os.path.join(label_folder_path, img_file)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(0 if label == 'WithMask' else 1)  # WithMask=0, WithoutMask=1
    
    return np.array(images), np.array(labels)

def predict_multiple_images(model, images):
    predictions = model.predict(images)

    results = []
    for pred in predictions:
        label = np.argmax(pred)
        label_text = 'With Mask' if label == 0 else 'Without Mask'
        confidence = np.max(pred) * 100
        results.append((label_text, confidence))
    
    return results

# Load the training, validation, and test datasets
train_images, train_labels = load_images_labels(data_path, 'Train')
val_images, val_labels = load_images_labels(data_path, 'Validation')
test_images, test_labels = load_images_labels(data_path, 'Test')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    batch_size=32,
                    epochs=10,
                    callbacks=[early_stopping])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Generate predictions on the test dataset
pred_labels = np.argmax(model.predict(test_images), axis=1)

# Display confusion matrix and classification report
print(confusion_matrix(test_labels, pred_labels))
print(classification_report(test_labels, pred_labels))

# Predict multiple images from a specified folder
def predict_images_from_folder(model, folder_path):
    images = []
    image_paths = []

    # Load and preprocess each image in the folder
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        image_paths.append(img_path)
        img = preprocess_image(img_path)
        images.append(img)
    
    images = np.array(images)
    
    # Make predictions
    predictions = predict_multiple_images(model, images)

    # Display results
    for i, (label_text, confidence) in enumerate(predictions):
        original_img = cv2.imread(image_paths[i])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(original_img)
        plt.title(f"{label_text} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

# Path to the folder containing images for prediction
test_images_folder = r'C:\Users\Bala Vignesh\ML Coding\Face Mask Detection\Face Mask Dataset\Multiple Test Images'
predict_images_from_folder(model, test_images_folder)