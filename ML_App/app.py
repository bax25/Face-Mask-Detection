import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model("face_mask_detector.h5")

# Load pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Function to detect and crop faces
def detect_and_crop_faces(img):
    gray_img = cv2.cvtColor(
        img, cv2.COLOR_RGB2GRAY
    )  # Convert image to grayscale for face detection
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    face_crops = []
    for x, y, w, h in faces:
        face_crop = img[y: y + h, x: x + w]  # Crop the face from the original image
        face_crops.append(face_crop)

    return face_crops


# Function to display demo images
def show_demo_images():
    demo_images = [
        "demo1.jpg",
        "demo2.png",
        "demo3.jpg",
        "demo4.jpg",
        "demo5.jpg",
        "demo6.jpg",
    ]
    # Create two rows for images
    rows = 2  # Number of rows
    cols_per_row = len(demo_images) // rows  # Calculate number of columns per row
    if len(demo_images) % rows != 0:
        cols_per_row += (
            1  # If there is a remainder, increase column count for the last row
        )

    # Loop through images and display them
    for row in range(rows):
        cols = st.columns(cols_per_row)  # Create columns for the current row
        for col in range(cols_per_row):
            index = row * (cols_per_row) + col  # Calculate index for the demo images
            if index < len(
                demo_images
            ):  # Check if the index is within the demo images list
                img_name = demo_images[index]
                img_path = os.path.join(
                    "Demo_Images", img_name
                )  # Assuming images are in a folder named 'Demo Images'

                # Open and resize the demo image
                with open(img_path, "rb") as file:
                    demo_img = Image.open(file)
                    demo_img = demo_img.resize(
                        (250, 250), Image.LANCZOS
                    )  # Resize to fixed size

                    # Set the caption based on the index
                    caption = f'Demo Image {index + 1}: {"With Mask" if index % 2 else "Without Mask"}'
                    cols[col].image(
                        demo_img, caption=caption, use_column_width=False
                    )  # Use column width set to false


# Streamlit app title
st.title("Face Mask Detection App")

# Upload image
uploaded_file = st.file_uploader("Drag and drop an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image into a PIL image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        st.write("**Uploaded Image:**")  # Bold caption
        st.image(image, use_column_width=True)

    with col2:
        # Detect faces
        st.write("Detecting faces...")
        faces = detect_and_crop_faces(image_np)

        if len(faces) == 0:
            st.write("No faces detected.")
        else:
            # Prepare to plot the detected faces
            fig, axes = plt.subplots(1, len(faces), figsize=(15, 5))
            if len(faces) == 1:
                axes = [axes]  # Ensure axes is iterable if there's only one face

            for i, face in enumerate(faces):
                # Resize and normalize the cropped face
                face_resized = cv2.resize(face, (128, 128)) / 255.0
                face_expanded = np.expand_dims(face_resized, axis=0)

                # Make a prediction on the cropped face
                pred = model.predict(face_expanded)

                # Label and confidence calculation
                label = (pred > 0.5).astype(int).flatten()[0]
                confidence = pred[0][0] * 100 if label == 1 else (1 - pred[0][0]) * 100
                label_text = "With Mask" if label == 0 else "Without Mask"

                # Plot each detected face with its prediction
                axes[i].imshow(face)
                axes[i].set_title(f"{label_text} ({confidence:.2f}%)")
                axes[i].axis("off")

            # Display the figure with predictions
            st.pyplot(fig)

# Show demo images for testing
st.write("Try the demo images below or upload your own:")
show_demo_images()
