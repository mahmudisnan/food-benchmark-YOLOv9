import streamlit as st
import subprocess
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import re  # Import the regex module


# Define path to Grad-cam
YOLO_CAM_PATH = r'/home/bdsrc/mahmud/Food Detection/'
if YOLO_CAM_PATH not in sys.path:
    sys.path.append(YOLO_CAM_PATH)
    
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

# Set Streamlit page configuration to centered layout
st.set_page_config(page_title="Food Detection using YOLOv9", layout="centered")

# Define the correct path to your trained weights
MODEL_PATH = r'/home/bdsrc/mahmud/Food Detection/YoloV9NewAnnotation.pt'

# Set the temporary directory within the YOLOv9 folder
TEMP_DIR = r'/home/bdsrc/mahmud/Food Detection/streamlit-app/prediction'  # Set a specific folder name 'prediction' for temporary use

# Ensure the temporary directory exists and is clean
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)  # Remove existing folder if any to start fresh
os.makedirs(TEMP_DIR, exist_ok=True)  # Create the 'prediction' folder

# Streamlit app setup
st.title("Food Detection using YOLOv9")

# File uploader for image input, allowing multiple file uploads
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Save the uploaded images temporarily in the specified temporary directory
    temp_image_paths = []
    for uploaded_file in uploaded_files:
        temp_image_path = os.path.join(TEMP_DIR, uploaded_file.name)
        image = Image.open(uploaded_file)
        image.save(temp_image_path)
        temp_image_paths.append(temp_image_path)

    # Set up two columns for displaying images and results, both covering half of the width
    col1, col2, col3 = st.columns(3)  # Two columns of equal width

    with col1:
        st.write("Uploaded Images")
        for temp_image_path in temp_image_paths:
            image = Image.open(temp_image_path)
            st.image(image, caption=f'Uploaded Image: {os.path.basename(temp_image_path)}', use_container_width=True)

    # Check if the model weights file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights file not found at: {MODEL_PATH}")
    else:
        
        model = YOLO(MODEL_PATH)
        model.eval()
        # Define the YOLO command to run for each uploaded image
        def run_yolo_command(temp_image_paths):
            try:
                # Display predicted images in the second column
                with col2:
                    st.write("Prediction Results")
                with col3:
                    st.write("EIGEN-CAM Heatmaps")


                for i, temp_image_path in enumerate(temp_image_paths):
                    run_name = f"run_{i}"

                    results = model.predict(
                        source=temp_image_path,
                        save=True,
                        project=TEMP_DIR,
                        name=run_name
                    )

                    predict_dir = os.path.join(TEMP_DIR, run_name)
                    predicted_image_paths = glob.glob(os.path.join(predict_dir, "*.jpg"))

                    if predicted_image_paths:
                        for predicted_image_path in predicted_image_paths:
                            with col2:
                                output_image = Image.open(predicted_image_path)
                                st.image(output_image, caption='Prediction Result', use_container_width=True)
                    else:
                        st.error(f"Could not find the predicted output images for {os.path.basename(temp_image_path)}.")

                            
                with col3:
                    rgb_img = cv2.imread(temp_image_path)
                    rgb_resized = cv2.resize(rgb_img, (640, 640))
                    rgb_copy = rgb_resized.copy()
                    rgb_resized = np.float32(rgb_resized) / 255

                    # Target layer for Grad-CAM
                    target_layers = [model.model.model[20]]  # adjust with your architecture

                    cam = EigenCAM(model, target_layers, task='od')
                    grayscale_cam = cam(rgb_copy)[0, :, :]
                    cam_image = show_cam_on_image(rgb_resized, grayscale_cam, use_rgb=True)

                    st.image(cam_image, caption='EIGEN-CAM Heatmap', use_container_width=True)
                    
                

            except Exception as e:
                st.error(f"Failed to execute YOLO command: {e}")

        # Run the YOLO command when the button is clicked
        if st.button("Run YOLO Detection"):
            run_yolo_command(temp_image_paths)

    # Clean up the temporary files after display
    for temp_image_path in temp_image_paths:
        os.remove(temp_image_path)

    # Optionally clean up the entire 'prediction' directory after use
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)  # Remove the entire 'prediction' directory
else:
    st.info("Please upload one or more images to get started.")