import streamlit as st
from PIL import Image
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import numpy as np

# Load the model
model = attempt_load('yolov7.pt', map_location='cpu')  # Adjust model path as necessary

# Set the title of the app
st.title('Image Upload and Object Detection App')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file).convert('RGB')
    img = np.array(image)

    # Preprocess image
    img = letterbox(img, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Inference
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Detect
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.size).round()

            # Print results
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=3)

    # Display the image
    st.image(image, caption='Processed Image.', use_column_width=True)