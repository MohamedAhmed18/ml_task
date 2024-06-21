import streamlit as st
from PIL import Image
from detect import detect
import numpy as np
# Set the title of the app
st.title('Image Upload and Object Detection App')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
path = 'Capture.png'

# Initialize or use the existing model from the session state
if 'model' not in st.session_state:
    st.session_state.model = None

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Convert image to numpy array to manipulate the channels
    img_array = np.array(image)
    # Swap the red and green channels
    red_channel = img_array[:, :, 0].copy()
    blue_channel = img_array[:, :, 2].copy()
    img_array[:, :, 0] = blue_channel
    img_array[:, :, 2] = red_channel
    # Convert array back to image
    image = Image.fromarray(img_array)
    image.save(path)
    # Use the model from session state
    image, out, st.session_state.model = detect(path, st.session_state.model)
    st.image(image, caption='Processed Image.', use_column_width=True)
    st.write(out)