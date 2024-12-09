import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import cv2

st.set_page_config(
    page_title="Home - Detect"
)

st.title('DeepFake Detection')

st.markdown(
"""
    This webpage allows people to detect any lumbar spine degenerative diseases.
    
    **Just upload an x-ray image of your spine** and it will help you diagnose your back
    condition.

"""
)

# Upload image and show
test_image = st.file_uploader("Upload test image", type=["jpg", "jpeg"])

# Let the user select the model
model_names = {
    'Choose an Option': None,
    'Option 1: Single-stage Inference ': ['cs.pt','ss.pt','nfn.pt'], # Change the model
    'Option 2: Two-Stage Inference using YOLO and ResNet': 'lsdc_resnet50_model1.pth' # Change the model
}

temp_model_name = ['cs.pt','ss.pt','nfn.pt']

model_option = st.selectbox('Choose the method for prediction:', options=list(model_names.keys()))

# Define a function to load the model, this will use Streamlit's newer caching mechanism
@st.cache_resource
def load_model_YOLO(model_filename):
    model = YOLO(f"weights/{model_filename}")
    return model

@st.cache_resource
def load_model_resnet(input_weight, image):
    model = resnet50(pretrained=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 3)

    model.load_state_dict(torch.load(f"weights/{input_weight}"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
        transforms.ToTensor(),
    ])

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        model.to(device)
        output = model(input_tensor)

    # Get the predicted class
    predicted_class = output.argmax(dim=1).item()

    # Define class labels
    class_labels = ["Normal/Mild", "Moderate", "Severe"]
    predicted_label = class_labels[predicted_class]
    return predicted_label




if test_image is not None:
    # Display the uploaded image
    uploaded_image = Image.open(test_image)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    uploaded_image_cv2 = np.array(uploaded_image)
    uploaded_image_cv2_copy = uploaded_image_cv2.copy()


    # Load the selected model (only once per session)
    model_filename = model_names[model_option]
    st.markdown(
    """
        ### Image Result
    """
    )
    if model_option == 'Option 1: Single-stage Inference ':

        temp = []
        
        model1 = load_model_YOLO(model_filename[0]) 
        model2 = load_model_YOLO(model_filename[1])
        model3 = load_model_YOLO(model_filename[2])
 
        results1 = model1(uploaded_image, stream=True)
        results2 = model2(uploaded_image, stream=True)
        results3 = model3(uploaded_image, stream=True)

        for result1 in results1:
            boxes1 = result1.boxes
            if len(boxes1) > 0:
                st.image(result1.plot(font_size=12, pil=True), caption="Spinal Canal Stenosis", use_column_width=True)
        for result2 in results2:
            boxes2 = result2.boxes
            if len(boxes2) > 0:
                st.image(result2.plot(font_size=12, pil=True), caption="Subarticular Stenosis", use_column_width=True)
        for result3 in results3:
            boxes3 = result3.boxes
            if len(boxes3) > 0:
                st.image(result3.plot(font_size=12, pil=True), caption="Neural Foraminal Narrowing", use_column_width=True)


    if model_option == 'Option 2: Two-Stage Inference using YOLO and ResNet':

        stage1_model1 = load_model_YOLO(temp_model_name[0])
        stage1_model2 = load_model_YOLO(temp_model_name[1])
        stage1_model3 = load_model_YOLO(temp_model_name[2])

        stage1_res1 = stage1_model1(uploaded_image, stream=True)
        stage1_res2 = stage1_model2(uploaded_image, stream=True)
        stage1_res3 = stage1_model3(uploaded_image, stream=True)

        for res1 in stage1_res1:
            boxes1 = res1.boxes
            if len(boxes1) > 0 :
                for box in boxes1:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(uploaded_image_cv2_copy, (x_min,y_min), (x_max, y_max), color=(0,255,255), thickness=2)
                    full_image = Image.fromarray(uploaded_image_cv2_copy)
                st.image(full_image, caption=f"Spinal Canal Stenosis Full Image", use_column_width=True)
                for box in boxes1:
                    class_id = int(box.cls[0])
                    label = stage1_model1.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cropped_image = uploaded_image_cv2[y_min:y_max, x_min:x_max]
                    cropped_pil_image = Image.fromarray(cropped_image)
                    resnet_image = load_model_resnet(model_filename, cropped_pil_image)
                    st.image(cropped_pil_image, caption=f"Severity: {label}", use_column_width=True)
        for res2 in stage1_res2:
            boxes2 = res2.boxes
            if len(boxes2) > 0 :
                for box in boxes2:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(uploaded_image_cv2_copy, (x_min,y_min), (x_max, y_max), color=(0,255,255), thickness=2)
                    full_image = Image.fromarray(uploaded_image_cv2_copy)
                st.image(full_image, caption=f"Subarticular Stenosis Full Image", use_column_width=True)
                for box in boxes2:
                    class_id = int(box.cls[0])
                    label = stage1_model2.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cropped_image = uploaded_image_cv2[y_min:y_max, x_min:x_max]
                    cropped_pil_image = Image.fromarray(cropped_image)
                    resnet_image = load_model_resnet(model_filename, cropped_pil_image)
                    st.image(cropped_pil_image, caption=f"Severity: {label}", use_column_width=True)
        for res3 in stage1_res3:
            boxes3 = res3.boxes
            if len(boxes3) > 0 :
                for box in boxes3:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(uploaded_image_cv2_copy, (x_min,y_min), (x_max, y_max), color=(0,255,255), thickness=2)
                    full_image = Image.fromarray(uploaded_image_cv2_copy)
                st.image(full_image, caption=f"Neural Foraminal Narrowing Full Image", use_column_width=True)
                for box in boxes3:
                    class_id = int(box.cls[0])
                    label = stage1_model3.names[class_id]
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    cropped_image = uploaded_image_cv2[y_min:y_max, x_min:x_max]
                    cropped_pil_image = Image.fromarray(cropped_image)
                    resnet_image = load_model_resnet(model_filename, cropped_pil_image)
                    st.image(cropped_pil_image, caption=f"Severity: {label}", use_column_width=True)


    
