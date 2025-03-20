import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForImageClassification
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import torch

# Model Loading
@st.cache_resource 
def load_model(model_name, token=None):
    try:
        if token:
            processor = AutoProcessor.from_pretrained(model_name, token=token)
            model = AutoModelForImageClassification.from_pretrained(model_name, token=token, from_tf=True)
        else:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name, from_tf=True)
        return processor, model
    except OSError as e:
        st.error(f"Error loading model: {e}")
        return None, None

model_name = "SynchoPass/food_image_classification" 
token = "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"
processor, model = load_model(model_name, token)

# Calorie Database Loading
@st.cache_data
def load_calorie_data(filepath):
    df = pd.read_csv(filepath)
    return df

calorie_df = load_calorie_data("nutrition.csv")

# Calorie Estimation Logic
def get_nutrition_info(food_label, calorie_df):
    food_row = calorie_df[calorie_df["label"] == food_label].iloc[0]
    if not food_row.empty:
        calorie = food_row["calories"]
        weight = food_row["weight"]
        return calorie, weight
    else:
        return None, None

# StreamLit App
st.title("FudAI")
st.title("Food Image Classification for Automated Dietary Monitoring")
st.subheader("Just Snap a Pic & We'll Take Care of Your Diet!")

uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_container_width=True)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class = model.config.id2label[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item() * 100

    st.write(f"**Predicted Food:** {predicted_class}")
    st.write(f"**Accuracy:** {confidence:.2f}%")

    calorie, weight = get_nutrition_info(predicted_class, calorie_df)
    if calorie is not None and weight is not None:
        st.write(f"**Estimated Calorie:** {calorie} kcal per {weight} g")
    else:
        st.write("Calorie and weight estimation not available for this food.")

# Image URL Input
st.subheader("Or")
st.subheader("Provide an image URL:")
image_url = st.text_input("Enter image URL:")

if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Image from URL.", use_container_width=True)

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        confidence = probabilities[predicted_class_idx].item() * 100

        st.write(f"**Predicted Food:** {predicted_class}")
        st.write(f"**Accuracy:** {confidence:.2f}%")

        calorie, weight = get_nutrition_info(predicted_class, calorie_df)
        if calorie is not None and weight is not None:
            st.write(f"**Estimated Calorie:** {calorie} kcal / {weight} g")
        else:
            st.write("Calorie and weight estimation not available for this food.")

    except Exception as e:
        st.error(f"Error processing image URL: {e}")