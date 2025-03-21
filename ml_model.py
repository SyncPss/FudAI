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

uploaded_files = st.file_uploader("Upload food images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)

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
        st.write("---")

# Image URL Input
st.subheader("-OR-")
st.subheader("Provide image URLs (separated by commas):")
image_urls = st.text_input("Enter image URLs:")

if image_urls:
    urls = [url.strip() for url in image_urls.split(",")]
    for image_url in urls:
        try:
            response = requests.get(image_url)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption=f"Image from URL: {image_url}", use_container_width=True)

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
            st.write("---")

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching image URL: {image_url} - {e}")
        except Exception as e:
            st.error(f"Error processing image URL: {image_url} - {e}")