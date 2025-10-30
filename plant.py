
#========= Importing Libraries =========

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from io import BytesIO
import base64
import requests
import tempfile

# ========= Setting Up the Page ==========

st.set_page_config(page_title="🌿 Plant Disease Recognition", layout="centered")

st.markdown("<h1 class='title'>🌿 Plant Disease Recognition System</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
.title {
    text-align: center;
    color: #2E8B57;
    font-family: 'Arial', sans-serif;
}
body {
    background-color: #F0FFF0;
    font-family: 'Arial', sans-serif;
    color: #333333;
}
.stButton>button {
    background-color: #2E8B57;  
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
}
.stButton>button:hover {    
    background-color: #3CB371;
}   
.stFileUploader>div {
    border: 2px dashed #2E8B57;
    border-radius: 8px;
    padding: 20px;
}   
.stFileUploader>div:hover {    
    border-color: #3CB371;
}
</style>    
""", unsafe_allow_html=True)
st.markdown("Upload an image of a plant leaf to identify potential diseases and get treatment suggestions.")

# ======== Background Image (Real City Houses - Dimmed) ========

import streamlit as st
import base64

def set_background(png_file, brightness=0.8):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: brightness({brightness});
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background("WhatsApp_Image_2025-06-25_at_09.41.01_f4759548.webp" , brightness=0.8)

# ========= Loading the Model and Class Names ==========

@st.cache_resource
def load_model():
    
    file_id = "1sWWrtDQEJS2-kLXUVN5H_gZNI5UB3qQG"
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

  
    response = requests.get(download_url)
    response.raise_for_status()  

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    
    model = tf.keras.models.load_model(tmp_path)
    return model


model = load_model()

with open("plant_disease.json", 'r', encoding='utf-8') as file:
    plant_disease = json.load(file)

# ========= Feature Extraction and Prediction Functions ==========

def extract_features(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    feature = tf.keras.utils.img_to_array(image)
    return np.expand_dims(feature, axis=0)

def model_predict(image: Image.Image):
    img = extract_features(image)
    prediction = model.predict(img)
    label = plant_disease[int(np.argmax(prediction))]
    return label

# ========= Streamlit App Interface ================================

uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([0.23,2,1])  
    with col2:
        st.image(image, caption="📷 Uploaded Image", width=600)

    col1, col2, col3 = st.columns([0.1,2,0.1])  
    with col2:
        if st.button("🔍 Recognize Disease..."):
            with st.spinner("⏳ Predicting..."):
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                label = model_predict(image)

                # ======== Card Layout ========
                st.markdown(f"""
                    <style>
                    .card {{
                        display: flex;
                        align-items: center;
                        background-color: #87CEEB;
                        border-radius: 12px;
                        padding: 5px;  /* أقل padding */
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);  /* ظل أخف */
                        margin-bottom: 10px;  
                        max-width: 800px; 
                    }}
                    .card img {{
                        border-radius: 10px;
                        max-width: 120px;  
                        height: auto;
                        margin-right: 15px;
                    }}
                    .card .info {{
                        text-align: left;
                    }}
                    .card .info h2 {{
                        margin: 0;
                        font-size: 16px;  
                        color: #2E8B57;
                    }}
                    .card .info p {{
                        margin: 3px 0;
                        font-size: 12px;  
                        color: #333333;
                    }}
                    </style>

                    <div class="card">
                        <img src="data:image/png;base64,{img_str}" />
                        <div class="info">
                            <h2>{label['name']}</h2>
                            <p>{label['cause']}</p>
                            <p><b>Treatment Suggestions:</b></p>
                            <p><b>
                                {label['cure']}
                            </b></p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)



# Footer
st.markdown("""
     <hr style="border:1px solid #ffffff30; margin-top:40px;">
     <p style="text-align:center; color:#f0f0f0;">Developed by <b>Mazin Soliman</b> 🌱</p>
    """, unsafe_allow_html=True)




