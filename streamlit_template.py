import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile  # To handle temporary file creation
from PIL import Image
# Store feedback in a list
feedback_list = []

# Modified feedback page
def feedback_page():
    st.header("We Value Your Feedback")
    
    # Get user feedback
    feedback = st.text_area("Please share your feedback here...", key="feedback_area")  # Added key for unique ID
    
    # Add feedback to the list and display it when the button is pressed
    if st.button("Submit Feedback"):
        if feedback:
            feedback_list.append(feedback)  # Store feedback in the list
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting.")
    
    # Display all feedback
    if feedback_list:
        st.subheader("Previous Feedback:")
        for feedback in feedback_list:
            st.write(f"- {feedback}")


# Load Environment Variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define Models and Categories
skin_cancer_categories = [
    "actinic keratosis", "basal cell carcinoma", "dermatofibroma", 
    "melanoma", "nevus", "pigmented benign keratosis", 
    "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"
]

brain_tumor_categories = ["glioma", "meningioma", "notumor", "pituitary"]
pneumonia_categories = ["Normal", "Pneumonia"]

# Placeholder models 
from tensorflow.keras.models import load_model

# Load actual models
skin_cancer_model = load_model("skin_cancer_model.h5")
brain_tumor_model = load_model("brain_tumor_model.h5")
pneumonia_model = load_model("pneumonia_model.h5")


# Helper function for predictions
def predict_image(model, image, categories):
    # Preprocess the image to match the input shape of the model
    image = image.resize((224, 224))  # Resize based on model requirements
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions using the model
    predictions = model.predict(image_array)[0]
    predicted_class = categories[np.argmax(predictions)]
    prediction_dict = dict(zip(categories, predictions))
    return predicted_class, prediction_dict

# Medical Chatbot Page

input_prompt = """
You are an experienced and highly skilled medical doctor specializing in diagnostics. You have been provided with patient information, including images and/or symptoms, to analyze for potential health issues. Your goal is to provide an accurate, detailed, and professional diagnosis based on the input data.

Follow these steps meticulously:

1. Carefully examine the provided image and/or read through the symptoms. Identify any visible signs, patterns, or notable indicators that may suggest common or uncommon medical conditions.
2. Cross-reference the identified symptoms or observations with typical presentations of various diseases. Provide a clear, organized list of possible diagnoses, starting with the most likely.
3. For each potential diagnosis:
   - Explain why it may be relevant based on the symptoms or visible indicators.
   - Include key information on what symptoms or signs typically accompany this condition.
4. Recommend actionable next steps for the patient. This might include further tests, specialist referrals, lifestyle changes, or over-the-counter remedies if appropriate.
5. Include a disclaimer: "This AI-based analysis does not replace a professional medical consultation. Always consult with a qualified healthcare provider before making health decisions."

If certain aspects cannot be determined from the image or symptoms alone, state: "Some aspects are inconclusive based on the provided information."

Use clear, medically accurate language, and provide your analysis in a structured, easy-to-read format. Be thorough, objective, and focus on patient safety.
"""


def prepare_image_for_analysis(uploaded_file):
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": image_data}]
        return image_parts
    else:
        return None


def get_gemini_response(input_prompt, symptoms, image_data=None):
    # Include symptoms in the prompt for analysis
    prompt_with_symptoms = f"{input_prompt}\n\nPatient Symptoms: {symptoms}"
    
    # Generate response based on the provided image and/or symptoms
    if image_data:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt_with_symptoms, image_data[0]])
    else:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt_with_symptoms])
    
    return response.text

# Medical Chatbot Page Function
def chatbot_page():
    st.header("Medical Chatbot")
    
    # Symptom input and image upload
    symptoms = st.text_area("Enter your symptoms:", "")
    uploaded_image = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])

    if st.button("Analyze Symptoms"):
        if symptoms:
            with st.spinner("Analyzing symptoms with AI..."):
                # Prepare image if uploaded
                image_data = prepare_image_for_analysis(uploaded_image) if uploaded_image else None
                result = get_gemini_response(input_prompt, symptoms, image_data)
                st.subheader("Diagnosis Results:")
                st.write(result)
        else:
            st.warning("Please enter some symptoms to analyze.")

# Skin Cancer Prediction
def skin_cancer_page():
    st.header("Skin Cancer Prediction")
    uploaded_file = st.file_uploader("Upload a Skin Lesion Image", type=["jpg", "png", "jpeg"])
    if st.button("Analyze Skin Cancer"):
        if uploaded_file:
            with st.spinner("Running Skin Cancer Model..."):
                image = Image.open(uploaded_file)
                predicted_class, predictions = predict_image(skin_cancer_model, image, skin_cancer_categories)
                st.write(f"**Predicted Class**: {predicted_class}")
                st.write("**Prediction Probabilities:**")
                fig, ax = plt.subplots(figsize=(10, 6))  # Set a reasonable figure size
                sns.barplot(x=list(predictions.keys()), y=list(predictions.values()), ax=ax)
                ax.set_title("Skin Cancer Prediction Probabilities", fontsize=16)
                ax.set_ylabel("Probability", fontsize=12)
                ax.set_xlabel("Skin Cancer Type", fontsize=12)

                #Rotate x-axis labels for better readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                #Display the plot in Streamlit
                st.pyplot(fig)


        else:
            st.warning("Please upload an image for analysis.")

# Brain Tumor Prediction
def brain_tumor_page():
    st.header("Brain Tumor Prediction")
    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])
    if st.button("Analyze Brain Tumor"):
        if uploaded_file:
            with st.spinner("Running Brain Tumor Model..."):
                image = Image.open(uploaded_file)
                predicted_class, predictions = predict_image(brain_tumor_model, image, brain_tumor_categories)
                st.write(f"**Predicted Class**: {predicted_class}")
                st.write("**Prediction Probabilities:**")
                fig, ax = plt.subplots()
                sns.barplot(x=list(predictions.keys()), y=list(predictions.values()), ax=ax)
                ax.set_title("Brain Tumor Prediction Probabilities")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
        else:
            st.warning("Please upload an image for analysis.")

# Pneumonia Prediction
def pneumonia_page():
    st.header("Pneumonia Prediction")
    uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "png", "jpeg"])
    if st.button("Analyze Pneumonia"):
        if uploaded_file:
            with st.spinner("Running Pneumonia Model..."):
                image = Image.open(uploaded_file)
                predicted_class, predictions = predict_image(pneumonia_model, image, pneumonia_categories)
                st.write(f"**Predicted Class**: {predicted_class}")
                st.write("**Prediction Probabilities:**")
                fig, ax = plt.subplots()
                sns.barplot(x=list(predictions.keys()), y=list(predictions.values()), ax=ax)
                ax.set_title("Pneumonia Prediction Probabilities")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
        else:
            st.warning("Please upload an image for analysis.")

# FAQ Page
def faq_page():
    st.header("Frequently Asked Questions")
    st.markdown("""
    **Q1:** What is this app about?  
    **A1:** This medical diagnostic app uses advanced AI models to predict skin cancer, brain tumors, and pneumonia based on uploaded medical images. It also features a chatbot to assist with symptom analysis and diagnosis.

    **Q2:** How accurate are the predictions?  
    **A2:** The models used in this app are trained on large datasets, but please note that these predictions should not be considered a substitute for professional medical advice. Always consult a doctor for a definitive diagnosis.

    **Q3:** Can I trust the chatbot diagnosis?  
    **A3:** The chatbot provides a general analysis based on the symptoms you describe but is not a replacement for a medical consultation. It offers suggestions for next steps and potential tests.

    **Q4:** How do I use the image upload feature?  
    **A4:** Simply click the "Upload" button and select an image (e.g., skin lesion, MRI, or X-ray) from your device to analyze. The app will process the image and provide predictions.

    **Q5:** Is my data safe?  
    **A5:** We take privacy seriously. Your data is not stored permanently. Uploaded images are only used for analysis during the session.
    """)

# About Page
def about_page():
    st.title("MedDiagnose - AI Medical Diagnostic Assistant")

    # About Section
    st.header("About the App")

    st.write("""
    Welcome to **MedDiagnose**, an advanced AI-driven diagnostic tool that helps in the early detection of health conditions by analyzing medical images and symptoms.  
    The app provides accurate predictions for various diseases and offers a user-friendly interface with interactive features.

    **Key Features**:
    - **Image Analysis**: Detects conditions like pneumonia, brain tumors, and skin cancer from medical images.
    - **Google Gemini Chatbot**: Offers personalized medical advice and diagnostics based on your symptoms and images.
    - **Health Information**: Learn about various diseases, their warning signs, and preventive measures.

    **How It Works**:
    1. **Upload Your Medical Image** (MRI, X-ray, Skin Image).
    2. **Input Symptoms** for analysis by our chatbot.
    3. **Get Diagnoses** based on your input.

    **Disclaimer**: This tool provides insights based on AI predictions but does not replace professional medical consultation. Always consult a healthcare provider for a definitive diagnosis.
    """)



# Footer with Copyright
def footer():
    st.markdown("""
    <hr>
    <footer style="text-align: center; padding: 10px;">
        <p>&copy; 2024 Medical Diagnostic App. All rights reserved.</p>
    </footer>
    """, unsafe_allow_html=True)

# Main Application
def main():
    st.set_page_config(page_title="Medical Diagnostic App", page_icon="🩺", layout="wide")
    
    # Sidebar Navigation
    
    st.markdown("""
    <style>
        .css-1d391kg {  # Sidebar title's CSS class
            font-size: 30px !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

    # Now you can set the title with the new size
    st.sidebar.title("MedDiagnose")  
    page = st.sidebar.radio("Navigate", ["About", "Skin Cancer Prediction", "Brain Tumor Prediction", "Pneumonia Prediction", "Medical Chatbot", "FAQ", "Feedback"])

    if page == "About":
        about_page()
    elif page == "Skin Cancer Prediction":
        skin_cancer_page()
    elif page == "Brain Tumor Prediction":
        brain_tumor_page()
    elif page == "Pneumonia Prediction":
        pneumonia_page()
    elif page == "Medical Chatbot":
        chatbot_page()  # Add the chatbot page here
    elif page == "FAQ":
        faq_page()
    elif page == "Feedback":
        feedback_page()
        
    footer()

if __name__ == "__main__":
    main()
