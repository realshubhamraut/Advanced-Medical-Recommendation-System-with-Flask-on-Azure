from dotenv import load_dotenv
import os
import google.generativeai as genai
import markdown
from flask import Blueprint, request, render_template
from PIL import Image

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

ai_nutritionist_bp = Blueprint('ai_nutritionist', __name__, template_folder='templates')

def get_gemini_response(input_text, image, prompt):
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content([input_text, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file, img_file_buffer):
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image_parts = [
            {
                "mime_type": uploaded_file.content_type,
                "data": bytes_data
            }
        ]
        return image_parts
    elif img_file_buffer is not None:
        bytes_data = img_file_buffer.read()
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

@ai_nutritionist_bp.route('/al-nutritionist', methods=['GET', 'POST'])
def nutritionist():
    if request.method == 'POST':
        uploaded_file = request.files.get('meal_image')
        age = request.form.get('age')
        gender = request.form.get('gender')
        physical_intensity = request.form.get('physical_intensity')
        height_feet = request.form.get('height_feet')
        height_inch = request.form.get('height_inch')
        weight = request.form.get('weight')
        
        if uploaded_file:
            input_prompt = f"""
dont say anything as introduction for the response, jump right into the answer
You are an expert in nutrition and dietetics with exceptional skills in analyzing food images.
Analyze the provided meal image and create a detailed nutritional report strictly in a table format with the following columns:
1. Food Item (e.g., Rice, Curry, Paneer, Dal, Curd, etc.)
2. Protein (g)
3. Fat (g)
4. Carbs (g)
5. Fiber (g)
6. Cholesterol (mg)
7. Sugar (g)
8. Vitamins (%DV)
9. Minerals (%DV)
10. Calories

After the table, add a summary section as follows:
TOTAL CALORIES:
Your total caloric intake from this meal is approximately XXX calories.

RECOMMENDATION:
State whether the meal is healthy or not along with a brief explanation and specific suggestions.

Additional personal details provided:
Age: {age}
Gender: {gender}
Physical Intensity: {physical_intensity}
Height: {height_feet} feet {height_inch} inches
Weight: {weight} kg
            """
            image_data = input_image_setup(uploaded_file, None)
            analysis = get_gemini_response(input_prompt, image_data, "")
            # Convert Markdown to HTML using the 'extra' and 'tables' extensions.
            analysis_html = markdown.markdown(analysis, extensions=['extra', 'tables'])
            return render_template("ai_nutritionist.html", nutritional_analysis=analysis_html)
    return render_template("ai_nutritionist.html")

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "your-secret-key"
    app.register_blueprint(ai_nutritionist_bp)
    app.run(debug=True)