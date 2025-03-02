import os
import google.generativeai as genai
import markdown
from flask import Blueprint, request, render_template, jsonify
from PIL import Image
import base64
import io

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

ai_nutritionist_bp = Blueprint('ai_nutritionist', __name__, template_folder='templates')

def get_gemini_response(input_text, image, prompt=None):
    print(f"Input Text length: {len(input_text)}")
    print(f"Image provided: {image is not None}")
    
    if not input_text or not image:
        raise ValueError("Input text and image are required. Please provide valid inputs.")

    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Create content array with required elements
    content = [input_text, image[0]]
    
    # Add prompt only if it's not empty
    if prompt:
        content.append(prompt)
    
    response = model.generate_content(content)
    return response.text

def input_image_setup(uploaded_file=None, img_file_buffer=None):
    try:
        if uploaded_file is not None:
            # Compress the image from file upload
            image = Image.open(uploaded_file)
            print(f"Original image size: {image.size}")
            
            # Convert to RGB (remove alpha channel if present)
            image = image.convert("RGB")
            
            # Resize if too large
            max_size = (1024, 1024)
            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)
                print(f"Resized to: {image.size}")
            
            # Compress the image
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50)  # Lower quality for smaller size
            bytes_data = buffer.getvalue()
            print(f"Compressed size: {len(bytes_data)} bytes")
            
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": bytes_data
                }
            ]
            return image_parts
            
        elif img_file_buffer is not None:
            # Handle base64 image from camera
            # Remove the header if present (e.g., "data:image/jpeg;base64,")
            if "," in img_file_buffer:
                img_file_buffer = img_file_buffer.split(",")[1]
            
            # Decode base64 image
            bytes_data = base64.b64decode(img_file_buffer)
            image = Image.open(io.BytesIO(bytes_data))
            print(f"Original camera image size: {image.size}")
            
            # Convert to RGB
            image = image.convert("RGB")
            
            # Resize if too large
            max_size = (1024, 1024)
            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)
                print(f"Resized to: {image.size}")
            
            # Compress the image
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=50)
            bytes_data = buffer.getvalue()
            print(f"Compressed size: {len(bytes_data)} bytes")
            
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": bytes_data
                }
            ]
            return image_parts
        else:
            raise FileNotFoundError("No image provided")
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise

@ai_nutritionist_bp.route('/al-nutritionist', methods=['GET', 'POST'])
def nutritionist():
    if request.method == 'POST':
        try:
            uploaded_file = request.files.get('meal_image')
            camera_image = request.form.get('camera_image')
            
            # Check if either file upload or camera image is provided
            if (not uploaded_file or uploaded_file.filename == '') and not camera_image:
                return render_template("ai_nutritionist.html", error="No image provided. Please upload an image or take a picture.")
            
            age = request.form.get('age')
            gender = request.form.get('gender')
            physical_intensity = request.form.get('physical_intensity')
            height_feet = request.form.get('height_feet')
            height_inch = request.form.get('height_inch')
            weight = request.form.get('weight')
            
            # Check required form fields
            if not all([age, gender, physical_intensity, height_feet, height_inch, weight]):
                return render_template("ai_nutritionist.html", error="All fields are required. Please complete the form.")
            
            # Check API key
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                return render_template("ai_nutritionist.html", error="Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
            
            input_prompt = f"""
            don't say anything as introduction for the response, jump right into the answer
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
            
            # Process either uploaded file or camera image
            if uploaded_file and uploaded_file.filename != '':
                print(f"Processing uploaded image: {uploaded_file.filename}")
                image_data = input_image_setup(uploaded_file=uploaded_file)
            else:
                print("Processing camera image")
                image_data = input_image_setup(img_file_buffer=camera_image)
            
            print("Calling Gemini API...")
            analysis = get_gemini_response(input_prompt, image_data)
            
            print("Converting markdown to HTML...")
            analysis_html = markdown.markdown(analysis, extensions=['extra', 'tables'])
            
            return render_template("ai_nutritionist.html", nutritional_analysis=analysis_html)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return render_template("ai_nutritionist.html", error=f"An error occurred: {str(e)}")
    
    return render_template("ai_nutritionist.html")