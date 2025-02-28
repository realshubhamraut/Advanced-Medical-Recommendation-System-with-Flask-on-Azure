from flask import Blueprint, request, render_template, jsonify, Response, session, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import ast
import os
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from model_downloader import ensure_local_model

symptom_analyzer_bp = Blueprint('symptom_analyzer_bp', __name__, template_folder='templates')

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")

# Function to load model from local or cloud storage
def load_model(model_name):
    # First try the local path (for development environment)
    local_path = os.path.join('models', model_name)
    if os.path.exists(local_path):
        print(f"Loading model from local path: {local_path}")
        return pickle.load(open(local_path, 'rb'))
    
    # If not found locally, try to download from Azure blob storage (for production/Azure)
    print(f"Local model not found at {local_path}. Attempting to download from Azure...")
    cloud_path = ensure_local_model(model_name)
    print(f"Loading model from downloaded path: {cloud_path}")
    return pickle.load(open(cloud_path, 'rb'))

# Load the model with fallback mechanism
try:
    svc = load_model('svc.pkl')
    print("SVC model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # You could add additional fallback logic here if needed
    raise

def helper(dis):
    # Get description as plain text.
    desc_row = description[description['Disease'] == dis]['Description']
    desc = desc_row.values[0] if not desc_row.empty else "No description available"
    
    # Get precautions as a list.
    prec_row = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    prec_list = prec_row.iloc[0].tolist() if not prec_row.empty else []
    
    # Get medications.
    med_row = medications[medications['Disease'] == dis]['Medication']
    if not med_row.empty:
        med_value = med_row.values[0]
        try:
            med_list = ast.literal_eval(med_value)
        except Exception:
            med_list = [med_value]
    else:
        med_list = []
         
    # Get diet recommendations.
    die_row = diets[diets['Disease'] == dis]['Diet']
    if not die_row.empty:
        die_value = die_row.values[0]
        try:
            die_list = ast.literal_eval(die_value)
        except Exception:
            die_list = [die_value]
    else:
        die_list = []
         
    # Get workout information.
    wrk_row = workout[workout['disease'] == dis]['workout']
    if not wrk_row.empty:
        wrk_value = wrk_row.values[0]
        try:
            wrk_list = ast.literal_eval(wrk_value)
        except Exception:
            wrk_list = [wrk_value]
    else:
        wrk_list = []
    
    return desc, prec_list, med_list, die_list, wrk_list

symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
    'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
    30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid',
    40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
    0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
    35: 'Psoriasis', 27: 'Impetigo'
}

@symptom_analyzer_bp.route('/symptom-suggestions', methods=['GET'])
def symptom_suggestions():
    term = request.args.get('term', '').lower()
    suggestions = [sym for sym in symptoms_dict.keys() if term in sym.lower()]
    return jsonify(suggestions)

@symptom_analyzer_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms or symptoms.strip() == "Symptoms":
            message = "Please enter valid symptoms."
            return render_template('symptom_analyzer.html', message=message)
        user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip()]
        # Create a full feature vector for all possible 132 symptoms
        feature_vector = [0] * len(symptoms_dict)
        for s in user_symptoms:
            if s in symptoms_dict:
                feature_vector[symptoms_dict[s]] = 1
        predicted_index = svc.predict([np.array(feature_vector)])[0]
        predicted_disease = diseases_list[predicted_index]
        dis_des, prec, med, rec_diet, wrkout = helper(predicted_disease)
        session['report'] = {
            "Predicted Disease": predicted_disease,
            "Description": dis_des,
            "Precautions": ", ".join(map(str, prec)),
            "Medications": ", ".join(map(str, med)),
            "Diet Recommendation": ", ".join(map(str, rec_diet)),
            "Workout": ", ".join(map(str, wrkout))
        }
        return render_template('symptom_analyzer.html', predicted_disease=predicted_disease,
                               dis_des=dis_des, my_precautions=prec,
                               medications=med, my_diet=rec_diet, workout=wrkout)
    return render_template('symptom_analyzer.html')

@symptom_analyzer_bp.route('/download', methods=['POST'])
def download():
    name = request.form.get('name')
    mobile = request.form.get('mobile')
    age = request.form.get('age')
    gender = request.form.get('gender')
    if not (name and mobile and age and gender):
        return redirect(url_for('index'))
    report = session.get('report', None)
    if not report:
        return redirect(url_for('index'))
    report.update({
        "Name": name,
        "Mobile": mobile,
        "Age": age,
        "Gender": gender
    })
    session['report'] = report
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    pdf.setFont("Helvetica", 12)
    for key in ["Name", "Mobile", "Age", "Gender"]:
        line = f"{key}: {report.get(key, '')}"
        pdf.drawString(50, y, line)
        y -= 20
    y -= 10
    for key in ["Predicted Disease", "Description", "Precautions", "Medications", "Diet Recommendation", "Workout"]:
        line = f"{key}: {report.get(key, '')}"
        pdf.drawString(50, y, line)
        y -= 20
        if y < 50:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 12)
    pdf.save()
    buffer.seek(0)
    return Response(buffer, mimetype='application/pdf',
                    headers={"Content-Disposition": "attachment;filename=report.pdf"})

@symptom_analyzer_bp.route('/symptom-analyzer')
def show_symptom_analyzer():
    return render_template("symptom_analyzer.html")

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.secret_key = "your-secret-key"
    app.register_blueprint(symptom_analyzer_bp)
    app.run(debug=True)