import logging
from flask import Flask, render_template

from symptom_analyzer import symptom_analyzer_bp
from analyze_xray import analyze_xray_bp
from ai_nutritionist import ai_nutritionist_bp
from medichat import medichat_bp
from model_downloader import ensure_local_model

app = Flask(__name__)
app.secret_key = "your-secret-key"

app.register_blueprint(symptom_analyzer_bp)
app.register_blueprint(analyze_xray_bp)
app.register_blueprint(ai_nutritionist_bp)
app.register_blueprint(medichat_bp)

def setup_models():
    model_files = [
        'best_model.pkl',
        'svc.pkl',
        'disease_label_encoder.pkl',
        'brain_tumor/Brain_Tumor_model.pt',
        'brain_tumor/weights.pt',
        'Pneumonia/Pneumonia.pth',
        'Pneumonia/PneumoniaResnet.pth',
        'Pneumonia/weights.pth',
        'skin_cancer/skin_cancer.pt',
        'skin_cancer/weights.pt',
        'vectorstore/db_faiss/index.faiss',
        'vectorstore/db_faiss/index.pkl',
    ]

    try:
        for model_file in model_files:
            ensure_local_model(model_file)
        app.logger.info("Models setup successfully.")
    except Exception as e:
        app.logger.error(f"Error setting up models: {e}")

# Run this setup function when the application starts
setup_models()

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(host="0.0.0.0", port=8000)
