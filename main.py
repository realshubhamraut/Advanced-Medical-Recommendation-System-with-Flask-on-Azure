from flask import Flask, render_template

# Import blueprints from each module.
from symptom_analyzer import symptom_analyzer_bp
from analyze_xray import analyze_xray_bp
from ai_nutritionist import ai_nutritionist_bp
from medichat import medichat_bp

app = Flask(__name__)
app.secret_key = "your-secret-key"

# Register blueprints.
app.register_blueprint(symptom_analyzer_bp)
app.register_blueprint(analyze_xray_bp)
app.register_blueprint(ai_nutritionist_bp)
app.register_blueprint(medichat_bp)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)