from flask import Blueprint, render_template

analyze_xray_bp = Blueprint('analyze_xray', __name__, template_folder='templates')

@analyze_xray_bp.route('/analyze-xray')
def show_analyze_xray():
    return render_template("analyze-xray.html")