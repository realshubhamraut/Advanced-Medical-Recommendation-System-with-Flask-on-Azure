from flask import Blueprint, render_template

medichat_bp = Blueprint('medichat', __name__, template_folder='templates')

@medichat_bp.route('/medichat')
def show_medichat():
    return render_template("medichat.html")