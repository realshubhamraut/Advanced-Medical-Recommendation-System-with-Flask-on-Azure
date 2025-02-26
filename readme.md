# Advanced Medical Recommendation System with Flask on Azure ⚕️
 <strong>with deep learning<strong>

<div style="display: flex; gap: 10px;">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://joblib.readthedocs.io/">
    <img src="https://img.shields.io/badge/-joblib-FF9900?style=flat-square&logo=python&logoColor=white" alt="Joblib">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit‑Learn">
  </a>
  <a href="https://azure.microsoft.com/">
    <img src="https://img.shields.io/badge/-Azure-0078D4?style=flat-square&logo=microsoft-azure&logoColor=white" alt="Azure">
  </a>
  <a href="https://docs.github.com/en/actions">
    <img src="https://img.shields.io/badge/-GitHub_Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white" alt="GitHub Actions">
  </a>
  <a href="https://git-lfs.github.com/">
    <img src="https://img.shields.io/badge/-Git%20LFS-F05133?style=flat-square&logo=git-lfs&logoColor=white" alt="Git LFS">
  </a>
</div>



---

**Advanced Medical Recommendation System with Flask on Azure** is an end‑to‑end solution designed to provide medical diagnostics and recommendations using machine learning and deep learning models.

The system comprises multiple modules including X-ray analysis, symptom analysis, AI nutritionist, and medical chat. It leverages deep learning (using PyTorch) for medical image classification (e.g., pneumonia, brain tumors, skin cancer), and rule‑based or ML‑based predictions for symptom analysis.

Large model files are then hosted on Azure Blob Storage and dynamically downloaded at runtime. Deployment is fully automated via GitHub Actions to Azure App Service.


[VIEW LIVE](https://advanced-medical-app.azurewebsites.net)

## Technology Stack

- #### Backend & API:
  
  ` Python, Flask, GitHub Actions `

- #### Deep Learning & Model Inference: 
  
    ` PyTorch, TorchScript, Scikit-Learn, Pickle/Joblib (SVC), Git LFS for managing large model files `

- #### Data & Storage:
  
    `Azure Blob Storage for models, Vector Storage for Document RAG based Query (context based)`

- #### Frontend & Templates:
  
  `TML with Bootstrap, jQuery UI, CSS`

- #### Deployment & Cloud Infrastructure:
  
    ` Azure App Service, Azure Blob Storage, GitHub Actions CI/CD `

## 🌟 Key Features

- ##### **Modular Medical Diagnostics:**  
  - **X-ray Analysis:** Upload and analyze medical images (MRI, X-ray, or skin lesion) for conditions such as pneumonia, brain tumor, and skin cancer.  
  - **Symptom Analyzer:** Input symptoms in free text to receive diagnostic predictions along with details like description, precautions, medications, diet, and workouts.  
  - **AI Nutritionist:** Analyze meal images and obtain detailed nutritional analysis in a structured table format.
  - **MediChat:** Engage with an interactive chatbot for medical consultations and answering FAQs.

- ##### **Scalable Model Management:**  
  - Leverages Azure Blob Storage to host and deliver large deep learning models at runtime without bloating the repository.
  - Uses Git LFS to manage large files during development.

- ##### **Cloud-Ready Deployment:**  
  - Fully automated deployment using GitHub Actions that build and deploy your application to Azure App Service.
  - Supports continuous deployment and easy scaling under the Azure ecosystem.

- ##### **User-Friendly Interface:**  
  - Dynamic and responsive web pages built with Bootstrap, HTML, and jQuery.
  - Interactive forms for uploading images, entering symptoms, and receiving real‑time diagnostic feedback.

## 🛠️ Installation & Setup

1. ##### **Clone the Repository:**

   ```bash
   git clone https://github.com/realshubhamraut/Advanced-Medical-Recommendation-System-with-Flask-on-Azure.git
   cd Advanced-Medical-Recommendation-System-with-Flask-on-Azure
   ```

2. ##### **Install Dependencides:**
    Create a virtual environment and install required packages:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. ##### **Configure Environment Variable:**
    Create a .env file in the root directory with any required keys, e.g.:

    ```bash
    GOOGLE_API_KEY=your_google_api_key_here
    HF_TOKEN=your_hugging_face_token
    ```

##### 4. **Azure Blob Storage Setup:**

- Upload your local models folder to your Azure Blob Storage container (container name: models).
- Ensure the container is public or appropriately configured.
- The helper module model_downloader.py is configured to download model files at runtime if not present locally.

##### 5. **Local Testing:**

**Run the Flask app**

`python main.py`



Visit `http://127.0.0.1:5000` in your browser to test the application.

##### 6. Deployment 🚀:

Deploying to Azure App Service via GitHub Actions
    
##### 1. Azure App Service:

    - Create a new Web App in the Azure Portal using the Web App option.
    - Choose Publish: Code and select a Python runtime matching your project (e.g., Python 3.12).

2. Configure GitHub Actions:

    Your repository includes a workflow file under workflows (e.g., azure-webapps-deploy.yml) which:

    - Sets up the Python environment.
    - Installs dependencies.
    - Zips and deploys the artifact to Azure App Service.

Note: Update GitHub Secrets with the following:

- __clientidsecretname__
- __tenantidsecretname__
- __subscriptionidsecretname__

Commit the workflow file, and every push to the main branch will trigger an automatic deployment.


---
and finally issues and improvements / pull requests are always welcomed.

#### ⚠️ Disclaimer: 

Important: This project is provided for experimental purposes only. and I have only designed it in a way to explore the possibilities of integrating healthcare with AI. Always seek the advice of qualified medical professionals regarding any medical diagnosis or treatment. Do not solely rely on the outputs of this repository for clinical decisions.

---
*connect with me on [LinkedIn](https://www.linkedin.com/in/contactshubhamraut/) to develop something cool.*