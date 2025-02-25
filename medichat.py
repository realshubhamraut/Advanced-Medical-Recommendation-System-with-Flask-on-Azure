from flask import Blueprint, render_template, request, redirect, url_for, flash, session
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

medichat_bp = Blueprint('medichat', __name__, template_folder='templates')

DB_FAISS_PATH = "vectorstore/db_faiss"

_vectorstore = None
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        _vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return _vectorstore

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN,
                      "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

@medichat_bp.route('/medichat', methods=["GET", "POST"])
def medichat():
    if "messages" not in session:
        session["messages"] = []
    
    conv_history = session.get("messages")
    answer_text = None
    docs = []
    
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            conv_history.append({"role": "user", "content": user_query})
            session["messages"] = conv_history
            session.modified = True

            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            HF_TOKEN = os.environ.get("HF_TOKEN")
            
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    flash("Failed to load the vector store", "error")
                    return redirect(url_for("medichat.medichat"))
    
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
    
                response = qa_chain.invoke({'query': user_query})
                answer_text = response.get("result")
                docs = response.get("source_documents", [])
                
                conv_history.append({"role": "assistant", "content": answer_text})
                session["messages"] = conv_history
                session.modified = True
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                flash(error_msg, "error")
                conv_history.append({"role": "assistant", "content": error_msg})
                session["messages"] = conv_history
                session.modified = True
    
    return render_template("medichat.html", messages=conv_history, answer=answer_text, docs=docs)