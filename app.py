from flask import Flask, request,render_template, redirect,session,url_for, send_from_directory,Response, flash,jsonify,send_file
from flask_sqlalchemy import SQLAlchemy 
import os
import numpy as np
import bcrypt
import pdfkit
from keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import pickle 
from sklearn.metrics import accuracy_score # Accuracy metrics 
import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
from live_prediction import LivePredictions
import pandas as pd
from datetime import date, time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import time
from models.User import db,User
from models.consultation import db,Consultation
from models.Patient import db,Patient,get_age_distribution,get_patients_by_name
import threading
from video_monitor import VideoCaptureThread, AudioCaptureThread
from text_monitor import transcribe_audio,nettoyage_app,predict_emotion
import base64
import json
from io import BytesIO
import plotly.express as px
import plotly.io as pio
import traceback


app = Flask(__name__,template_folder='template')

config=pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///MyScanEmotiondataa.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] ='mykey'
app.config['ALLOWED_EXTENSIONS'] = ['.mp3', '.flac', '.aac', '.wav']
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['SQLALCHEMY_POOL_SIZE'] = 20  # Taille maximale de la mise en pool des connexions
app.config['SQLALCHEMY_POOL_TIMEOUT'] = 30  # Délai d'attente pour obtenir une connexion de la mise en pool
app.config['SQLALCHEMY_POOL_RECYCLE'] = 3600  # Durée en secondes après laquelle une connexion sera réinitialisée
pdf_filename=""

# Variables globales pour les threads et le contrôle d'arrêt
video_thread = None
audio_thread = None
terminate_threads = False
run1=False
run2=False

exit=threading.Event()
# Configurer le logger de Flask
db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/')
def welcome():
    return render_template('acceuil.html')

@app.route('/notice')
def notice():
    if session['email']:
            user = User.query.filter_by(email=session['email']).first()
    return render_template('notice.html',user=user)

@app.route('/search')
def search():
    role = session['role']
    query = request.args.get('query', '')
    patients = get_patients_by_name(query)
    
    if role == 'technicien':
        anonymized_patients = [
            {
                "id": p.id,
                "nom": f"Patient {chr(65 + i)}",      # Anonymized name
                "profession": "Confidential",         # Anonymized profession
                "age": p.age,                         # Display actual age
                "etat_civil": p.etat_civil            # Display actual marital status
            }
            for i, p in enumerate(patients)
        ]
        return jsonify(anonymized_patients)
    else:
        return jsonify(patients)    
         
#--------------------------------------------------------------------------------------------------------------------------
@app.route('/logout')
def logout():
    nettoyage_app()
    session.pop('email',None)
    return redirect('/login_interface')

@app.route('/predict/<int:consultation_id>/<int:patient_id>', methods=['GET', 'POST'])
def predict(patient_id,consultation_id):
    patient=Patient.getPatient(patient_id)
    consultation=Consultation.getConsultation(consultation_id)
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_DIRECTORY'],
                secure_filename(file.filename))
            file.save(filename)
            return redirect(url_for('classify_and_show_results',
                filename=filename,patient_id=patient_id,consultation_id=consultation_id))
    return 'File uploaded and saved.'

@app.route('/start_capture/<int:consultation_id>/<int:patient_id>', methods=['POST', 'GET'])
def start_capture(patient_id, consultation_id):
    patient = Patient.getPatient(patient_id)
    consultation = Consultation.getConsultation(consultation_id)
    
    # Récupérer le rôle de l'utilisateur depuis la session
    role = session.get('role')
    
    # Anonymiser le nom du patient si le rôle est technicien
    if role == 'technicien':
        patient_name = f"Patient {chr(65 + patient_id)}"  # Anonymisé, exemple avec l'ID du patient
    else:
        patient_name = f"{consultation.patient.nom}_{consultation.patient.prenom}"

    # Démarrage des threads pour les captures
    video_thread = VideoCaptureThread()
    audio_thread = AudioCaptureThread()
    
    video_thread.start()
    audio_thread.start()
    
    video_thread.join()
    audio_thread.join()
    
    nouveau_nom = f'C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/audio/{patient_name}_consultation_{consultation_id}.wav'

    os.rename('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/audio/audioThread.wav', nouveau_nom)
    live_prediction = LivePredictions(nouveau_nom)
    emotion = live_prediction.make_predictions()

        # Charger le fichier CSV
    df = pd.read_csv('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/static/data2.csv', sep=';', names=['emotion', 'intensity', 'timestamp'])
    result = df.groupby('emotion')['intensity'].sum().reset_index()

    # Convertir les données en format pour le rendu HTML
    data = result.to_dict(orient='records')

    # Calculer la durée totale par émotion
    total_duration = calculate_duration(df)

    # Calculer le total de toutes les durées pour les pourcentages
    total_duration['percentage'] = total_duration['duration'] / total_duration['duration'].sum() * 100

    # Créer un graphique en camembert avec Plotly
    fig = px.pie(total_duration, names='emotion', values='percentage')

    # Convertir le graphique en HTML
    graph_html = pio.to_html(fig, full_html=False)

    # Convertir le DataFrame en tableau HTML
    table_html = total_duration.drop(columns='percentage').to_html(classes='table table-striped', index=False)


    # Transcrire le texte et prédire l'émotion du discours
    text = transcribe_audio(nouveau_nom)
    text_emotion = predict_emotion(text)
    
    # PDF options
    options = {
        "orientation": "landscape",
        "page-size": "A4",
        "margin-top": "1.0cm",
        "margin-right": "1.0cm",
        "margin-bottom": "1.0cm",
        "margin-left": "1.0cm",
        "enable-local-file-access": "",
        "encoding": "UTF-8",
    } 

    if session['email']:
        user1 = User.query.filter_by(email=session['email']).first()
    
    return render_template("rapport.html", resultat=emotion, patient=patient, consultation=consultation,user=user1, table_html=table_html,data=json.dumps(data), graph_html=graph_html, text_emotion=text_emotion)



@app.route('/pdf/<int:consultation_id>/<int:patient_id>/<string:resultat>/<string:user>/<string:text_emotion>', methods=['GET','POST'])
def pdf(consultation_id, patient_id, resultat, user, text_emotion):
    patient = Patient.getPatient(patient_id)
    consultation = Consultation.getConsultation(consultation_id)

    # Vérifier le rôle de l'utilisateur
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()

    # Afficher un nom de patient anonyme si l'utilisateur est un technicien
    if user.role == 'technicien':
        patient_name = f"Patient {chr(65 + patient_id)}"
    else:
        patient_name = f"{consultation.patient.nom}_{consultation.patient.prenom}"

    if request.method == 'POST':
        # Récupérer les données envoyées depuis le formulaire
        avis_expert = request.form.get('avis_expert')

    options = {
        "orientation": "landscape",
        "page-size": "A4",
        "margin-top": "1.0cm",
        "margin-right": "1.0cm",
        "margin-bottom": "1.0cm",
        "margin-left": "1.0cm",
        "enable-local-file-access": "",
        "encoding": "UTF-8", 
    }

    
     # Charger le fichier CSV pour obtenir les données du tableau
    df = pd.read_csv('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/static/data2.csv', sep=';', names=['emotion', 'intensity', 'timestamp'])

    # Calculer la durée totale par émotion
    total_duration = calculate_duration(df)

    # Calculer le total de toutes les durées pour les pourcentages
    total_duration['percentage'] = total_duration['duration'] / total_duration['duration'].sum() * 100

    # Convertir le DataFrame en tableau HTML (sans la colonne des pourcentages)
    table_html = total_duration.drop(columns='percentage').to_html(classes='table table-striped', index=False)
    probabilite_totale = df.groupby('emotion')['intensity'].sum()
    class_max_probabilite = probabilite_totale.idxmax()
    max_probabilite = probabilite_totale.max()

    if (int(max_probabilite)) >= 80:
        inputValue = class_max_probabilite
    else:
        inputValue = resultat

    first_rows = []
    audio_file = f"C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/audio/{patient_name}_consultation_{consultation_id}.wav"

    previous_emotion = None
    previous_time = None

    live_prediction = LivePredictions(audio_file)
    emotion = live_prediction.make_predictions()

   

    # Traitement de l'émotion
    if inputValue == 'calm':
        inputValue = 'neutral'
        
    if class_max_probabilite == inputValue == text_emotion:
        res = text_emotion
    elif class_max_probabilite != inputValue and inputValue != text_emotion and class_max_probabilite != text_emotion:
        res = text_emotion
    elif class_max_probabilite == inputValue and inputValue != text_emotion:
        res = inputValue
    elif class_max_probabilite != inputValue and inputValue == text_emotion:
        res = text_emotion
    else:
        res = text_emotion
    img_base64 = img_to_base64('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/static/assets/img/tampon.jpg')

    # Générer le PDF avec les données de l'émotion et de la durée
    out = render_template("pdf.html", user=user, table_html=table_html, consultation=consultation, patient=patient, resultat=res, inputValue=inputValue, speech_emotion=emotion, text_emotion=text_emotion, class_max_probabilite=class_max_probabilite, avis_expert=avis_expert,
                                   img_base64=img_base64)

    # Déterminer le nom du fichier PDF
    pdf_filename = f"{patient_name}_consultation_{consultation_id}.pdf"
    pdf = pdfkit.from_string(out, options=options, configuration=config)

    # Chemin de sauvegarde du fichier PDF
    pdf_path = f"C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/{pdf_filename}"

    with open(pdf_path, 'wb') as file:
        file.write(pdf)

    with open(pdf_path, 'rb') as f:
        c = Consultation.getConsultation(consultation_id)
        c.resume_pdf = f.read()
        c.emotion = res
        c.avis_expert = avis_expert
    c.save()

    # Créer une réponse avec le PDF
    response = Response(pdf, mimetype="application/pdf")
    response.headers["Content-Disposition"] = f"inline; filename={pdf_filename}"

    return response



if __name__ == '__main__':

    app.run(debug = True)    