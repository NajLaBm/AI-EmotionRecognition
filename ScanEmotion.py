import os
from flask import Flask, render_template, request, jsonify, send_file,Response
import cv2
import moviepy.editor as mp
import numpy as np
import librosa
from keras.models import load_model
import subprocess

app = Flask(__name__,template_folder='template')

# Charger votre modèle de détection d'émotions pré-entraîné
emotion_model = load_model('Emotion_Voice_Detection_Model.h5')

# Dossier de stockage temporaire pour les fichiers vidéo et audio
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)




def extract_audio_from_video(video_path):
    # Extraire l'audio de la vidéo en utilisant ffmpeg via subprocess
    audio_path = os.path.join(UPLOAD_FOLDER, 'temp_audio.wav')
    command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path


def predict_emotion(audio_path):
    # Charger l'audio et le prétraiter
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    X = np.expand_dims(mfccs, axis=0)

    # Prédiction avec le modèle
    emotion_class = np.argmax(emotion_model.predict(X), axis=1)
    emotion_labels = ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']

    return emotion_labels[emotion_class[0]]


@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/recordxx', methods=['GET'])
def recordxx():
    global camera
    # Initialisez votre caméra ici
    camera = cv2.VideoCapture(0)  # 0 pour la caméra par défaut
    # Capturer une vidéo
    video_path = os.path.join(UPLOAD_FOLDER, 'temp_video.mp4')
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    """for _ in range(100):
        ret, frame = video_capture.read()
        if ret:
            out.write(frame)

    out.release()
    video_capture.release()"""

    # Extraire l'audio de la vidéo
    """audio_path = extract_audio_from_video(video_path)

    # Prédire l'émotion à partir de l'audio extrait
    emotion = predict_emotion(audio_path)

    # Supprimer les fichiers temporaires après utilisation
    os.remove(video_path)
    os.remove(audio_path)"""

    return jsonify({'emotion': "emotion"})

@app.route('/stop_recording')
def stop_recording():
    global camera
    # Arrêtez l'enregistrement vidéo
    if camera is not None:
        camera.release()  # Libère la ressource de la caméra
        camera = None



def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/record')
def record():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)