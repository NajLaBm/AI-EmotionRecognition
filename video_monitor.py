import cv2
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import queue
import time
from datetime import datetime
import mediapipe as mp
import pickle
import csv
from utils.recorder import StreamParams,Recorder
import pandas as pd
from live_prediction import LivePredictions
import threading
import os


# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
camera_active = True 

def ajouter_entree(v1, v2, v3):
    with open('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/static/data2.csv', mode='a', newline='') as file:
        writer = csv.writer(file,delimiter=';')
        writer.writerow([v1, v2, v3])
# Load pre-trained model
with open('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/body_languagemydataset_finale.pkl', 'rb') as f:
    model = pickle.load(f)

# Fonction pour capturer la vidéo avec prédiction d'émotion
def captureVideo():

    run1=True
    debut = time.time()
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
     # Définir les paramètres de sortie pour le fichier vidéo
    video_output = 'output_video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30  # Changer la fréquence d'images si nécessaire
    frame_size = (640, 480)  # Changer la résolution si nécessaire
    
    # Initialisez l'enregistreur vidéo
    video_writer = cv2.VideoWriter(video_output, fourcc, fps, frame_size)
# Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
     while cap.isOpened() and camera_active and run1:
            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                # Extract Right Hand landmarks
                right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
                right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                
                # Extract Left Hand landmarks
                left_hand = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
                left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                
                # Concatenate rows
                row = pose_row + face_row + right_hand_row + left_hand_row
                
    #             # Append class name 
    #             row.insert(0, class_name)
                
    #             # Export to CSV
    #             with open('coords.csv', mode='a', newline='') as f:
    #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                 csv_writer.writerow(row) 

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
                #ajouter_entree("Classe","Probabilité /n Posture/main","Temps en ms")
                
                # Grab ear coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                print()
                ajouter_entree(body_language_class,str(round(body_language_prob[np.argmax(body_language_prob)],2)),datetime.now())
            except:
                pass             

            cv2.imshow('Raw Webcam Feed', image)

            # Clé de sortie de la boucle
            key = cv2.waitKey(1) & 0xFF
            
            # Condition de sortie après 5 secondes
            if time.time() - debut > 60:
                run1 = False
            
            if key == ord('q'):
                break
             
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    # Implémentez la capture vidéo avec prédiction d'émotion ici
    # Utilisez OpenCV ou une autre bibliothèque pour la capture vidéo
    #time.sleep(5)  # Exemple de capture pendant 5 secondes
    #video_emotion_prediction = "happy"  # Exemple de résultat de prédiction
    #return video_emotion_prediction

# Fonction pour capturer l'audio avec prédiction d'émotion
def captureAudio():
    debut = time.time()
    run2=True 
    while run2:   
        stream_params = StreamParams()
        recorder = Recorder(stream_params)
        recorder.record(30, "audioThread.wav") 
        if time.time() - debut > 25:
                run2 = False
        # Implémentez la capture audio avec prédiction d'émotion ici
        # Utilisez pyaudio, sounddevice ou une autre bibliothèque pour la capture audio
    filename = 'audioThread.wav'
    live_prediction = LivePredictions(filename)
    emotion=live_prediction.make_predictions()
    return emotion
# Sous-classe de Thread pour capturer la vidéo
class VideoCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.video_result = None

    def run(self):
        self.video_result = captureVideo()

    def get_result(self):
        return self.video_result

# Sous-classe de Thread pour capturer l'audio
class AudioCaptureThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.audio_result = None

    def run(self):
        self.audio_result = captureAudio()

    def get_result(self):
        return self.audio_result
    
def nettoyage_app():
    # Chemin vers votre fichier CSV
    fichier_csv = 'C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/static/data2.csv'
    try:
        # Vérifier si le fichier existe avant de le supprimer
        if os.path.exists(fichier_csv):
            os.remove(fichier_csv)
            print(f'Le fichier {fichier_csv} a été supprimé avec succès.')
        else:
            print(f'Le fichier {fichier_csv} n\'existe pas.')
    except Exception as e:
        print(f'Erreur lors de la suppression du fichier : {e}')       