import pickle 
import speech_recognition as sr
import os
# Charger le modèle pré-entraîné et les outils de prétraitement
with open('C:/Users/MSI/Desktop/lastversions/ScanEmotionProject/ScanEmotionProject/text_emotion_dataset_finale.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    vectorizer = data['vectorizer']
    label_encoder = data['label_encoder']

def predict_emotion(text):
    # Vectoriser le texte
    text_vectorized = vectorizer.transform([text])
    
    # Prédiction de l'émotion
    predicted_class_encoded = model.predict(text_vectorized)[0]  # Classe prédite (encodée)
    
    # Convertir la classe encodée en étiquette d'émotion
    predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
    
    return predicted_class

def transcribe_audio(audio_file_path):
    emotion=""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f'Transcript: {text}')
            # Prédire l'émotion à partir du texte
            emotion = predict_emotion(text)
            print(f'Predicted Emotion: {emotion}')
        except sr.UnknownValueError:
            print('Google Web Speech API could not understand audio')
        except sr.RequestError as e:
            print(f'Could not request results from Google Web Speech API; {e}')

    return emotion 


def nettoyage_app():
   

    # Chemin vers votre fichier CSV
    fichier_csv = 'static\data2.csv'

    # Vérifier si le fichier existe avant de le supprimer
    if os.path.exists(fichier_csv):
        os.remove(fichier_csv)
        print(f'Le fichier {fichier_csv} a été supprimé avec succès.')
    else:
        print(f'Le fichier {fichier_csv} n\'existe pas.')