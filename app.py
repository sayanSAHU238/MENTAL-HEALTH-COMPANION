import base64
import requests
import json
from transformers import pipeline
from flask import Flask, request, jsonify, render_template, Response
import speech_recognition as sr
import cv2
import numpy as np
from pydub import AudioSegment
import io
import logging
import threading
import logging

logging.basicConfig(level=logging.INFO)


# Initialize the emotion analysis model
emotion_model = pipeline('text-classification', model='bhadresh-savani/distilbert-base-uncased-emotion')

# Initialize the Clarifai API key and URL
CLARIFAI_API_KEY = "0f5753c35e8a4577b60c8387aecc8d0f"
CLARIFAI_API_URL = "https://api.clarifai.com/v2/models/face-detection/outputs"

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the Hugging Face API endpoint and your API key
API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
API_KEY = "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"

# List of tips for overcoming a bad day
tips_for_bad_day = [
    "Take a few deep breaths and try to relax.",
    "Go for a walk or get some fresh air.",
    "Talk to a friend or family member about how you're feeling.",
    "Engage in an activity that you enjoy or find relaxing.",
    "Write down your thoughts and feelings in a journal.",
    "Practice mindfulness or meditation.",
    "Consider doing something creative like drawing or painting."
]

# Function to get a response from the Hugging Face model
def get_model_response(user_input):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": user_input}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'No response text found')
        else:
            return "Unexpected response format"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making API request: {str(e)}")
        return "Error generating response"

# Function to transcribe audio
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return "Error during transcription"

# Convert audio file to PCM WAV format
def convert_to_pcm_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    pcm_wav_file = io.BytesIO()
    audio.export(pcm_wav_file, format='wav', codec='pcm_s16le')  # Ensure PCM encoding
    pcm_wav_file.seek(0)
    return pcm_wav_file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if user_input:
        try:
            # Analyze emotions from user input
            emotion_result = emotion_model(user_input)
            emotion = emotion_result[0]['label']
            logging.info(f"Detected emotion: {emotion}")
            
            # Generate a response using the text generation model
            response = get_model_response(user_input)
            
            # Define responses for severe emotional states
            severe_responses = {
                "anger": "I understand you're feeling really angry right now. It's important to talk to someone who can help. Consider reaching out to a mental health professional or a trusted person in your life.",
                "sadness": "I'm really sorry you're feeling this way. It can be helpful to talk to a mental health professional or a trusted friend. They can provide the support you need.",
                "fear": "I can see that you're feeling scared. It's important to talk to a mental health professional who can offer you the support and guidance you need. Please reach out for help."
            }
            
            # Tips for overcoming a bad day
            if "bad day" in user_input.lower():
                tip = np.random.choice(tips_for_bad_day)
                empathetic_response = f"Here is a tip to overcome a bad day: {tip}"
            elif emotion in severe_responses:
                empathetic_response = severe_responses[emotion]
            else:
                empathetic_responses = [
                    "It sounds like you're feeling overwhelmed right now.",
                    "I understand that this can be really tough.",
                    "It's okay to feel this way; talking about it might help.",
                    "I'm here for you. Sometimes sharing helps lighten the load.",
                    "You're not alone in this; I'm here to listen."
                ]
                empathetic_response = np.random.choice(empathetic_responses) + " " + response.strip()
            
            # Log the combined response for debugging
            logging.info(f"Combined response: {empathetic_response}")
            
            return jsonify({'response': empathetic_response})
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return jsonify({'error': 'Error generating response'}), 500
    return jsonify({'error': 'No message provided'}), 400

@app.route('/voice', methods=['POST'])
def voice():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    logging.info(f"Received audio file: {audio_file.filename}")

    # Convert audio to a format compatible with speech_recognition
    try:
        if audio_file.filename.endswith('.wav'):
            # Convert to PCM WAV format
            audio_file = convert_to_pcm_wav(audio_file)
            logging.info("Audio file converted to PCM WAV format.")
        elif audio_file.filename.endswith('.mp3'):
            # Convert MP3 to PCM WAV format
            audio = AudioSegment.from_mp3(audio_file)
            pcm_wav_file = io.BytesIO()
            audio.export(pcm_wav_file, format='wav', codec='pcm_s16le')  # Ensure PCM encoding
            pcm_wav_file.seek(0)
            audio_file = pcm_wav_file
            logging.info("Audio file converted to PCM WAV format.")
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Transcribe the audio
        audio_file.seek(0)  # Reset file pointer before processing
        try:
            transcribed_text = transcribe_audio(audio_file)
            logging.info(f"Transcribed text: {transcribed_text}")
            
            # Send the transcribed text to the /chat endpoint
            chat_response = requests.post("http://127.0.0.1:5000/chat", json={"message": transcribed_text})
            
            if chat_response.status_code == 200:
                response_json = chat_response.json()
                return jsonify({'transcribed_text': transcribed_text, 'response': response_json.get('response', 'Error in response')})
            else:
                logging.error("Error from chat endpoint: " + chat_response.text)
                return jsonify({'error': 'Error from chat endpoint'}), 500
        except Exception as e:
            logging.error(f"Error during transcription: {str(e)}")
            return jsonify({'error': 'Error transcribing audio'}), 500
    except Exception as e:
        logging.error(f"Error processing audio file: {str(e)}")
        return jsonify({'error': 'Error processing audio file'}), 500



@app.route('/facial', methods=['POST'])
def facial():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    headers = {
        "Authorization": f"Key {CLARIFAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": [
            {
                "data": {
                    "image": {
                        "base64": base64.b64encode(image_data).decode('utf-8')
                    }
                }
            }
        ]
    }

    try:
        response = requests.post(CLARIFAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        logging.info(f"Clarifai API response: {result}")

        if 'outputs' in result and result['outputs']:
            regions = result['outputs'][0]['data'].get('regions', [])
            if regions:
                concepts = regions[0]['data'].get('concepts', [])
                if concepts:
                    # Get the top concept with the highest confidence score
                    top_concept = max(concepts, key=lambda x: x['value'])
                    
                    # Define mappings for more specific emotions
                    emotion_mapping = {
                        'BINARY_POSITIVE': 'Happy',
                        'BINARY_NEGATIVE': 'Sad',
                        # Add more mappings if available
                    }
                    
                    # Determine the emotion based on the concept and confidence score
                    confidence_score = top_concept['value']
                    emotion = emotion_mapping.get(top_concept['name'], 'Unknown')

                    # Adjust emotion based on confidence thresholds
                    if confidence_score < 0.5:
                        emotion = 'Neutral'
                    elif confidence_score < 0.7:
                        emotion = 'Somewhat ' + emotion.lower()
                    elif confidence_score >= 0.7:
                        emotion = 'Very ' + emotion.lower()

                    return jsonify({'emotion': emotion, 'score': confidence_score})
        return jsonify({'error': 'Unable to detect emotion'}), 500
    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return jsonify({'error': 'Error analyzing image'}), 500



def start_video_feed():
    global output_frame, lock
    output_frame = None
    lock = threading.Lock()

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame to the global variable
        with lock:
            output_frame = frame.copy()

    # Release the video capture
    video_capture.release()

@app.route('/video_feed')
def video_feed():
    global output_frame, lock

    def generate():
        while True:
            with lock:
                if output_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', output_frame)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    video_thread = threading.Thread(target=start_video_feed)
    video_thread.daemon = True
    video_thread.start()
    app.run(debug=True)
