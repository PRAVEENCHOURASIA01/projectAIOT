# modules/voice_command.py
import threading
import speech_recognition as sr

def process_command(command):
    # Process the command and trigger corresponding actions.
    print(f"Voice Command Received: {command}")
    # For example, if command == "stop", then handle it accordingly.
    # Extend this function as needed.

def voice_listener():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Voice command listener started. Speak into the microphone...")
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            process_command(command)
        except sr.UnknownValueError:
            print("Could not understand the audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

def start_voice_listener():
    # Run the voice listener in a separate thread
    thread = threading.Thread(target=voice_listener, daemon=True)
    thread.start()
