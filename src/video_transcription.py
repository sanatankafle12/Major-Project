import speech_recognition as sr
from moviepy.editor import *

# Load the video file
video_path = "../videos/neso.mp4"
try:
    video = VideoFileClip(video_path)
except Exception as e:
    print("Error loading video file:", e)
    exit()

# Extract the audio from the video file
audio = video.audio
print(audio)

# Use SpeechRecognition to transcribe the audio
r = sr.Recognizer()
try:
    text = r.recognize_google(audio)
except sr.UnknownValueError:
    print("Error: Speech recognition could not understand audio")
    exit()
except sr.RequestError as e:
    print("Error: Could not request results from speech recognition service; {0}".format(e))
    exit()

# Save the transcription to a text file
try:
    with open("transcription.txt", "w") as file:
        file.write(text)
except Exception as e:
    print("Error saving transcription:", e)
    exit()

print("Transcription saved to 'transcription.txt'.")