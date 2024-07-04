#!/usr/bin/python
# https://www.youtube.com/watch?v=pi6gr_YHSuc
from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) # Load the .env file.
import requests
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import pyttsx3
import pyaudio
import pyalsa
import time
import re
import sys
import subprocess
import threading
import contextlib
try:
    import torch
except ModuleNotFoundError:
    # Error handling
    whisp_device = 'cpu'
else:
    whisp_device = 'cuda'

import cv2

# Solve ALSA error messages
# https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time/13453192#13453192


@contextlib.contextmanager
def ignoreStderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)


with ignoreStderr():
    audio_interface = pyaudio.PyAudio()

wake_trigger = os.getenv("wake_trigger")
wake_word = wake_trigger
groq_client = Groq(api_key=os.getenv("groq_api_key"))
genai.configure(api_key=os.environ["genai_api_key"])
openai_client = OpenAI(api_key=os.getenv("openai_api_key"))
web_cam = cv2.VideoCapture(0)
resource_lock = threading.Lock()
dummy = subprocess.check_output('pacmd list-sources|grep "sample spec"|cut -d " " -f5|cut -d "H" -f1|head -1|grep [0-9]', shell=True)
samp_rate = int(dummy.decode().strip())
local_tts = os.getenv("local_tts")


sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config=generation_config, safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device=whisp_device,
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)
r = sr.Recognizer()
# r.energy_threshold = 979

source = sr.Microphone(sample_rate=samp_rate)

def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))


def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n    IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content


def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message

    return response.content


def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)


def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        exit()
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    cv2.imwrite(path, frame)


def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None


def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the visual analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text


def speak(text):
    if local_tts == "alltalk":
        data = {
            "text_input": text,
            "text_filtering": "standard",
            "character_voice_gen": "en_US-hfc_female-medium.onnx",
            "output_file_name": "myoutputfile",
            "output_file_timestamp": "true",
            "narrator_enabled": "false",
            "narrator_voice_gen": "male_01.wav",
            "text_not_inside": "character",
            "language": "en",
            "autoplay": "true",
            "autoplay_volume": "0.8"
        }

        response = requests.post("http://127.0.0.1:7851/api/tts-generate", data=data)
        # os.remove("/home/sputnik/.git/alltalk_tts/outputs/myoutputfile*")
    else:
        engine = pyttsx3.init()
        engine.setProperty('rate', 190)  # Speed percent
        # engine.setProperty('volume', 0.9)  # Volume 0-1
        engine.say(text)
        engine.runAndWait()    


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    # print('wav_to_text = ', text)
    return text


def callback(recognizer, audio):
    try:
        # print("Audio data received in callback")
        prompt_audio_path = '/run/user/1000/prompt.wav'
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)

        if clean_prompt:
            prGreen(f'USER: {clean_prompt}')
            call = function_call(clean_prompt)
            visual_context = None
            if 'take screenshot' in call:
                print('Taking screenshot.')
                take_screenshot()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
            elif 'capture webcam' in call:
                print('Capturing webcam.')
                web_cam_capture()
                visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
            elif 'extract clipboard' in call:
                print('Extracting clipboard text.')
                paste = get_clipboard_text()
                clean_prompt = f'{clean_prompt} \n\n    CLIPBOARD CONTENT: {paste}'
                visual_context = None
            # else:
                # visual_context = None                
            response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
            print(f'ASSISTANT: {response}')
            speak(response)
    except Exception as e:
        print(f"Error in callback: {e}")


def start_listening():
    with source as s:
        print('\nPlease wait while microphone is adjusted for ambient noise')
        r.adjust_for_ambient_noise(s, duration=5)
    print('\nReady.  Please say', wake_word, 'followed with your prompt when you wish to chat. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.49)


def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None


start_listening()
