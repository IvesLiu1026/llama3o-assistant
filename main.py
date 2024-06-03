import os
from dotenv import load_dotenv
from groq import Groq
from PIL import ImageGrab, Image
from openai import OpenAI
import google.generativeai as genai
import pyperclip
import cv2
import time
import pyaudio

load_dotenv()

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
web_cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

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
    # 'max_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUAL_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_VIOLENCE',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_ILLEGAL',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_MEDICAL',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_PRIVACY',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SECURITY',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SPAM',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_COPYRIGHT',
        'threshold': 'BLOCK_NONE'
    }
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config
                              # safety_settings=safety_settings
)

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    # convo = [{'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    convo.append(response)
    
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an Ai function calling model. You will determine whether extracting the uses clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the uses prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract_clipboard", "take_screenshot", "capture_webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )
    
    function_convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message

    return response.content

def take_screeshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_scr = screenshot.convert('RGB')
    rgb_scr.save(path, quality=15) # quality set as 15 to reduce file size and make it faster to upload
    
def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Could not open webcam')
        return
    path = 'webcam_capture.jpg'
    time.sleep(2)  # Add delay for initialization
    ret, frame = web_cam.read()
    if not ret:
        print('Error: Failed to capture image')
        return
    cv2.imwrite(path, frame)
    print(f'Image saved to {path}')
    
def extract_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        return 'Error: Could not extract clipboard content'
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the use. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \n\nUSER: {prompt}'
    )
    
    respond = model.generate_content([prompt, img])
    return respond.text

def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False
    
    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='onyx',
        response_format='pcm',
        input=text,
    ) as response:
        silenece_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silenece_threshold:
                    player_stream.write(chunk)
                    stream_start = True
    
while True:
    prompt = input('USER: ')
    call = function_call(prompt)
    
    if 'take_screenshot' in call:
        print('Screenshot taken')
        take_screeshot()
        visual_context = vision_prompt(prompt, 'screenshot.jpg')
    elif 'capture_webcam' in call:
        print('Webcam capture taken')
        web_cam_capture()
        visual_context = vision_prompt(prompt, 'webcam_capture.jpg')
    elif 'extract_clipboard' in call:
        print('Clipboard extracted')
        paste = extract_clipboard()
        prompt = f'{prompt}\n\n CLIPBOARD CONTENT: {paste}'
        visual_context = None
    else:
        visual_context = None
        
    response = groq_prompt(prompt=prompt, img_context=visual_context)
    print(response)
    # speak(response)