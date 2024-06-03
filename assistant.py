from groq import Groq
from PIL import ImageGrab # for taking screenshots

groq_client = Groq(api_key="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")

def groq_prompt(prompt):
    convo = [{'role': 'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    
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
    


prompt = input('USER: ')
function_response = function_call(prompt)
print(f'FUNCTION: {function_response}')
response = groq_prompt(prompt)
print(f'ASSISTANT: {response}')

