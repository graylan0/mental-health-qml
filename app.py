import eel
import asyncio
import re
import speech_recognition as sr
import time
import collections
import threading
import logging
from textblob import TextBlob
from pennylane import numpy as np
import pennylane as qml
from concurrent.futures import ThreadPoolExecutor
import sounddevice as sd
import uuid
from scipy.io.wavfile import write as write_wav
from llama_cpp import Llama
from bark import generate_audio, SAMPLE_RATE  # Assuming you have Bark installed
from weaviate import Client
import aiosqlite

# Initialize EEL with the web folder
eel.init('web')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Weaviate client
client = Client("http://localhost:8080")

# Initialize Llama model
llm = Llama(
    model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_gpu_layers=-1,
    n_ctx=3900,
)

# Initialize a quantum device
dev = qml.device("default.qubit", wires=4)

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=3)

# Initialize variables for speech recognition
is_listening = False
recognizer = sr.Recognizer()
mic = sr.Microphone()

# Function to start/stop speech recognition
@eel.expose
def set_speech_recognition_state(state):
    global is_listening
    is_listening = state

# Function to run continuous speech recognition
def continuous_speech_recognition():
    global is_listening
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            if is_listening:
                try:
                    audio_data = recognizer.listen(source, timeout=1)
                    text = audio_to_text(audio_data)  # Convert audio to text
                    if text not in ["Could not understand audio", ""]:
                        asyncio.run(run_llm(text))
                        eel.update_chat_box(f"User: {text}")
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    eel.update_chat_box(f"An error occurred: {e}")
            else:
                time.sleep(1)

# Start the continuous speech recognition in a separate thread
thread = threading.Thread(target=continuous_speech_recognition)
thread.daemon = True  # Set daemon to True
thread.start()

# Initialize SQLite database
async def init_db():
    async with aiosqlite.connect("emotional_mapping.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS emotional_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                color_code TEXT,
                quantum_state TEXT,
                amplitude REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

# Function to insert new emotional mapping into SQLite
async def insert_emotional_mapping(color_code, quantum_state, amplitude):
    async with aiosqlite.connect("emotional_mapping.db") as db:
        await db.execute("""
            INSERT INTO emotional_mapping (color_code, quantum_state, amplitude)
            VALUES (?, ?, ?)
        """, (color_code, str(quantum_state.tolist()), amplitude))
        await db.commit()

# Function to get the most recent emotional mapping from SQLite
async def get_recent_emotional_mapping():
    async with aiosqlite.connect("emotional_mapping.db") as db:
        cursor = await db.execute("SELECT * FROM emotional_mapping ORDER BY timestamp DESC LIMIT 1")
        row = await cursor.fetchone()
        return row

async def query_weaviate_for_phones(keywords):
    try:
        query = {
            "operator": "Or",
            "operands": [
                {
                    "path": ["description"],
                    "operator": "Like",
                    "valueString": keyword
                } for keyword in keywords
            ]
        }
        results = (
            client.query
            .get('Phone', ['name', 'description', 'price'])
            .with_where(query)
            .do()
        )
        
        if 'data' in results and 'Get' in results['data']:
            return results['data']['Get']['Phone']
        else:
            return []
    except Exception as e:
        logging.error(f"An error occurred while querying Weaviate: {e}")
        return []

async def update_weaviate_with_quantum_state(quantum_state):
    try:
        # Generate a unique ID for each quantum state; you can use any other method to generate a unique ID
        unique_id = str(uuid.uuid4())
        
        # Create the data object in Weaviate
        client.data_object.create(
            "CustomerSupport",  # class_name as a positional argument
            {
                "id": unique_id,
                "properties": {
                    "quantumState": list(quantum_state)  # Convert numpy array to list
                }
            }
        )
    except Exception as e:
        logging.error(f"An error occurred while updating Weaviate: {e}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging


def audio_to_text(audio_data):
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"
    
# Function to extract keywords using summarization technique
async def extract_keywords_with_summarization(prompt):
    # Tokenize the text into individual words
    words = re.findall(r'\b\w+\b', prompt.lower())
    
    # Define a set of stop words to ignore
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    
    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Always include the keyword "phone" for better context
    filtered_words.append("phone")
    
    # Count the frequency of each word
    word_count = collections.Counter(filtered_words)
    
    # Find the 5 most common words
    common_words = word_count.most_common(5)
    
    # Extract just the words from the list of tuples
    keywords = [word[0] for word in common_words]
    
    # Print the extracted keywords to the console
    print("Extracted keywords:", keywords)
    
    return keywords

async def summarize_to_color_code(prompt):
    # Use TextBlob to analyze the sentiment of the prompt for amplitude
    analysis = TextBlob(prompt)
    sentiment_score = analysis.sentiment.polarity

    # Normalize the sentiment score to an amplitude between 0 and 1
    amplitude = (sentiment_score + 1) / 2

    # Initialize color_code to None
    color_code = None

    # Loop to keep trying until a valid color code is found
    while color_code is None:
        color_prompt = "Generate a single map to emotion html color code based upon the following text;" + prompt
        color_response = llm(color_prompt, max_tokens=350)['choices'][0]['text'].strip()
        # Print the Llama model's reply to the console
        print("Llama model's reply:", color_response)
        # Use advanced regex to find a color code in the Llama2 response
        match = re.search(r'#[0-9a-fA-F]{6}', color_response)
        if match:
            color_code = match.group(0)
        else:
            print("Retrying to get a valid color code...")

    return color_code, amplitude        

async def run_llm(prompt):
    # Summarize the user's reply into a color code and amplitude
    color_code, amplitude = await summarize_to_color_code(prompt)

    # Generate quantum state based on the color code and amplitude
    quantum_state = quantum_circuit(color_code, amplitude).numpy()
    
    # Insert the new emotional mapping into SQLite
    await insert_emotional_mapping(color_code, quantum_state, amplitude)
    
    # Get the most recent emotional mapping from SQLite for personalized recommendations
    recent_mapping = await get_recent_emotional_mapping()
    
    # Update the GUI with the quantum state before generating Llama model's reply
    eel.update_chat_box(f"Quantum State based on User's Reply: {quantum_state}")

    # Define the new role with an experimental design disclaimer
    agi_prompt = ("[Experimental Design] You are Gray00's Personal Mental Health Humanoid Partner Science AI. "
                  "Your primary role is experimental and aims to assist in mental well-being to the best of your capabilities by providing: \n"
                  "1. Emotional support and understanding.\n"
                  "2. Personalized mental health tips and exercises.\n"
                  "3. Music and recipe recommendations based on the user's mood.\n"
                  "4. Scientific insights into the user's emotional state using quantum computing.\n")

    # Custom logic based on quantum state and recent emotional mapping
    if np.argmax(quantum_state) == 0:
        query_prompt = f"Based on the recent emotional mapping {recent_mapping}, please suggest a calming recipe."
    elif np.argmax(quantum_state) == 1:
        query_prompt = f"Based on the recent emotional mapping {recent_mapping}, please suggest an uplifting song."
    else:
        query_prompt = f"Please analyze the user's input as {quantum_state}. This is the amplitude: {amplitude}. Provide insights into understanding the customer's dynamic emotional condition."

    full_prompt = agi_prompt + query_prompt
    
    # Generate the response using the Llama model
    response = llm(full_prompt, max_tokens=900)['choices'][0]['text']
    
    # Update the GUI with the Llama model's reply
    eel.update_chat_box(f"AI: {response}")
    await update_weaviate_with_quantum_state(quantum_state)
    # Convert the Llama model's reply to speech
    generate_and_play_audio(response)


# Function to generate audio for each sentence and add pauses
def generate_audio_for_sentence(sentence):
    audio = generate_audio(sentence, history_prompt="v2/en_speaker_6")
    silence = np.zeros(int(0.75 * SAMPLE_RATE))  # quarter second of silence
    return np.concatenate([audio, silence])

# Function to generate and play audio for a message
def generate_and_play_audio(message):
    sentences = re.split('(?<=[.!?]) +', message)
    audio_arrays = []
    
    for sentence in sentences:
        audio_arrays.append(generate_audio_for_sentence(sentence))
        
    audio = np.concatenate(audio_arrays)
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()

def sentiment_to_amplitude(text):
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

@qml.qnode(dev)
def quantum_circuit(color_code, amplitude):
    r, g, b = [int(color_code[i:i+2], 16) for i in (1, 3, 5)]
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    qml.RY(r * np.pi, wires=0)
    qml.RY(g * np.pi, wires=1)
    qml.RY(b * np.pi, wires=2)
    qml.RY(amplitude * np.pi, wires=3)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    return qml.state()

# EEL function to send message to Llama and get a response
@eel.expose
def send_message_to_llama(message):
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(run_llm(message))
    generate_and_play_audio(response)  # Changed this line to use the new function
    return response


# Entry point of the script
if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
        eel.start('index.html')
    except KeyboardInterrupt:
        print("Exiting program...")
        # Perform any necessary cleanup here
        exit(0)