import base64
import json
import logging
import os
import ngrok
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()
from flask import Flask
from flask_sock import Sock
from flask import Flask, request, Response, jsonify
import io
from pydub import AudioSegment
import audioop
import time
import requests
import openai
from prompts import SYSTEM_PROMPT
from array import array
import numpy as np
from conversation_manager import ConversationManager
from vad import VADModel


app = Flask(__name__)
sock = Sock(app)

HTTP_SERVER_PORT = 5000
INCOMING_CALL_ROUTE = '/'
WEBSOCKET_ROUTE = '/media'
OUTBOUND_CALL_ROUTE = "/outbound"

# external endpoits
ASR_ENDPOINT = os.getenv("ASR_ENDPOINT", "http://localhost:8001/inference")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:8000/v1")
TTS_ENDPOINT = os.getenv("TTS_ENDPOINT", "http://localhost:8003/tts")


# Twilio authentication
account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
ngrok_auth_token = os.environ["NGROK_AUTHTOKEN"]
TWILIO_NUMBER = os.environ['TWILIO_NUMBER']
  
client = Client(account_sid, auth_token)
ngrok.set_auth_token(ngrok_auth_token)

def transcribe_audio(audio_file):
  # Prepare the files and data for the request
  files = {
      'file': open(audio_file, 'rb'),  # Update with your audio file path
  }
  data = {
      # 'model': 'openai/whisper-large-v2',  # Your model name
      'language': 'es',  # Optional: language of the audio
      'response_format': 'json',  # Optional: desired response format
      'temperature': 0.0,  # Optional: temperature setting
  }

  # Send the POST request
  response = requests.post(ASR_ENDPOINT, files=files, data=data)
  files['file'].close()
  text = response.json()["text"]
  return text


client_openai = openai.OpenAI(
    base_url=LLM_ENDPOINT, # "http://<Your api-server IP>:port"
    api_key="sk-no-key-required"
)

def generate_llm_response(messages):
  model = client_openai.models.list().data[0].id
  completion = client_openai.chat.completions.create(
  model=model,
  messages=messages,
  max_tokens=100
  )
  llm_prediction = completion.choices[0].message.content
  return llm_prediction

def to_mulaw_bytes(audio_bytes, sample_width=2, original_frame_rate=22050, desired_frame_rate=8000):
  # Resample the audio to 8000 Hz if it's different from the original frame rate
  if original_frame_rate != desired_frame_rate:
      audio_bytes, _ = audioop.ratecv(audio_bytes, sample_width, 1, original_frame_rate, desired_frame_rate, None)

  # Convert to μ-law
  mu_law_audio_bytes = audioop.lin2ulaw(audio_bytes, sample_width)
  return mu_law_audio_bytes

def generate_tts_response(llm_response):
  data = {
      "text": llm_response,
  }

  response = requests.post(TTS_ENDPOINT, data=data)

  audio_bytes = response.content
  mulaw_bytes = to_mulaw_bytes(audio_bytes, sample_width=2, original_frame_rate=22050, desired_frame_rate=8000)
  mulaw_bytes_b64_decoded = base64.b64encode(mulaw_bytes).decode('utf-8')
  return mulaw_bytes_b64_decoded




@app.route(INCOMING_CALL_ROUTE, methods=['GET', 'POST'])
def receive_call():
    if request.method == 'POST':
        xml = f"""
<Response>
    <Say>
        Hello
    </Say>
    <Connect>
        <Stream url='wss://{request.host}{WEBSOCKET_ROUTE}' />
    </Connect>
</Response>
""".strip()
        return Response(xml, mimetype='text/xml')
    else:
        return f"Real-time phone call transcription app"

conversation_manager = ConversationManager()


@sock.route(WEBSOCKET_ROUTE)
def echo(ws):
    app.logger.info("Connection accepted")
    session_id = conversation_manager.create_session()
    vad_model = VADModel(time_until_stop_talking=1.2, speech_prob_stop_talking=0.7, sample_rate=16_000)
    # A lot of messages will be sent rapidly. We'll stop showing after the first one.
    has_seen_media = False
    message_count = 0
    audio_append = AudioSegment.empty()
    silence_detected = False
    audios_in_silence_counter = 0
    n_audios_to_detect_silence = 20
    primera_interaccion = True
    leo_is_speaking = False
    array_concatenated = np.array([], dtype=np.int16)
    launch_processing = False


    while True:
      if primera_interaccion and not has_seen_media:
        start_time = time.time()
        message = ws.receive()
        print("Tiempo del receive para llegar a media ", time.time() - start_time)
        data = json.loads(message)
        primera_interaccion = False
      else:
        try:
          start_time = time.time()
          message = ws.receive(timeout=0.05)
          data = json.loads(message)
        except:
           data = {"media": {"payload": b''}, "event": "media"}


        # Using the event type you can determine what type of message you are receiving
        if data['event'] == "connected":
            app.logger.info("Connected Message received: {}".format(message))
        if data['event'] == "start":
            streamSid = data["start"]["streamSid"]
            app.logger.info("Start Message received: {}".format(message))
        # if data['event'] == "media" and not leo_is_speaking:
        if data['event'] == "media" and not leo_is_speaking:
            has_seen_media = True
            # if not has_seen_media:
                # app.logger.info("Media message: {}".format(message))
            payload = data['media']['payload']
            # app.logger.info("Payload is: {}".format(payload))
            if payload == b'':
               app.logger.info("Enviando silencio")
               audio_array = np.zeros(319, dtype=np.int16)
               chunk = audio_array.tobytes()
            else:
              app.logger.info("Enviando audio")
              chunk_mulaw = base64.b64decode(payload)
              linear_bytes = audioop.ulaw2lin(chunk_mulaw, 2)
              chunk, state = audioop.ratecv(linear_bytes, 2, 1, 8000, 16000, None)
              audio_array = np.array(array("h", chunk), dtype=np.int16) # este se debe de usar para el vad
              # print(audio_array)
              # app.logger.info("audio array shape when speaking: {}".format(audio_array)))
              app.logger.info("audio array shape when speaking: {}".format(audio_array.shape))


            array_concatenated = np.concatenate((array_concatenated, audio_array), dtype=np.int16)
            if array_concatenated.shape[0]>=512:
              launch_processing, speech_prob = vad_model.is_end_of_speaking(array_concatenated[:512])
              array_concatenated = array_concatenated[512:]

              app.logger.info("Speech prob {}".format(speech_prob))
              # app.logger.info("That's {} bytes".format(len(chunk)))
              audio = AudioSegment(chunk, sample_width=2, frame_rate=16000, channels=1)
              audio_append = audio_append + audio

            if launch_processing:
              audio_file = "appended_output.wav"
              audio_append.export(audio_file, format="wav")
              app.logger.info("Transcribing audio...")
              start_time_asr = time.time()
              text = transcribe_audio(audio_file)
              end_time_asr = time.time() - start_time_asr
              app.logger.info("Text transcribe is: {} and time consumed. {}".format(text, end_time_asr))
              conversation_manager.add_user_message(session_id, text)
              conversation = conversation_manager.get_conversation(session_id)
              start_time_llm = time.time()
              llm_response = generate_llm_response(conversation)
              end_time_llm = time.time() - start_time_llm
              app.logger.info("LLM response is: {} and time consumed. {}".format(llm_response, end_time_llm))
              conversation_manager.add_ai_message(session_id, llm_response)
              start_time_tts = time.time()
              tts_response = generate_tts_response(llm_response)
              end_time_tts = time.time() - start_time_tts
              app.logger.info("TTS time consumed. {}".format(end_time_tts))
              response_data = {
                'event': 'media',
                'streamSid': streamSid,
                'media': {
                  'payload': tts_response
                  }
                  }
              ws.send(json.dumps(response_data))
              audio_append = AudioSegment.empty()
              launch_processing = False
              mark_message = { 
                  "event": "mark",
                  "streamSid": streamSid,
                  "mark": {
                      "name": "leo_stopped_talking"
                      }   
              }
              ws.send(json.dumps(mark_message))
              leo_is_speaking = True
  
        if data['event'] == "closed":
            app.logger.info("Closed Message received: {}".format(message))
            break
        message_count += 1

        if data["event"] == "mark":
            leo_is_speaking = False
            app.logger.info("mark message {}".format(data))

    app.logger.info("Connection closed. Received a total of {} messages".format(message_count))




@app.route(OUTBOUND_CALL_ROUTE, methods=['POST'])
def make_outbound_call():
    global NGROK_URL
    data = request.json
    to_number = data.get('to')

    if not to_number:
        return jsonify({"error": "Recipient phone number is required"}), 400
    
    call = client.calls.create(
    url=NGROK_URL,
    to=to_number,
    from_=TWILIO_NUMBER,
    )
    return jsonify({"message": "Call initiated", "call_sid": call.sid})


if __name__ == '__main__':
    try:
        app.logger.setLevel(logging.DEBUG)
        # from gevent import pywsgi
        # from geventwebsocket.handler import WebSocketHandler

        # server = pywsgi.WSGIServer(('', HTTP_SERVER_PORT), app, handler_class=WebSocketHandler)
        print("Server listening on: http://localhost:" + str(HTTP_SERVER_PORT))
        listener = ngrok.forward(f"http://localhost:{HTTP_SERVER_PORT}")
        print(f"Ngrok tunnel opened at {listener.url()} for port {HTTP_SERVER_PORT}")
        NGROK_URL = listener.url()

        # Set ngrok URL to ne the webhook for the appropriate Twilio number
        twilio_numbers = client.incoming_phone_numbers.list()
        twilio_number_sid = [num.sid for num in twilio_numbers if num.phone_number == TWILIO_NUMBER][0]
        client.incoming_phone_numbers(twilio_number_sid).update(account_sid, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")
        # server.serve_forever()
        app.run(port=HTTP_SERVER_PORT, debug=False)
    finally:
        ngrok.disconnect()
