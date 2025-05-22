import base64
import json
import logging
import os
import ngrok
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi import Request
import io
from pydub import AudioSegment
import audioop
import time
import requests
import openai
from prompts import SYSTEM_PROMPT
from array import array
import numpy as np
from starlette.websockets import WebSocketState
import asyncio  # For async operations
import aiohttp  # For async requests
from conversation_manager import ConversationManager
from vad import VADModel

app = FastAPI()

HTTP_SERVER_PORT = 5000
INCOMING_CALL_ROUTE = '/'
WEBSOCKET_ROUTE = '/media'
OUTBOUND_CALL_ROUTE = "/outbound"

# External endpoints
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
conversation_manager = ConversationManager()

# Helper to transcribe audio
def transcribe_audio(audio_file):
    files = {'file': open(audio_file, 'rb')}
    data = {'language': 'es', 'response_format': 'json', 'temperature': 0.0}
    response = requests.post(ASR_ENDPOINT, files=files, data=data)
    files['file'].close()
    text = response.json()["text"]
    return text

client_openai = openai.OpenAI(
    base_url=LLM_ENDPOINT,
    api_key="sk-no-key-required"
)

async def stream_llm_response(messages):
    model = client_openai.models.list().data[0].id
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{LLM_ENDPOINT}/chat/completions",
            json={"model": model, "messages": messages, "stream": True}
        ) as resp:
            async for chunk in resp.content:
                chunk_decoded = chunk.decode('utf-8').strip()
                if chunk_decoded.startswith('data: '):
                    message = chunk_decoded.split("data:")[-1].strip()
                    # logging.info("message {}".format(message))

                    if message=="[DONE]":
                        yield message
                        break
                    
                    data = json.loads(message)
                    text = data["choices"][0]["delta"]["content"]
                    yield text

def to_mulaw_bytes(audio_bytes, sample_width=2, original_frame_rate=22050, desired_frame_rate=8000):
    if original_frame_rate != desired_frame_rate:
        audio_bytes, _ = audioop.ratecv(audio_bytes, sample_width, 1, original_frame_rate, desired_frame_rate, None)
    mu_law_audio_bytes = audioop.lin2ulaw(audio_bytes, sample_width)
    return mu_law_audio_bytes

async def generate_tts_response(llm_chunk):
    data = {"text": llm_chunk}
    async with aiohttp.ClientSession() as session:
        async with session.post(TTS_ENDPOINT, data=data) as response:
            audio_bytes = await response.read()
            mulaw_bytes = to_mulaw_bytes(audio_bytes, sample_width=2, original_frame_rate=22050, desired_frame_rate=8000)
            mulaw_bytes_b64_decoded = base64.b64encode(mulaw_bytes).decode('utf-8')
            return mulaw_bytes_b64_decoded

@app.post(INCOMING_CALL_ROUTE)
async def receive_call(request: Request):
        global NGROK_URL
        host = NGROK_URL.split("//")[-1]

        xml = f"""
    <Response>
        <Say>
            Hello
        </Say>
        <Connect>
            <Stream url='wss://{host}{WEBSOCKET_ROUTE}' />
        </Connect>
    </Response>
    """.strip()
        return Response(content=xml, media_type='text/xml')

@app.websocket(WEBSOCKET_ROUTE)
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = conversation_manager.create_session()
    vad_model = VADModel(time_until_stop_talking=1, speech_prob_stop_talking=0.7, sample_rate=16_000)
    has_seen_media = False
    audio_append = AudioSegment.empty()
    array_concatenated = np.array([], dtype=np.int16)
    leo_is_speaking = False
    streamSid = None
    launch_processing = False

    try:
        while True:
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            if not has_seen_media:
                message = await websocket.receive_text()
                data = json.loads(message)
            else:
                try:
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                    data = json.loads(message)
                except:
                    data = {"media": {"payload": b''}, "event": "media"}

            if data['event'] == "connected":
                logging.info("Connected Message received: {}".format(message))

            if data['event'] == "start":
                streamSid = data["start"]["streamSid"]
                logging.info("Start Message received: {}".format(message))

            if data['event'] == "media" and not leo_is_speaking:
                has_seen_media = True
                payload = data['media']['payload']
                if payload == b'':
                    audio_array = np.zeros(319, dtype=np.int16)
                    chunk = audio_array.tobytes()
                    logging.info("sendig silence")
                else:
                    chunk_mulaw = base64.b64decode(payload)
                    linear_bytes = audioop.ulaw2lin(chunk_mulaw, 2)
                    chunk, _ = audioop.ratecv(linear_bytes, 2, 1, 8000, 16000, None)
                    audio_array = np.array(array("h", chunk), dtype=np.int16)
                    logging.info("sendig audio")
                
                array_concatenated = np.concatenate((array_concatenated, audio_array), dtype=np.int16)
                
                if array_concatenated.shape[0] >= 512:
                    launch_processing, speech_prob = vad_model.is_end_of_speaking(array_concatenated[:512])
                    array_concatenated = array_concatenated[512:]
                    audio = AudioSegment(chunk, sample_width=2, frame_rate=16000, channels=1)
                    audio_append = audio_append + audio

                    logging.info("Speech prob {}".format(speech_prob))
                    logging.info("launch_processing {}".format(launch_processing))

                
                if launch_processing:
                    audio_file = "appended_output.wav"
                    audio_append.export(audio_file, format="wav")
                    logging.info("Transcribing audio...")
                    text_transcribed = transcribe_audio(audio_file)
                    logging.info("text transcribed {}".format(text_transcribed))
                    conversation_manager.add_user_message(session_id, text_transcribed)
                    conversation = conversation_manager.get_conversation(session_id)
                    llm_sentence=""
                    llm_complete_response = ""
                    async for llm_chunk in stream_llm_response(conversation):

                        if llm_chunk!="[DONE]":
                            llm_sentence += llm_chunk
                            llm_complete_response += llm_chunk

                        is_text_remanent = (llm_sentence != "" and llm_chunk == "[DONE]")

                        if llm_sentence.endswith(('.', '!', '?', ',')) or is_text_remanent:
                            logging.info("Sending chunk to TTS: {}".format(llm_sentence))
                            tts_response = await generate_tts_response(llm_sentence)
                            # logging.info("tts response bytes: {}".format(len(tts_response)))
                            llm_sentence = ""
                            response_data = {
                                'event': 'media',
                                'streamSid': streamSid,
                                'media': {'payload': tts_response}
                            }
                            await websocket.send_json(response_data)
                            leo_is_speaking = True
                        
                        if llm_chunk=="[DONE]":
                            conversation_manager.add_ai_message(session_id, llm_complete_response)
                            audio_append = AudioSegment.empty()
                            mark_message = { 
                                "event": "mark",
                                "streamSid": streamSid,
                                "mark": {
                                    "name": "leo_stopped_talking"
                                    }   
                                }
                            await websocket.send_json(mark_message)

            if data["event"] == "mark":
                leo_is_speaking = False
                logging.info("mark message {}".format(data))

    except WebSocketDisconnect:
        logging.info("Connection closed.")

@app.post(OUTBOUND_CALL_ROUTE)
async def make_outbound_call(request: dict):
    global NGROK_URL
    to_number = request.get('to')

    if not to_number:
        return JSONResponse({"error": "Recipient phone number is required"}, status_code=400)

    call = client.calls.create(
        url=NGROK_URL,
        to=to_number,
        from_=TWILIO_NUMBER,
    )
    return JSONResponse({"message": "Call initiated", "call_sid": call.sid})

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.DEBUG)
        print("Server listening on: http://localhost:" + str(HTTP_SERVER_PORT))
        listener = ngrok.forward(f"http://localhost:{HTTP_SERVER_PORT}")
        print(f"Ngrok tunnel opened at {listener.url()} for port {HTTP_SERVER_PORT}")
        NGROK_URL = listener.url()

        twilio_numbers = client.incoming_phone_numbers.list()
        twilio_number_sid = [num.sid for num in twilio_numbers if num.phone_number == TWILIO_NUMBER][0]
        client.incoming_phone_numbers(twilio_number_sid).update(account_sid, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=HTTP_SERVER_PORT)

    finally:
        ngrok.disconnect()
