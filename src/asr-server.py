import soundfile as sf
import tempfile
import torch
from flask import Flask, request, jsonify
from loguru import logger
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from urllib.parse import unquote
from pydub import AudioSegment
import numpy as np
import io

app = Flask(__name__)

class TranscriptionEngine:
    def transcribe(self, file, audio_content, **kwargs):
        raise NotImplementedError

class FasterWhisperEngine(TranscriptionEngine):
    def __init__(self):
        from faster_whisper import WhisperModel

        # not know
        model_id = "large-v3"
        # 350ms
        # model_id = "large-v2"
        # # 300ms
        # model_id = "distil-large-v2"
        # # 280ms
        # model_id = "distil-medium.en"
        # # 300ms
        # model_id = "distil-large-v3"
        
        self.model = WhisperModel(model_id, device="cuda", compute_type="float16")

    def transcribe(self, filename, audio_content, **kwargs):
        language = kwargs.get("language", "es")
        segments, _ = self.model.transcribe(io.BytesIO(audio_content), beam_size=5, language=language)

        full_text = "".join(segment.text for segment in segments)
        logger.info(full_text)

        return full_text, [{"start": s.start, "end": s.end, "text": s.text} for s in segments]



class TransformersEngine(TranscriptionEngine):
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        model_id = "distil-whisper/large-v2"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, file_, audio_content, **kwargs):
        result = self.pipe(audio_content, **kwargs)
        return result["text"], result.get("chunks", [])

# engine = TransformersEngine()
# logger.info("Using TransformersEngine")

engine = FasterWhisperEngine()
logger.info("Using FasterWhisperEngine")

@app.route("/v1/audio/transcriptions", methods=["POST"])
def create_transcription():
    file = request.files['file']
    # model = request.form.get('model')
    language = request.form.get('language')
    prompt = request.form.get('prompt')
    response_format = request.form.get('response_format', 'json')
    temperature = float(request.form.get('temperature', 0.0))

    audio_content = file.read()
    text, _ = engine.transcribe(audio_content, generate_kwargs={"language": language, "task": "transcribe"})
    response = {"text": text}
    
    return jsonify(response)

@app.route("/v1/audio/translations", methods=["POST"])
def create_translation():
    file = request.files['file']
    # model = request.form.get('model')
    prompt = request.form.get('prompt')
    response_format = request.form.get('response_format', 'json')
    temperature = float(request.form.get('temperature', 0.0))

    audio_content = file.read()
    text, _ = engine.transcribe(file, audio_content, generate_kwargs={"task": "translate"})
    response = {"text": text}
    
    return jsonify(response)

@app.route("/inference", methods=["POST"])
def inference():
    file_ = request.files['file']
    temperature = float(request.form.get('temperature', 0.0))
    temperature_inc = float(request.form.get('temperature_inc', 0.0))
    response_format = request.form.get('response_format', 'json')

    audio_content = file_.read()
    # audio_segment = AudioSegment.from_file(io.BytesIO(audio_content))
    # audio_segment = audio_segment.set_channels(1)

    # # Convert to numpy array
    # audio_data = np.array(audio_segment.get_array_of_samples())

    # # If stereo, reshape the array
    # if audio_segment.channels == 2:
    #     audio_data = audio_data.reshape((-1, 2))
    # temperature += temperature_inc


    # print("file name", file_.name)
    # logger.info(f"filename {file_}")
    text, segments = engine.transcribe(
        file_, 
        audio_content,
        generate_kwargs={
            "temperature": temperature,
            "do_sample": True,
            'language': 'es'
        }
    )
    
    response = {"text": text}
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
