from flask import Flask, request, jsonify, render_template_string, Response
from TTS.api import TTS
import librosa
import numpy as np
import io
import time
import re
import soundfile as sf
import torch

app = Flask(__name__)

print('Loading VITS...')
t0 = time.time()
vits_model = 'tts_models/es/css10/vits'

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

tts_vits = TTS(vits_model).to(device)
elapsed = time.time() - t0
print(f"Loaded in {elapsed:.2f}s")

@app.route("/", methods=["GET"])
def get_form():
    return render_template_string("""
    <html>
        <body>
            <style>
            textarea, input { display: block; width: 100%; border: 1px solid #999; margin: 10px 0px }
            textarea { height: 25%; }
            </style>
            <h2>TTS VITS</h2>
            <form method="post" action="/tts">
                <textarea name="text">This is a test.</textarea>
                <input name="speaker" value="p273" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """)

@app.route("/tts", methods=["POST"])
def text_to_speech():
    text = request.form.get("text", "").strip()
    # speaker = request.form.get("speaker", "p273")

    # Text preprocessing
    text = re.sub(r'~+', '!', text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"(\*[^*]+\*)|(_[^_]+_)", "", text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    try:
        t0 = time.time()
        wav_np = tts_vits.tts(text)
        generation_time = time.time() - t0

        audio_duration = len(wav_np) / 22050  # Assuming 22050 Hz sample rate
        rtf = generation_time / audio_duration
        print(f"Generated in {generation_time:.2f}s")
        print(f"Real-Time Factor (RTF): {rtf:.2f}")

        wav_np = np.array(wav_np)
        wav_np = np.clip(wav_np, -1, 1)

        # Convert to WAV using an in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, wav_np, 22050, format='wav')  # Save as WAV format
        buffer.seek(0)

        return Response(buffer, mimetype="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003)
