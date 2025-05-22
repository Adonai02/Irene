from prompts import SYSTEM_PROMPT
from collections import deque
import uuid
import time

class ConversationManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout for sessions

    def create_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "conversation": [{"role": "system", "content": SYSTEM_PROMPT}],
            "llm_output_sentences": deque(),
            "current_turn": 0,
            "is_processing": False,
            "audio_buffer": b'',  # New: Buffer to accumulate audio data
            "last_activity": time.time(),
            "first_audio_sent": False,
            "latency_metrics": {
                "start_time": 0,
                "srt_start": 0,
                "srt_end": 0,
                "llm_start": 0,
                "llm_first_token": 0,
                "llm_first_sentence": 0,
                "tts_start": 0,
                "tts_end": 0,
                "first_audio_response": 0,
            }
        }
        return session_id

    def reset_latency_metrics(self, session_id):
        self.sessions[session_id]["latency_metrics"] = {
            "start_time": time.time(),
            "srt_start": 0,
            "srt_end": 0,
            "llm_start": 0,
            "llm_first_token": 0,
            "llm_first_sentence": 0,
            "tts_start": 0,
            "tts_end": 0,
            "first_audio_response": 0,
        }

    def update_latency_metric(self, session_id, metric, value):
        self.sessions[session_id]["latency_metrics"][metric] = value

    def calculate_latencies(self, session_id):
        metrics = self.sessions[session_id]["latency_metrics"]
        start_time = metrics["start_time"]
        
        return {
            "total_voice_to_voice": metrics["first_audio_response"] - start_time,
            "srt_duration": metrics["srt_end"] - metrics["srt_start"],
            "llm_ttft": metrics["llm_first_token"] - metrics["llm_start"],
            "llm_ttfs": metrics["llm_first_sentence"] - metrics["llm_start"],
            "tts_duration": metrics["tts_end"] - metrics["tts_start"],
        }

    def add_user_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "user", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def add_ai_message(self, session_id, message):
        self.sessions[session_id]["conversation"].append({"role": "assistant", "content": message})
        self.sessions[session_id]["current_turn"] += 1
        self.sessions[session_id]["last_activity"] = time.time()

    def get_conversation(self, session_id):
        return self.sessions[session_id]["conversation"]

    def clean_old_sessions(self):
        current_time = time.time()
        sessions_to_remove = [
            session_id for session_id, session_data in self.sessions.items()
            if current_time - session_data["last_activity"] > self.session_timeout
        ]
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def add_to_audio_buffer(self, session_id, audio_data):
        self.sessions[session_id]["audio_buffer"] += audio_data

    def get_and_clear_audio_buffer(self, session_id):
        audio_data = self.sessions[session_id]["audio_buffer"]
        self.sessions[session_id]["audio_buffer"] = b''
        return audio_data