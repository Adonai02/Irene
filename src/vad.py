import torch
import numpy as np 


class VADModel:
  def __init__(self, time_until_stop_talking=1.5, speech_prob_stop_talking=0.7, sample_rate=16_000) -> None:
    
    self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    # time_per_chunk = 0.064 # para un array de 1024 
    time_per_chunk = 0.032 # para un array de 512 
    time_per_chunk = 0.045 # para un array de 512 
    self.chunks_stop_talking = 30 #round(time_until_stop_talking/time_per_chunk)
    self.speech_prob_talking_array = np.array([])
    self.speech_prob_stop_talking = speech_prob_stop_talking
    self.total_chunks_to_take_in_account = 30
    self.sample_rate = sample_rate
    self.user_is_talking = False

  # def get_speech_prob(self, audio_array):

  def get_chunks_to_iterate(self,audio_array):
      num_samples = 512 if self.sample_rate == 16000 else 256
      chunks_to_iterate = audio_array.shape[0] // num_samples
      return chunks_to_iterate, num_samples
  
      
  @torch.no_grad()
  def is_end_of_speaking(self, audio_array):

      is_end_of_speaking = False
      inputs = audio_array
      tensor = torch.from_numpy(inputs)
      speech_prob = self.model(tensor, self.sample_rate).item()

      self.speech_prob_talking_array = np.append(self.speech_prob_talking_array,speech_prob)

      if not self.user_is_talking:
        self.user_is_talking = speech_prob >= self.speech_prob_stop_talking

      # wait until there is enough chunks
      if len(self.speech_prob_talking_array) >= self.chunks_stop_talking and self.user_is_talking:
          # last X chunks has a prob lower than a tresh, then stop talking
          if all(self.speech_prob_talking_array[-self.chunks_stop_talking:] < self.speech_prob_stop_talking):
                is_end_of_speaking = True
                self.speech_prob_talking_array = np.array([])
                self.user_is_talking = False

                # if self.onnx_vad:
                #   # Reset states of the model VAD
                #   # #print("reseting states vad")
                #   self.model_vad.reset_states()
      


      return is_end_of_speaking, speech_prob
      
