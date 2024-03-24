import base64
import wave
import urllib.request
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment


class InferlessPythonModel:
    def download_file(self,url):
        filename = "file.mp3"
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"File {filename} downloaded successfully.")
        except Exception as e:
            print(f"Failed to download the file. Error: {e}")
        
        return filename
    
    def wav_to_base64(self,file_path):
        with open(file_path, 'rb') as wav_file:
            # Read the content of the WAV file
            wav_content = wav_file.read()
    
            # Encode the content as base64
            base64_encoded = base64.b64encode(wav_content).decode('utf-8')
    
        return base64_encoded
        
    def initialize(self):
        self.pipeline = Pipeline.from_pretrained(
          "pyannote/speaker-diarization-3.1",
          use_auth_token="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")
        
        self.pipeline.to(torch.device("cuda"))

        
    def infer(self, inputs):
        audio_url = inputs["audio_url"]
        file_name = self.download_file(audio_url)
        diarization = self.pipeline(file_name)

        audio = AudioSegment.from_file(file_name,format = "mp3")
        speaker_segments_audio = {}
        
        audio_data = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)  
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]
        
            if speaker in speaker_segments_audio:
                speaker_segments_audio[speaker] += segment
            else:
                speaker_segments_audio[speaker] = segment
        
        for speaker, segment in speaker_segments_audio.items():
            segment.export(f"{speaker}.wav", format="wav")
            base64_data = self.wav_to_base64(f"{speaker}.wav")
            audio_data.append(base64_data)
            
        return {"generated_data":audio_data}


    def finalize(self):
        pass