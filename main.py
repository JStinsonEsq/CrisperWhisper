import os
import sys
import torch
import shutil
import uuid
import requests
import subprocess
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from datasets import load_dataset, Dataset, Audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# why you no import relative class file python?!
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from segmenter import AudioSegmenter
from huggingface_hub import login
HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)


app = FastAPI()

class TranscriptionRequest(BaseModel):
    audio_url: str
    callback_url: Optional[str] = None

@app.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):

    audio_url = request.audio_url
    callback_url = request.callback_url
    audio_id = str(uuid.uuid4())
    original_audio_path = f"audio/{audio_id}_original"
    flac_audio_path = f"audio/{audio_id}.flac"

    # Download the audio file
    try:
        r = requests.get(audio_url, stream=True)
        if r.status_code == 200:
            with open(original_audio_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            raise HTTPException(status_code=400, detail=f"Failed to download audio from URL. Status code: {r.status_code}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading audio: {e}")

    # Transcode audio to FLAC 16kHz using ffmpeg
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", original_audio_path,
            "-ac", "1", "-ar", "16000", flac_audio_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Audio transcoding failed: {e}")


    def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
        """
        Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
        """

        adjusted_chunks = pipeline_output["chunks"].copy()

        for i in range(len(adjusted_chunks) - 1):
            current_chunk = adjusted_chunks[i]
            next_chunk = adjusted_chunks[i + 1]

            current_start, current_end = current_chunk["timestamp"]
            next_start, next_end = next_chunk["timestamp"]
            pause_duration = next_start - current_end

            if pause_duration > 0:
                if pause_duration > split_threshold:
                    distribute = split_threshold / 2
                else:
                    distribute = pause_duration / 2

                # Adjust current chunk end time
                adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

                # Adjust next chunk start time
                adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
        pipeline_output["chunks"] = adjusted_chunks

        return pipeline_output


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "nyrahealth/CrisperWhisper"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=25,
        batch_size=16,
        return_timestamps='word',
        torch_dtype=torch_dtype,
        device=device,
    )


    segmenter = AudioSegmenter(max_segment_length=25.0, face_hugger_token=HF_TOKEN)
    segments = segmenter.segment_audio(flac_audio_path)

    trans_text = ""
    trans_chunks = []
    for sample in segments:
        hf_pipeline_output = pipe(sample)
        crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
        trans_text += ' ' + crisper_whisper_result.get('text')
        trans_chunks = trans_chunks + crisper_whisper_result.get('chunks')
        #print(crisper_whisper_result)

    os.remove(original_audio_path)
    os.remove(flac_audio_path)

    print(trans_text)
    print(trans_chunks)
    
    return {
        "transcript": trans_text,
        "timestamps": trans_chunks
    }
    


# Run server if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=360)
