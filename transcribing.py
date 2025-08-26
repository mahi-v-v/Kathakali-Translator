import torch
import whisper
import time
from pydub import AudioSegment
import os

# --- Step 1: Transcribe the Entire Audio File ---
def transcribe_audio_segmented(audio_file_path):
    """
    Transcribes an entire audio file by manually segmenting it.
    """
    start_load_time = time.time()
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We will use the medium model based on our analysis
    model = whisper.load_model("medium").to(device)
    load_duration = time.time() - start_load_time
    print(f"Model loaded successfully! ({load_duration:.2f} seconds)")
    
    # Read the initial prompt from a file
    try:
        with open("prompt.txt", "r", encoding="utf-8") as file:
            initial_prompt = file.read().replace('\n', ' ')
        print("Using initial prompt from the file given")
    except FileNotFoundError:
        print("Warning: prompt.txt not found. Using default prompt.")
        initial_prompt = "ajitha hare jaya madhava vishnu"

    print(f"\nTranscribing '{audio_file_path}'...")
    transcription_start_time = time.time()
    
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_file_path)
    audio_length_ms = len(audio)
    chunk_size_ms = 60 * 1000 
    
    transcribed_segments = []
    
    # Loop through the audio file and transcribe each chunk
    for i in range(0, audio_length_ms, chunk_size_ms):
        start_chunk_ms = i
        end_chunk_ms = min(i + chunk_size_ms, audio_length_ms)
        chunk = audio[start_chunk_ms:end_chunk_ms]
        
        # Save the chunk to a temporary .mp3 file for Whisper
        temp_chunk_path = f"temp_chunk_{i}.mp3"
        chunk.export(temp_chunk_path, format="mp3")

        print(f"Transcribing segment {start_chunk_ms/1000:.2f}s - {end_chunk_ms/1000:.2f}s...")
        
        result = model.transcribe(
            temp_chunk_path,
            fp16=torch.cuda.is_available(),
            language='sa',
            initial_prompt=initial_prompt
        )
        
        transcribed_segments.append(result["text"])
        os.remove(temp_chunk_path)
    
    transcription_duration = time.time() - transcription_start_time
    print(f"Transcription completed in {transcription_duration:.2f} seconds.")
    
    # Combine all transcribed segments into one long string
    full_transcription = " ".join(transcribed_segments)
    
    return full_transcription

# --- Main function to run the pipeline ---
if __name__ == "__main__":
    audio_file_path = "audio.mp3"
    
    try:
        start_total_time = time.time()
        
        transcribed_text = transcribe_audio_segmented(audio_file_path)
        print("\n--- Transcription Complete ---")
        print(f"Original Text: {transcribed_text}")
        
        total_duration = time.time() - start_total_time
        print(f"\nTotal script execution time: {total_duration:.2f} seconds.")

    except FileNotFoundError:
        print(f"Error: The audio file '{audio_file_path}' was not found.")
        print("Please place your audio file in the project folder and try again.")
    except Exception as e:
        print(f"An error occurred: {e}")