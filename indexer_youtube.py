import os
import whisper
import faiss
import pickle
import numpy as np
import csv
import traceback
#from pytube import YouTube
#from moviepy.editor import VideoFileClip
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

VIDEO_DIR = "video"
AUDIO_DIR = "audio"
CLIP_DIR = "ui/clips"
CHUNK_SECONDS = 20
INDEX_FILE = "vector_store.faiss"
METADATA_FILE = "metadata.json"
fields = ["text", "video_path"]
CSV_FILE = "text_video_path.csv"
ROOT = Path(__file__).parent.resolve()
METADATA_FILE = ROOT / "metadata.json"
metadata_content = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []

model = whisper.load_model("base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_file_name_from_url(url:str):
    filename = url[url.rfind("/") + 1:]
    if "watch?v=" in filename:
        filename = filename[filename.index("watch?v=") + len("watch?v="): ]
    return filename

def download_youtube_video(url: str, filename:str) -> str:  
    os.makedirs(VIDEO_DIR, exist_ok=True)
    #yt = YouTube(url)
    yt = YouTube(url, on_progress_callback=on_progress)
    #stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
    
    #stream.download(output_path=VIDEO_DIR, filename=filename)
    
    #ys = yt.streams.get_highest_resolution()

    
    #print(yt.title)

    ys = yt.streams.get_highest_resolution()
    #ys.download()
    file_path = filename + ".mp4"
    output_path = os.path.join(VIDEO_DIR, file_path)
    ys.download(output_path=VIDEO_DIR, filename=file_path)
    return output_path

def transcribe_and_chunk(video_path:str, filename:str, url:str):
    video = VideoFileClip(video_path)
    duration = int(video.duration)
    os.makedirs(CLIP_DIR, exist_ok=True)

    texts, embeddings, metadata = [], [], []
    for start in range(0, duration, CHUNK_SECONDS):
        end = min(start + CHUNK_SECONDS, duration)
        #clip = video.subclip(start, end)
        clip = video.subclipped(start, end)
        clip_path = f"{CLIP_DIR}/{filename}_{start}_{end}.mp4"
        #clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
        audio = clip.audio
        os.makedirs(AUDIO_DIR, exist_ok=True)
        audio_path = f"{AUDIO_DIR}/{filename}.wav"
        #audio.write_audiofile(audio_path, verbose=False, logger=None)
        audio.write_audiofile(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = model.transcribe(audio_path)
        text = result["text"]
        emb = embedder.encode([text])[0]

        texts.append(text)
        embeddings.append(emb)
        metadata.append({"start": start, "end": end, "clip_path": clip_path, "url": url})

        os.remove(audio_path)

    return np.array(embeddings).astype("float32"), texts, metadata

def build_index_from_youtube(url: str):
    try:
        print(f"build_index_from_youtube: begin: {metadata_content} ")
        filename = get_file_name_from_url(url)
        video_path = download_youtube_video(url, filename)
        new_embeddings, new_texts, new_metadata = transcribe_and_chunk(video_path, filename, url)
        index = faiss.read_index(INDEX_FILE) if os.path.exists(INDEX_FILE) else None
        if index is None:
            index = faiss.IndexFlatL2(new_embeddings.shape[1])
        index.add(new_embeddings)
        faiss.write_index(index, INDEX_FILE)
        #texts, metadata = [], []
        metadata_list = []

        #if os.path.exists(METADATA_FILE):
        #    with open(METADATA_FILE, 'rb') as f:
        #        texts, metadata = pickle.load(f)

        #texts.extend(new_texts)
        #metadata.extend(new_metadata) 
        #new_clip_path_list = [nmd_record['clip_path'] for nmd_record in new_metadata if 'clip_path' in nmd_record and nmd_record['clip_path'] is not None ]       

        #with open(METADATA_FILE, "a+b") as f:
        #    pickle.dump((texts, metadata), f)
        
        for record in zip(new_texts, new_metadata):
            obj = {}
            obj['clip_path'] = record[1]['clip_path']
            obj['text'] = record[0]
            metadata_list.append(obj)

        print(f"build_index_from_youtube: metadata_list: {metadata_list} ")
            
        metadata_content.extend(metadata_list)
        print(f"build_index_from_youtube: before json dump: {metadata_content} ")
        METADATA_FILE.write_text(json.dumps(metadata_content, indent=2))
            
    
        csv_file_already_exists = False
        if os.path.exists(CSV_FILE):
            csv_file_already_exists = True            
            
        
        with open(CSV_FILE, 'a+', newline='') as csvfile:
        # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            if not csv_file_already_exists:
               csvwriter.writerow(fields)
            for record in zip(new_texts, new_metadata):
                if record[0].strip():
                    csvwriter.writerow([record[0], record[1]['clip_path']]) 

        return True
    except:
        traceback.print_exc()
        return False

    #print("Index built and stored.")

if __name__ == "__main__":
    #test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your own
    test_url = "https://www.youtube.com/watch?v=-WaSBjK0qwk"
    build_index_from_youtube(test_url)

