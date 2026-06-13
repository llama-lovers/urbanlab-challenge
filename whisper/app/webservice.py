from app.tools.transcribe_tools import Model

from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from pathlib import Path
import shutil
import torch
from gc import collect
import os
from app.tools.locks import gpu_lock

base_dir = Path(".")
model = Model(os.getenv("WHISPER_MODEL"), "nvidia/diar_streaming_sortformer_4spk-v2.1")
model.load_models()
app = FastAPI()

def del_garbage():
    for item in base_dir.iterdir():
        if item.name in {"app", "req.txt", "models"}:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

@app.post("/asr")
async def asr(response: Response, audio_file: UploadFile = File(...), use_context: bool = False):
    try:
        save_path = Path(audio_file.filename)

        with open(save_path, "wb") as f:
            while chunk := await audio_file.read(1024 * 1024):  # 1MB
                f.write(chunk)

        async with gpu_lock:
            result = model.transribe(save_path, use_context)
        response.headers["asr-engine"] = model.engine_name
        response.headers["asr-engine-version"] = model.engine_version
        response.headers["model"] = model.model_transcribe_name
        
        del_garbage()
    
    except Exception as e:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            collect()
        except Exception:
            pass
        del_garbage()
        
        raise  HTTPException(status_code=500, detail=f"ASR failed: {e}")

    return result
    