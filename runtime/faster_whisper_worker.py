"""Line-delimited JSON worker for VoxBridge's Faster Whisper backend."""

import json
import os
import sys
import traceback

# NVIDIA's Windows wheels keep runtime DLLs in package-specific bin directories.
# Register them before importing CTranslate2 so the isolated VoxBridge environment
# can use CUDA without modifying the user's system PATH.
if os.name == "nt":
    runtime_root = os.path.dirname(os.path.abspath(__file__))
    dll_directory_handles = []
    for relative in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
        candidate = os.path.join(runtime_root, ".venv", "Lib", "site-packages", relative)
        if os.path.isdir(candidate):
            os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
            # Keep each handle alive for the lifetime of the worker. Dropping it
            # removes the directory from Windows' DLL search path.
            dll_directory_handles.append(os.add_dll_directory(candidate))

import numpy as np
from faster_whisper import WhisperModel


model = None


def respond(request_id, **payload):
    print(json.dumps({"id": request_id, **payload}), flush=True)


def handle(request):
    global model
    command = request["command"]

    if command == "load":
        model = WhisperModel(
            request["model"],
            device=request.get("device", "auto"),
            compute_type=request.get("compute_type", "auto"),
            download_root=request["model_cache_dir"],
            local_files_only=False,
        )
        respond(
            request["id"],
            ok=True,
            device=model.model.device,
            compute_type=model.model.compute_type,
        )
        return

    if command == "transcribe":
        if model is None:
            raise RuntimeError("Faster Whisper model is not loaded")
        samples = np.frombuffer(
            bytes.fromhex(request["samples_f32_le_hex"]), dtype="<f4"
        ).copy()
        segments, _ = model.transcribe(
            samples,
            language=request.get("language") or None,
            initial_prompt=request.get("prompt") or None,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = "".join(segment.text for segment in segments).strip()
        respond(request["id"], ok=True, text=text)
        return

    raise ValueError(f"Unknown command: {command}")


for line in sys.stdin:
    try:
        message = json.loads(line)
        handle(message)
    except Exception as error:
        request_id = message.get("id", 0) if "message" in locals() else 0
        traceback.print_exc(file=sys.stderr)
        respond(request_id, ok=False, error=str(error))
