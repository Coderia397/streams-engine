"""Starts the Streams engine inside a Hugging Face Gradio Space and forwards requests to it.

The Space only lets requests through under /gradio_api/, so the engine is served under /gradio_api/engine/<its own path>.
The engine binary sits next to this file (uploaded by the deploy workflow); its data comes from a private dataset repo and is
downloaded when the Space starts. Settings come from Space variables and secrets: DATA_REPO, HF_TOKEN, SUPABASE_URL,
SUPABASE_PUBLISHABLE_KEY."""
import os, pathlib, subprocess, threading, time

import gradio as gr
import httpx
import spaces
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path("/tmp/streams-data")
PORT = 8088
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate", "host", "content-length", "content-encoding"}
state = {"stage": "starting", "restarts": 0}


@spaces.GPU
def noop(text):
    # ZeroGPU only starts a Space that has a GPU function; this one is never called by the engine
    return text


def download():
    from huggingface_hub import snapshot_download
    state["stage"] = "downloading data"
    snapshot_download(repo_id=os.environ["DATA_REPO"], repo_type="dataset", local_dir=DATA, token=os.environ.get("HF_TOKEN"))


def refresh_fresh():
    """fresh.json (new, upcoming and trending titles) is rebuilt every six hours in the dataset; fetch it again every two hours, the
    engine re-reads the file when it changes."""
    from huggingface_hub import hf_hub_download
    while True:
        time.sleep(7200)
        try:
            hf_hub_download(repo_id=os.environ["DATA_REPO"], repo_type="dataset", filename="fresh.json", local_dir=DATA, token=os.environ.get("HF_TOKEN"), force_download=True)
        except Exception as e:
            print("fresh.json refresh failed:", e, flush=True)


def run_engine():
    binary = HERE / "streams-engine"
    binary.chmod(0o755)
    env = {**os.environ, "CATALOG_DIR": str(DATA), "PORT": str(PORT), "HNSW_TRUST_INDEX": "1", "RUST_LOG": os.environ.get("RUST_LOG", "info")}
    while True:
        state["stage"] = "engine running"
        code = subprocess.call([str(binary)], env=env, cwd=str(HERE))
        state["stage"] = f"engine stopped ({code}), restarting"
        state["restarts"] += 1
        time.sleep(3)


def boot():
    try:
        download()
        threading.Thread(target=refresh_fresh, daemon=True).start()
        run_engine()
    except Exception as e:  # shown by /gradio_api/engine-status; the log keeps the trace
        state["stage"] = f"failed: {e}"
        raise


client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=60)


async def forward(request: Request, path: str):
    if not state["stage"].startswith("engine running"):
        return JSONResponse({"status": "starting", "stage": state["stage"]}, status_code=503, headers={"retry-after": "5", "access-control-allow-origin": "*"})
    try:
        upstream = await client.send(
            client.build_request(request.method, "/" + path, params=request.query_params, headers={k: v for k, v in request.headers.items() if k.lower() not in HOP}, content=await request.body()),
            stream=True)
    except httpx.TransportError:
        return JSONResponse({"status": "starting", "stage": "engine not answering yet"}, status_code=503, headers={"retry-after": "5", "access-control-allow-origin": "*"})
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP}
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=headers, background=None)


def status():
    return JSONResponse(state)


demo = gr.Interface(noop, "text", "text")
app, _, _ = demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
app.router.routes.insert(0, APIRoute("/gradio_api/engine/{path:path}", forward, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]))
app.router.routes.insert(0, APIRoute("/gradio_api/engine-status", status, methods=["GET"]))
threading.Thread(target=boot, daemon=True).start()
while True:
    time.sleep(3600)
