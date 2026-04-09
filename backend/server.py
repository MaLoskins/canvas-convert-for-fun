"""
Live Canvas Art - Backend Server

Multi-model live sketch-to-image pipeline with WebSocket API.
Supports T2I-Adapter, ControlNet, and ControlNet Union model families.
"""

import asyncio
import base64
import io
import logging
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from models import MODEL_REGISTRY, ModelManager, GenRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREVIEW_STEPS = 12
PREVIEW_SIZE = 768
HQ_STEPS = 30
HQ_SIZE = 1024
DEFAULT_MODEL = "t2i_sketch"

# ── Global model manager ────────────────────────────────────────────
manager = ModelManager()


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── App lifecycle ────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()

    async def _load_default():
        await loop.run_in_executor(None, manager.load_model, DEFAULT_MODEL)

    asyncio.create_task(_load_default())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ──────────────────────────────────────────────────
@app.get("/status")
async def status():
    return manager.get_status()


def _serialize_model(info):
    entry = {
        "key": info.key,
        "name": info.name,
        "family": info.family,
        "description": info.description,
        "default_conditioning_scale": info.default_conditioning_scale,
        "quality": info.quality,
        "speed": info.speed,
        "adherence": info.adherence,
        "size": info.size,
        "input_type": info.input_type,
    }
    if info.union_modes:
        entry["union_modes"] = list(info.union_modes.keys())
    return entry


@app.get("/models")
async def list_models():
    result = [_serialize_model(info) for info in MODEL_REGISTRY.values()]
    return {"models": result, "active": manager.active_key}


# ── WebSocket live generation ───────────────────────────────────────
#
# Protocol (request-reply, no queue):
#   Client connects → server sends status
#   When ready → server sends {"type":"ready_for_next"}
#   Client sends {"type":"generate", model, prompt, image, ...}
#   Server sends {"type":"generating"}, runs inference, sends result
#   Server sends {"type":"ready_for_next"} → repeat
#
#   Client sends {"type":"switch_model", "model":"..."} to change model.
#   Server unloads old, loads new, sends status updates, then ready_for_next.
# ────────────────────────────────────────────────────────────────────
@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket client connected")

    loop = asyncio.get_event_loop()
    generation_id = 0

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "generate")

            # ── Ping ──────────────────────────────────────────
            if msg_type == "ping":
                await ws.send_json({"type": "pong"})
                continue

            # ── Status query ──────────────────────────────────
            if msg_type == "status":
                s = manager.get_status()
                await ws.send_json({"type": "status", **s})
                if manager.is_ready:
                    await ws.send_json({"type": "ready_for_next"})
                continue

            # ── Model list ────────────────────────────────────
            if msg_type == "list_models":
                await ws.send_json({
                    "type": "model_list",
                    "models": [_serialize_model(info) for info in MODEL_REGISTRY.values()],
                    "active": manager.active_key,
                })
                continue

            # ── Switch model ──────────────────────────────────
            if msg_type == "switch_model":
                requested = data.get("model", "")
                if requested not in MODEL_REGISTRY:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown model: {requested}",
                    })
                    continue

                if requested == manager.active_key and manager.is_ready:
                    await ws.send_json({"type": "status", "status": "ready", "model": requested})
                    await ws.send_json({"type": "ready_for_next"})
                    continue

                await ws.send_json({
                    "type": "status",
                    "status": "loading",
                    "message": f"Loading {MODEL_REGISTRY[requested].name}...",
                    "model": requested,
                })

                await loop.run_in_executor(None, manager.load_model, requested)

                s = manager.get_status()
                await ws.send_json({"type": "status", **s})
                if manager.is_ready:
                    await ws.send_json({"type": "ready_for_next"})
                continue

            # ── Generate ──────────────────────────────────────
            if msg_type != "generate":
                continue

            prompt = data.get("prompt", "")
            image_b64 = data.get("image", "")
            negative_prompt = data.get("negative_prompt", "")
            steps = int(data.get("steps", PREVIEW_STEPS))
            guidance = float(data.get("guidance", 7.5))
            conditioning_scale = float(data.get("adapter_scale", 0.9))
            seed = data.get("seed", None)
            if seed is not None:
                seed = int(seed)
            size = int(data.get("size", PREVIEW_SIZE))
            union_mode = data.get("union_mode", None)

            if not image_b64:
                await ws.send_json({"type": "ready_for_next"})
                continue

            if not manager.is_ready:
                s = manager.get_status()
                await ws.send_json({"type": "status", **s})
                continue

            generation_id += 1
            gid = generation_id
            image_bytes = base64.b64decode(image_b64)

            await ws.send_json({"type": "generating", "generation_id": gid})

            req = GenRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_bytes=image_bytes,
                steps=steps,
                guidance=guidance,
                conditioning_scale=conditioning_scale,
                seed=seed,
                size=size,
                union_mode=union_mode,
            )

            try:
                t0 = time.perf_counter()
                result_img = await loop.run_in_executor(None, manager.generate, req)
                elapsed = time.perf_counter() - t0

                if result_img is not None:
                    result_b64 = image_to_base64(result_img)
                    await ws.send_json({
                        "type": "result",
                        "image": result_b64,
                        "generation_id": gid,
                        "elapsed": round(elapsed, 2),
                    })
                else:
                    await ws.send_json({
                        "type": "skipped",
                        "generation_id": gid,
                        "reason": "blank_sketch",
                    })
            except Exception as e:
                logger.exception("Generation %d failed", gid)
                await ws.send_json({
                    "type": "error",
                    "message": str(e),
                    "generation_id": gid,
                })

            await ws.send_json({"type": "ready_for_next"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8188, log_level="info")
