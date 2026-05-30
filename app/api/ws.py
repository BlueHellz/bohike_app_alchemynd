from fastapi import APIRouter,WebSocket,WebSocketDisconnect
from app.session.controller import SESSIONS,run_session
import asyncio
ws_router=APIRouter()

@ws_router.websocket("/ws/audio/{sid}")
async def audio_stream(ws:WebSocket,sid:str):
    await ws.accept();s=SESSIONS.get(sid)
    if not s: await ws.close(code=4004);return
    task=asyncio.create_task(run_session(sid))
    try:
        while True:
            chunk=await asyncio.wait_for(s["audio_q"].get(),timeout=5.0)
            await ws.send_bytes(chunk)
            if s["state"] in ("completed","stopped"): break
    except(WebSocketDisconnect,asyncio.TimeoutError):
        from app.session.controller import stop_session;stop_session(sid)
    finally: task.cancel()

@ws_router.websocket("/ws/light/{sid}")
async def light_stream(ws:WebSocket,sid:str):
    await ws.accept();s=SESSIONS.get(sid)
    if not s: await ws.close(code=4004);return
    try:
        while True:
            data=await asyncio.wait_for(s["light_q"].get(),timeout=5.0)
            await ws.send_json(data)
            if s["state"] in ("completed","stopped"): break
    except(WebSocketDisconnect,asyncio.TimeoutError): pass
