from fastapi import APIRouter
from app.schemas import DesignRequest,DesignResponse,SessionBlueprint,RatingRequest
from app.ai.architect import design_session
from app.session.controller import create_session,pause_session,resume_session,stop_session,SESSIONS
from app.safety.validator import safety_summary
from app.profile.store import save_session

router=APIRouter()

@router.post("/session/design",response_model=DesignResponse)
async def design(req:DesignRequest):
    return await design_session(req)

@router.post("/session/start")
async def start(bp:SessionBlueprint):
    sid=await create_session(bp)
    return {"session_id":sid,"summary":safety_summary(bp)}

@router.post("/session/{sid}/pause")
def pause(sid:str): pause_session(sid);return {"status":"paused"}
@router.post("/session/{sid}/resume")
def resume(sid:str): resume_session(sid);return {"status":"running"}
@router.post("/session/{sid}/stop")
def stop(sid:str): stop_session(sid);return {"status":"stopped"}
@router.get("/session/{sid}/status")
def status(sid:str): s=SESSIONS.get(sid,{});return {"state":s.get("state"),"t":s.get("t")}
@router.post("/session/{sid}/rate")
async def rate(sid:str,req:RatingRequest):
    s=SESSIONS.get(sid)
    if s: await save_session(s["blueprint"].user_id,sid,s["blueprint"],req.rating,req.notes or "")
    return {"status":"saved"}
