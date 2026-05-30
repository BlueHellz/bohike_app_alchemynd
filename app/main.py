import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.ws import ws_router
from app.ai.architect import warm_up
from dotenv import load_dotenv
import logging,os

_project_root=str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path: sys.path.insert(0,_project_root)
load_dotenv(Path(_project_root)/".env")
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))

@asynccontextmanager
async def lifespan(app:FastAPI):
    await warm_up();logging.info("NeuroSync Engine ready.");yield;logging.info("Shutdown.")

app=FastAPI(title="NeuroSync Engine",version="3.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
app.include_router(router,prefix="/api")
app.include_router(ws_router)
