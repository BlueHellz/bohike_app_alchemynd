import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.api.ws import ws_router
from app.ai.session_designer import warm_up
import logging
import os
from dotenv import load_dotenv

# Ensure the project root is in sys.path (so `uvicorn app.main:app` works)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load .env from project root
load_dotenv(Path(_project_root) / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await warm_up()
    logging.info("NeuroSync Engine ready.")
    yield
    logging.info("NeuroSync Engine shutting down.")


app = FastAPI(title="NeuroSync Engine", version="3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
app.include_router(ws_router)
