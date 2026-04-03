from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import sensor, upload, history, generate, model , train
from app.services.ml_service import load_model





@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield



app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sensor.router)
app.include_router(upload.router)
app.include_router(history.router)
app.include_router(generate.router)
app.include_router(model.router)
app.include_router(train.router)
