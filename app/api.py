from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.graph import sales_graph

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="AI Sales Agent",
    description="Agentic AI sales assistant with RAG and sales tools",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "demo-user-1"


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "AI Sales Agent API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    result = sales_graph.invoke(
        {
            "customer_message": request.message
        },
        config=config
    )

    return {
        "answer": result.get(
            "answer",
            "I couldn't process that request."
        )
    }


if FRONTEND_DIR.is_dir():
    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def sales_ui():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")