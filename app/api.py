import os
from pathlib import Path

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

from fastapi import FastAPI, Request
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
    intent: str = ""
    lead: dict = {}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        result = sales_graph.invoke(
            {"customer_message": request.message},
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "answer": f"Something went wrong: {exc}",
            "intent": "error",
            "lead": {},
        }
    return {
        "answer": result.get("answer", "I couldn't process that request."),
        "intent": result.get("intent", ""),
        "lead": {
            "name": result.get("name", ""),
            "email": result.get("email", ""),
            "company": result.get("company", ""),
            "use_case": result.get("use_case", ""),
            "budget": result.get("budget", ""),
            "preferred_time": result.get("preferred_time", ""),
            "site_url": result.get("site_url", ""),
        },
    }


if FRONTEND_DIR.is_dir():

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/styles.css", include_in_schema=False)
    def styles():
        return FileResponse(FRONTEND_DIR / "styles.css")

    @app.get("/app.js", include_in_schema=False)
    def script():
        return FileResponse(FRONTEND_DIR / "app.js")

    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
