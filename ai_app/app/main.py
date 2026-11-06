import os
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from .db import Base, engine, SessionLocal
from .models import Conversation, Memory, ResearchTopic
from .ai import answer, auto_extract_learnings, save_memory
from .file_ingest import extract_text, chunks

app = FastAPI(title="Self-Learning AI")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "dev-secret-change-me"), https_only=True)
app.mount("/static", StaticFiles(directory="ai_app/app/static"), name="static")
templates = Jinja2Templates(directory="ai_app/app/templates")
Base.metadata.create_all(bind=engine)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")
AUTO_LEARN_DEFAULT = os.getenv("AUTO_LEARN_DEFAULT", "false").lower() == "true"

def authed(request: Request):
    return request.session.get("auth") is True

def require_auth(request: Request):
    if not authed(request):
        raise HTTPException(401, "Login required")

@app.get("/health")
def health():
    return {"ok": True, "openai_configured": bool(os.getenv("OPENAI_API_KEY"))}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if not authed(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request":request, "auto_learn":AUTO_LEARN_DEFAULT})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request":request})

@app.post("/login")
def login(request: Request, password: str = Form(...)):
    if password != ADMIN_PASSWORD:
        return templates.TemplateResponse("login.html", {"request":request, "error":"Wrong password"}, status_code=401)
    request.session["auth"] = True
    return RedirectResponse("/", status_code=303)

@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

class ChatIn(BaseModel):
    message: str
    use_web: bool = True
    auto_learn: bool = False

@app.post("/api/chat")
def chat(payload: ChatIn, request: Request):
    require_auth(request)
    with SessionLocal() as db:
        try:
            text, sources = answer(db, payload.message, payload.use_web)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        db.add(Conversation(user_text=payload.message, assistant_text=text, used_web=payload.use_web))
        db.commit()
        learned = auto_extract_learnings(db, payload.message, text, sources) if payload.auto_learn and payload.use_web else 0
        return {"answer": text, "sources": sources, "learned": learned}

class MemoryIn(BaseModel):
    content: str

@app.get("/api/memories")
def memories(request: Request):
    require_auth(request)
    with SessionLocal() as db:
        rows=db.query(Memory).order_by(Memory.id.desc()).limit(200).all()
        return [{"id":m.id,"content":m.content,"source_type":m.source_type,"source_url":m.source_url,"confidence":m.confidence} for m in rows]

@app.post("/api/memories")
def add_memory(payload: MemoryIn, request: Request):
    require_auth(request)
    with SessionLocal() as db:
        try:
            m=save_memory(db,payload.content)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        return {"id":m.id}

@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id:int, request:Request):
    require_auth(request)
    with SessionLocal() as db:
        m=db.get(Memory,memory_id)
        if m:
            db.delete(m); db.commit()
    return {"ok":True}

@app.post("/api/files")
async def upload_file(request: Request, file: UploadFile = File(...)):
    require_auth(request)
    data=await file.read()
    text=extract_text(file.filename or "upload.txt", data)
    if not text.strip():
        raise HTTPException(400,"No readable text found")
    count=0
    with SessionLocal() as db:
        try:
            for ch in chunks(text):
                save_memory(db, f"File {file.filename}: {ch}", source_type="file", confidence=0.95)
                count+=1
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return {"ok":True,"chunks":count}

class TopicIn(BaseModel):
    topic:str

@app.get("/api/topics")
def topics(request:Request):
    require_auth(request)
    with SessionLocal() as db:
        rows=db.query(ResearchTopic).order_by(ResearchTopic.id.desc()).all()
        return [{"id":r.id,"topic":r.topic,"active":r.active,"last_run_at":r.last_run_at.isoformat() if r.last_run_at else None} for r in rows]

@app.post("/api/topics")
def add_topic(payload:TopicIn, request:Request):
    require_auth(request)
    with SessionLocal() as db:
        r=ResearchTopic(topic=payload.topic)
        db.add(r); db.commit(); db.refresh(r)
        return {"id":r.id}
