"""AI Video Assistant: Flask + MongoDB + Claude vision, with user accounts."""
import base64, logging, os, tempfile
from datetime import datetime, timezone
from functools import wraps

import anthropic
import cv2
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash

load_dotenv()
env = os.getenv
DEV = env("FLASK_ENV") == "development"
SECRET = env("SECRET_KEY")
if not SECRET:
    if not DEV:
        raise RuntimeError("SECRET_KEY must be set (see .env.example).")
    SECRET = "dev-only-secret"
MAX_MB = int(env("MAX_UPLOAD_MB", "100"))
NUM_FRAMES = max(2, min(int(env("NUM_FRAMES", "8")), 20))
MODEL = env("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SYSTEM = ("You are a video analysis assistant. You are given evenly spaced frames from one video, "
          "each labelled with its timestamp. Answer only from what the frames show, cite timestamps, "
          "and say so when something can't be determined from sampled frames.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
app.config.update(SECRET_KEY=SECRET, MAX_CONTENT_LENGTH=MAX_MB * 1024 * 1024,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=not DEV, PERMANENT_SESSION_LIFETIME=7 * 86400)
if env("TRUST_PROXY") == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
limiter = Limiter(get_remote_address, app=app, storage_uri=env("RATELIMIT_STORAGE_URI", "memory://"))
db = None


def get_db():
    global db
    if db is None:
        client = MongoClient(env("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5000)
        db = client[env("MONGO_DB", "video_assistant")]
        db.users.create_index("email", unique=True)
        db.sessions.create_index([("user_id", 1), ("created_at", -1)])
        db.sessions.create_index("created_at", expireAfterSeconds=int(env("SESSION_TTL_DAYS", "30")) * 86400)
    return db


def error(msg, code):
    return jsonify(error=msg), code


def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if "uid" not in session:
            return error("Please sign in.", 401)
        return f(*a, **k)
    return wrapper


def oid(value):
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


@app.before_request
def csrf_guard():
    # Cross-site forms/requests cannot set custom headers, so this plus SameSite=Lax blocks CSRF.
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("X-Requested-With") != "fetch":
        return error("Invalid request.", 403)


@app.after_request
def security_headers(resp):
    resp.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
                         "Referrer-Policy": "same-origin"})
    if not DEV:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000"
    return resp


@app.errorhandler(413)
def too_large(_):
    return error(f"File is too large. The limit is {MAX_MB} MB.", 413)


@app.errorhandler(429)
def rate_limited(_):
    return error("Too many requests. Wait a moment and try again.", 429)


@app.errorhandler(PyMongoError)
def db_error(e):
    app.logger.exception("database error")
    return error("The database is unavailable. Try again shortly.", 503)


# ---------- video + AI ----------
def extract_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not cap.isOpened() or total <= 0 or fps <= 0:
        cap.release()
        raise ValueError("This file could not be read as a video.")
    frames = []
    for i in range(NUM_FRAMES):
        idx = int((i + 0.5) * total / NUM_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        s = min(1.0, 768 / max(h, w))
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (int(w * s), int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append({"t": round(idx / fps, 1), "b64": base64.b64encode(buf).decode()})
    cap.release()
    if not frames:
        raise ValueError("No frames could be extracted from this video.")
    return frames, round(total / fps, 1)


def frames_message(frames):
    content = [{"type": "text", "text": "Frames from the video:"}]
    for fr in frames:
        content.append({"type": "text", "text": f"Frame at {fr['t']}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": fr["b64"]}})
    content.append({"type": "text", "text": "Summarize what happens in this video, with key moments and timestamps."})
    return {"role": "user", "content": content}


def ask(messages):
    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("no key")
    resp = anthropic.Anthropic(api_key=key).messages.create(model=MODEL, max_tokens=1200, system=SYSTEM, messages=messages)
    return "".join(b.text for b in resp.content if b.type == "text")


def ai_error(e):
    app.logger.error("AI error: %s", e)
    if isinstance(e, RuntimeError):
        return error("The server has no ANTHROPIC_API_KEY set.", 503)
    return error("The AI service returned an error. Try again.", 502)


# ---------- pages + health ----------
@app.get("/")
def index():
    if "uid" not in session:
        return redirect("/login")
    return render_template("index.html", max_mb=MAX_MB, email=session.get("email"),
                           configured=bool(env("ANTHROPIC_API_KEY")))


@app.get("/login")
def login_page():
    return render_template("login.html")


@app.get("/healthz")
def healthz():
    try:
        get_db().command("ping")
        return jsonify(status="ok")
    except Exception:
        return jsonify(status="db unavailable"), 503


# ---------- auth ----------
def credentials():
    d = request.get_json(silent=True) or {}
    email, pw = str(d.get("email", "")).strip().lower(), str(d.get("password", ""))
    return email, pw


def start_session(uid, email):
    session.clear()
    session.permanent = True
    session["uid"], session["email"] = str(uid), email


@app.post("/api/register")
@limiter.limit("10/hour")
def register():
    email, pw = credentials()
    if "@" not in email or len(email) > 254:
        return error("Enter a valid email address.", 400)
    if not 8 <= len(pw) <= 128:
        return error("Password must be 8 to 128 characters.", 400)
    try:
        r = get_db().users.insert_one({"email": email, "pw": generate_password_hash(pw),
                                       "created_at": datetime.now(timezone.utc)})
    except DuplicateKeyError:
        return error("An account with this email already exists.", 409)
    start_session(r.inserted_id, email)
    return jsonify(ok=True)


@app.post("/api/login")
@limiter.limit("10/minute")
def login():
    email, pw = credentials()
    user = get_db().users.find_one({"email": email})
    if not user or not check_password_hash(user["pw"], pw):
        return error("Incorrect email or password.", 401)
    start_session(user["_id"], email)
    return jsonify(ok=True)


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


# ---------- sessions ----------
def own(sid, projection=None):
    return get_db().sessions.find_one({"_id": oid(sid), "user_id": session["uid"]}, projection)


@app.get("/api/sessions")
@login_required
def list_sessions():
    rows = get_db().sessions.find({"user_id": session["uid"]}, {"title": 1, "created_at": 1}).sort("created_at", -1).limit(50)
    return jsonify([{"id": str(r["_id"]), "title": r["title"], "created_at": r["created_at"].isoformat() + "Z"} for r in rows])


@app.get("/api/sessions/<sid>")
@login_required
def get_session(sid):
    d = own(sid)
    if not d:
        return error("Analysis not found.", 404)
    return jsonify(title=d["title"], duration=d["duration"],
                   frames=[{"t": f["t"], "src": "data:image/jpeg;base64," + f["b64"]} for f in d["frames"]],
                   messages=[{"role": "assistant", "content": d["summary"]}] + d["chat"])


@app.delete("/api/sessions/<sid>")
@login_required
def delete_session(sid):
    r = get_db().sessions.delete_one({"_id": oid(sid), "user_id": session["uid"]})
    return jsonify(ok=True) if r.deleted_count else error("Analysis not found.", 404)


@app.post("/api/upload")
@login_required
@limiter.limit("20/hour")
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return error("Choose a video file to upload.", 400)
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return error(f"Unsupported format. Use {', '.join(sorted(ALLOWED))}.", 400)
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(path)
        frames, duration = extract_frames(path)
        summary = ask([frames_message(frames)])
    except ValueError as e:
        return error(str(e), 422)
    except (anthropic.APIError, RuntimeError) as e:
        return ai_error(e)
    finally:
        os.remove(path)
    r = get_db().sessions.insert_one({"user_id": session["uid"], "title": f.filename[:80], "duration": duration,
                                      "frames": frames, "summary": summary, "chat": [],
                                      "created_at": datetime.now(timezone.utc)})
    return jsonify(session_id=str(r.inserted_id), duration=duration, summary=summary,
                   frames=[{"t": x["t"], "src": "data:image/jpeg;base64," + x["b64"]} for x in frames])


@app.post("/api/chat")
@login_required
@limiter.limit("30/minute")
def chat():
    data = request.get_json(silent=True) or {}
    text = str(data.get("message") or "").strip()
    d = own(data.get("session_id"))
    if not d:
        return error("This analysis was not found. Upload the video again.", 404)
    if not text or len(text) > 2000:
        return error("Enter a question of up to 2000 characters.", 400)
    msgs = [frames_message(d["frames"]), {"role": "assistant", "content": d["summary"]}] + d["chat"][-20:]
    msgs.append({"role": "user", "content": text})
    try:
        answer = ask(msgs)
    except (anthropic.APIError, RuntimeError) as e:
        return ai_error(e)
    get_db().sessions.update_one({"_id": d["_id"]}, {"$push": {"chat": {"$each": [
        {"role": "user", "content": text}, {"role": "assistant", "content": answer}]}}})
    return jsonify(answer=answer)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(env("PORT", "5000")), debug=False)
