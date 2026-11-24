# Video Assistant

Upload a video, get an AI summary, and chat about it. Flask + MongoDB + Claude vision, with per-user accounts.

Flow: sign in -> upload -> OpenCV samples frames -> Claude summarizes -> analysis, frames and chat are saved in MongoDB and listed in the sidebar history. The uploaded video file is deleted right after frame extraction; only the sampled frames are stored.

## Deploy (Docker Compose: app + MongoDB + Caddy with automatic HTTPS)
1. Get a server with Docker and point a domain's DNS A record at it.
2. `cp .env.example .env` and set `ANTHROPIC_API_KEY`, `SECRET_KEY`, `MONGO_PASSWORD` (URL-safe characters), and `DOMAIN` (your domain; keep `localhost` to try it locally).
3. `docker compose up -d --build`
4. Open `https://<DOMAIN>`, create an account. Health check: `/healthz`.

MongoDB is not exposed publicly, uses a password, and stores data in a Docker volume. Back that volume up.

## Run locally without Docker
    pip install -r requirements.txt
    # needs a MongoDB at MONGO_URI (default mongodb://localhost:27017)
    cp .env.example .env    # set ANTHROPIC_API_KEY; use FLASK_ENV=development for http
    python app.py

## Security notes
Passwords are hashed; cookies are HttpOnly, SameSite=Lax and Secure in production; state-changing requests need an `X-Requested-With` header (CSRF); login, register, upload and chat are rate limited; users can only read their own analyses; sessions expire after `SESSION_TTL_DAYS` (default 30) via a MongoDB TTL index. Rate-limit counters are per worker unless `RATELIMIT_STORAGE_URI` points to Redis.

## Limits
Analysis uses sampled frames only; audio and speech are not analyzed. No password reset or email verification yet.
