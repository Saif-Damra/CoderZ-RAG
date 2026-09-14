# Runs the FastAPI app (api/main.py) behind uvicorn. The chat widget is
# served by the same process via the /widget static mount (see api/main.py),
# so this one container is everything a teammate needs to hit.
#
# data/feedback.db (SQLite) must live on a mounted volume — see
# docker-compose.yml's `volumes: - ./data:/app/data` — otherwise a rebuild
# discards it.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# --proxy-headers + --forwarded-allow-ips: this container sits behind a
# reverse proxy (Traefik) that terminates TLS. Without these, uvicorn builds
# redirects (e.g. the StaticFiles trailing-slash redirect for /review) using
# the plain-HTTP scheme it actually receives, sending browsers to http://
# instead of https://.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
