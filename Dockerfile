# Base install only: requirements.txt, never requirements-local.txt.
#
# The local classifiers pull torch and about 1.6GB of weights, which takes the
# running process from 97MB to 1,090MB — measured, not estimated. That does not
# fit a 512MB free tier, and the evaluation put the layer's contribution at 4.6%
# of judge calls, so it is a poor trade even where it does fit. Without them the
# content, injection and grounding rails fall through to the Claude judge, which
# is the documented base path rather than a degraded one.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Wheels cover everything in requirements.txt on slim, except that pymupdf and
# cryptography want a compiler if a wheel is ever missing for the platform.
# Installed and removed in one layer so the build tools do not ship.
COPY requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm \
 && apt-get purge -y --auto-remove build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY config/   ./config/
COPY eval/     ./eval/
COPY run.py    ./

# `data/` is written at runtime: accounts, sessions, the corpus, transcripts.
# On a free tier the filesystem is ephemeral, so this resets on every deploy —
# which means the built-in demo accounts come back and the corpus reseeds. Mount
# a volume here if any of it needs to survive.
RUN mkdir -p data

# 127.0.0.1 is the right default for a laptop and useless in a container: the
# platform's proxy would never reach it. PORT is overridden by both Render and
# Cloud Run, so it is a default rather than a setting.
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# The app's own readiness answer — it reports config errors and whether the
# model rails are live, so a green check here means more than "the port is open".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT','8000')}/api/health\", timeout=4).status==200 else 1)"

CMD ["python", "run.py"]
