# Deployment image for the HASEBHA / CommercePilot AI FastAPI service
# (src/ai_service). Uses requirements-deploy.txt (scoped runtime deps),
# not the full research requirements.txt -- see that file's comments.
FROM python:3.12-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Everything else needed at runtime is copied by the deploy tool
# (railway up, filtered by .railwayignore) rather than baked in here,
# so this Dockerfile stays valid whether built from an upload or from
# git in the future once/if artifacts are made available another way.
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8123

CMD ["sh", "-c", "uvicorn src.ai_service.main:app --host 0.0.0.0 --port ${PORT:-8123}"]
