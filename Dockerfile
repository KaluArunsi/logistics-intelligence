FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

COPY src ./src
COPY frontend ./frontend
COPY tools ./tools
COPY config ./config
COPY HEART.md ./HEART.md
COPY SOUL.md ./SOUL.md
COPY core_user.md ./core_user.md
COPY core_user_flow_spec.md ./core_user_flow_spec.md
COPY pipeline_guide.md ./pipeline_guide.md
COPY principles.md ./principles.md

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "tools/run_backend.py", "--host", "0.0.0.0"]
