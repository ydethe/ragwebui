# Stage 1: Build
FROM python:3.13-slim AS builder

ARG LOGLEVEL
ARG QDRANT_HOST
ARG QDRANT_HTTPS
ARG QDRANT_PORT
ARG QDRANT_QUERY_LIMIT
ARG QDRANT_API_KEY
ARG OPENAI_API_KEY
ARG COLLECTION_NAME
ARG DAV_ROOT
ARG EMBEDDING_MODEL
ARG EMBEDDING_MODEL_TRUST_REMOTE_CODE
ARG OPEN_MODEL_PREF
ARG TORCH_NUM_THREADS

ENV LOGLEVEL=$LOGLEVEL
ENV QDRANT_HOST=$QDRANT_HOST
ENV QDRANT_HTTPS=$QDRANT_HTTPS
ENV QDRANT_PORT=$QDRANT_PORT
ENV QDRANT_QUERY_LIMIT=$QDRANT_QUERY_LIMIT
ENV QDRANT_API_KEY=$QDRANT_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV COLLECTION_NAME=$COLLECTION_NAME
ENV DAV_ROOT=$DAV_ROOT
ENV EMBEDDING_MODEL=$EMBEDDING_MODEL
ENV EMBEDDING_MODEL_TRUST_REMOTE_CODE=$EMBEDDING_MODEL_TRUST_REMOTE_CODE
ENV OPEN_MODEL_PREF=$OPEN_MODEL_PREF
ENV TORCH_NUM_THREADS=$TORCH_NUM_THREADS

WORKDIR /app

RUN python3 -m venv venv --system-site-packages && \
    ./venv/bin/python -m pip install --no-cache-dir -U torch --index-url https://download.pytorch.org/whl/cpu && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

COPY requirements.txt .
RUN ./venv/bin/python -m pip install --no-cache-dir -U -r requirements.txt && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

COPY dist/*.whl .
RUN ./venv/bin/python -m pip install --no-cache-dir *.whl && \
    rm -f *.whl && \
    find /app/venv \( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+

# # Stage 2: Production
# FROM python:3.13-slim

# # Set the working directory
# WORKDIR /app

# # Copy only the necessary files from the build stage
# COPY --from=builder /app /app

# Expose the port the app will run on
EXPOSE 7860

CMD ["/app/venv/bin/python", "-m" , "ragwebui"]
