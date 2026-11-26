FROM python:3.12-slim
ARG EXTRAS=dashboard
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libgles2 libegl1 libglib2.0-0 libportaudio2 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ego-vta
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir ".[${EXTRAS}]"
WORKDIR /data
ENTRYPOINT ["ego-vta"]
