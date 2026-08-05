FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=glfw

# GLFW/X11 enables the interactive viewer. EGL/OSMesa also make the same
# image usable for offscreen rendering on machines without an X display.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libgles2 \
        libglfw3 \
        libosmesa6 \
        libx11-6 \
        libxcursor1 \
        libxinerama1 \
        libxi6 \
        libxrandr2 \
        ffmpeg \
        ca-certificates \
        git \
        tk \
    && rm -rf /var/lib/apt/lists/*

ARG MUJOCO_MENAGERIE_COMMIT=c1a4eeb85694ae1dffe33ff1797d4e528928a133
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/google-deepmind/mujoco_menagerie.git \
        /opt/mujoco_menagerie \
    && git -C /opt/mujoco_menagerie sparse-checkout init --cone \
    && git -C /opt/mujoco_menagerie sparse-checkout set google_robot \
    && git -C /opt/mujoco_menagerie checkout "$MUJOCO_MENAGERIE_COMMIT" \
    && rm -rf /opt/mujoco_menagerie/.git

WORKDIR /workspace

COPY mujoco_scenes/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY mujoco_scenes/ /workspace/mujoco_scenes/

ENV MUJOCO_SEMANTIC_PROCESS_ISOLATION=1 \
    MUJOCO_MENAGERIE_PATH=/opt/mujoco_menagerie

ENTRYPOINT ["python", "-m", "mujoco_scenes.scene_loader"]
CMD ["--scene", "S1_coffee_missing_mug", "--viewer", "--camera", "front_camera"]
