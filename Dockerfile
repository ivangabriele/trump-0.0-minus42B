# https://huggingface.co/docs/hub/en/spaces-sdks-docker-first-demo#create-the-dockerfile

FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV RUNNING_IN_DOCKER=true

RUN apt-get update
RUN apt-get install -y \
  bash \
  curl \
  git \
  git-lfs \
  htop \
  openssh-client \
  procps \
  python-is-python3 \
  python3 \
  nano \
  vim \
  wget
RUN rm -fr /var/lib/apt/lists/*

WORKDIR /app
RUN chown ubuntu /app
RUN chmod 755 /app

USER ubuntu
ENV PATH="/home/ubuntu/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY --chown=ubuntu . /app

ENV UV_NO_CACHE=true
RUN uv venv
RUN uv sync

SHELL ["/usr/bin/bash", "-c"]

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"

COPY --chown=ubuntu ./scripts/docker/start.sh /usr/local/bin/start.sh
RUN chmod 700 /usr/local/bin/start.sh

CMD ["/usr/local/bin/start.sh"]
