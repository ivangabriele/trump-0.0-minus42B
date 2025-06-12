# https://huggingface.co/docs/hub/spaces-dev-mode#docker-spaces

FROM python:3.13-bookworm

RUN apt-get update
RUN apt-get install -y \
  bash \
  curl \
  git \
  git-lfs \
  htop \
  procps \
  nano \
  vim \
  wget
RUN rm -fr /var/lib/apt/lists/*
RUN curl -fsSL https://pyenv.run | bash
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
COPY --link . /app
RUN chown user /app

RUN uv sync
