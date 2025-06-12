FROM python:3.13-slim

# For build order optimization
RUN mkdir -p /workspace

COPY . /workspace

RUN apt-get update
RUN apt-get install -y \
    curl \
    git \
    vim \
    wget
RUN curl -fsSL https://pyenv.run | bash
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
