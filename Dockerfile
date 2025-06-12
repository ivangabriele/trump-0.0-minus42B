# Must be debian-based because of https://huggingface.co/docs/hub/spaces-dev-mode#docker-spaces
FROM python:3.13-bookworm

# Named `app` because of https://huggingface.co/docs/hub/spaces-dev-mode#docker-spaces
# For build order optimization
RUN mkdir -p /app

# Owned by 1000:1000 because of https://huggingface.co/docs/hub/spaces-dev-mode#docker-spaces
RUN chown -R 1000:1000 /app

COPY . /app

RUN apt-get update
# Including https://huggingface.co/docs/hub/spaces-dev-mode#docker-spaces requirements (procps)
RUN apt-get install -y \
    curl \
    git \
    git-lfs \
    procps \
    vim \
    wget
RUN curl -fsSL https://pyenv.run | bash
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
