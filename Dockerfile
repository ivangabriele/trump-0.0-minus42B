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

RUN useradd -m -u 1000 user

WORKDIR /app
RUN chown user /app
RUN chmod 755 /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"
RUN curl -fsSL https://pyenv.run | bash
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# COPY . /app
COPY --chown=user . /app

RUN ls -la /app

RUN uv sync

CMD ["python", "-m", "http.server", "--directory", "public"]
