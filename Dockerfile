FROM python:3.13-slim

WORKDIR /workspace

# Copy entire root directory except gitignore files
COPY . /workspace
