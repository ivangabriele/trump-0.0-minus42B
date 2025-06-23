#!/bin/bash
set -euo pipefail

# Only do the SSH dance when a key is supplied (so the image still runs in jobs where no pushing is needed).
if [[ -n "${GIT_SSH_PRIVATE_KEY:-}" ]]; then
  echo "Info: Configuring SSH key for Git pushes…"

  if [[ -z "${GIT_USER_NAME:-}" ]] || [[ -z "${GIT_USER_EMAIL:-}" ]] || [[ -z "${GIT_REMOTE_URL:-}" ]]; then
    echo "Error: 'GIT_USER_NAME', 'GIT_USER_EMAIL' and 'GIT_REMOTE_URL' must be set when 'GIT_SSH_PRIVATE_KEY' is provided."
    exit 1
  fi

  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"
  printf '%s\n' "$GIT_SSH_PRIVATE_KEY" > "${HOME}/.ssh/id_ed25519"
  chmod 600 "${HOME}/.ssh/id_ed25519"

  # Accept GitHub & HF Hub host keys to avoid the first-time prompt
  ssh-keyscan github.com hf.co >> "${HOME}/.ssh/known_hosts" 2>/dev/null

  # Global Git identity
  git config --global user.name  "${GIT_USER_NAME}"
  git config --global user.email "${GIT_USER_EMAIL}"

  # Tell Git which key to use *without* having to run ssh-agent
  export GIT_SSH_COMMAND="ssh -i ${HOME}/.ssh/id_ed25519 -o IdentitiesOnly=yes"

  echo "Info: SSH key configured for Git pushes."

  # ---

  echo "Info: Setting up Git for /app…"
  sleep 1  # Ensure the directory is ready
  rm -fr /app/.git
  sleep 1  # Ensure the directory is ready
  git init /app
  git lfs install /app
  git -C /app remote add hf "${GIT_REMOTE_URL}"
  git fetch hf
  echo "Info: Git setup complete for /app."
fi

make serve
