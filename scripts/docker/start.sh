#!/bin/bash
set -Eeuo pipefail # stop on errors, undefined vars, or pipe breaks

retry_delete() {
  local target=$1
  local max_tries=${2}
  local delay=${3}
  local i
  for ((i = 1; i <= max_tries; ++i)); do
    echo "Info: Attempt $i of $max_tries to remove '$target'…"
    rm -rf "$target" && return 0
    sleep "$delay"
  done
  echo "Warning: couldn't fully remove $target after ${max_tries} tries." >&2
}

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

  # There seem to be concurrency issues with HF space dev mode which injects some scripts into the container.
  echo "Info: attempting to remove .git directory…"
  sleep 5
  git lfs uninstall || true
  retry_delete .git 10 1
  echo "Info: .git directory removed."

  echo "Info: Setting up Git…"
  git init
  git lfs install
  git remote add origin "${GIT_REMOTE_URL}"
  git fetch --depth 1 origin main
  git reset --hard origin/main
  git branch --set-upstream-to=origin/main main
  echo "Info: Git setup complete."
fi

make serve
