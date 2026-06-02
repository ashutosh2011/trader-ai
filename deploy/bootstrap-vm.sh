#!/usr/bin/env bash
# One-time (idempotent) VM bootstrap for tradebot Docker deployment.
# Targets Debian/Ubuntu — works on GCE, EC2, Linode, Hetzner, DigitalOcean, etc.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/OWNER/REPO/main/tradebot/deploy/bootstrap-vm.sh | bash
#   # or: scp deploy/bootstrap-vm.sh user@host: && ssh user@host 'sudo bash bootstrap-vm.sh'
#
# No cloud-provider CLI calls — open ports 80/443 in your provider firewall separately.

set -euo pipefail

APP_USER="${APP_USER:-tradebot}"
APP_DIR="${APP_DIR:-/opt/tradebot}"
DEPLOY_GROUP="${DEPLOY_GROUP:-docker}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    log "Re-run as root: sudo bash $0"
    exit 1
  fi
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    log "Docker already installed: $(docker --version)"
    return 0
  fi

  log "Installing Docker Engine + Compose plugin..."
  apt-get update -qq
  apt-get install -y ca-certificates curl gnupg

  install -m 0755 -d /etc/apt/keyrings
  if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
      | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
  fi

  local codename
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}")"
  if [[ -z "${codename}" ]]; then
    codename="bookworm"
  fi

  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu ${codename} stable" \
    > /etc/apt/sources.list.d/docker.list

  apt-get update -qq
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
  log "Docker installed: $(docker --version)"
}

ensure_app_user() {
  if ! id "${APP_USER}" >/dev/null 2>&1; then
    log "Creating user ${APP_USER}..."
    useradd --create-home --shell /bin/bash "${APP_USER}"
  else
    log "User ${APP_USER} already exists"
  fi

  usermod -aG "${DEPLOY_GROUP}" "${APP_USER}" || true
}

prepare_directories() {
  log "Preparing ${APP_DIR}..."
  install -d -m 0755 "${APP_DIR}"
  install -d -m 0755 "${APP_DIR}/config" "${APP_DIR}/runtime" "${APP_DIR}/logs"

  if [[ -f "${APP_DIR}/config.docker.yaml" && ! -f "${APP_DIR}/config/config.yaml" ]]; then
    cp "${APP_DIR}/config.docker.yaml" "${APP_DIR}/config/config.yaml"
  fi

  if [[ -f "${APP_DIR}/env.example" && ! -f "${APP_DIR}/.env" ]]; then
    cp "${APP_DIR}/env.example" "${APP_DIR}/.env"
    log "Created ${APP_DIR}/.env from env.example — edit secrets before first deploy"
  fi

  chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
}

print_next_steps() {
  cat <<EOF

Bootstrap complete.

Next steps:
  1. Point DNS A record for your domain to this VM's public IP.
  2. Open inbound TCP 80 and 443 in your cloud firewall (see docs/deployment.md).
  3. Edit ${APP_DIR}/.env — set DEPLOY_DOMAIN, IMAGE, BASIC_AUTH_*, Kite, and LLM keys.
  4. Generate basic-auth hash:
       docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'
  5. Add GitHub repository secrets (DEPLOY_HOST, DEPLOY_USER, DEPLOY_SSH_KEY, ...).
  6. Push to main or run the deploy workflow manually.

Deploy directory: ${APP_DIR}
Runtime user:     ${APP_USER}

EOF
}

main() {
  require_root
  install_docker
  ensure_app_user
  prepare_directories
  print_next_steps
}

main "$@"
