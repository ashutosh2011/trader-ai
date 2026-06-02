# VM deployment (Docker Compose + Caddy)

Deploy tradebot to any SSH-reachable Linux VM using Docker Compose, Caddy (HTTPS + basic auth), and GitHub Actions. The workflow and scripts are **cloud-agnostic** — today's target might be GCP; tomorrow Linode or AWS. No provider CLI calls in this repo.

## Architecture

```
GitHub (main) → Actions → GHCR image + SSH deploy → VM
VM → Docker Compose → Caddy (443) → dashboard (8765)
                   └→ paper profile (optional)
                   └→ live profile (optional, dry-run by default)
```

Persistent data lives on the VM under `/opt/tradebot/`:

| Path | Purpose |
|------|---------|
| `config/` | `config.yaml`, universe files |
| `runtime/` | DuckDB state (`order_state.duckdb`, `dashboard.duckdb`, candles) |
| `logs/` | Paper/live journals |
| `.env` | Secrets (never in the image or git) |

## Prerequisites

- Debian/Ubuntu VM with a public IP
- DNS `A` record for your domain → VM IP
- Inbound TCP **80** and **443** open in your cloud firewall
- GitHub repository with Actions enabled

## One-time VM bootstrap

SSH into the VM as root (or sudo):

```bash
# Option A: download from your repo after pushing this branch
curl -fsSL https://raw.githubusercontent.com/ashutosh2011/trader-ai/main/deploy/bootstrap-vm.sh | sudo bash

# Option B: copy from your laptop
scp deploy/bootstrap-vm.sh user@VM_IP:
ssh user@VM_IP 'sudo bash bootstrap-vm.sh'
```

This installs Docker Engine + Compose plugin, creates user `tradebot`, and prepares `/opt/tradebot`.

Copy or edit deployment config on the VM:

```bash
sudo -u tradebot cp /opt/tradebot/config.docker.yaml /opt/tradebot/config/config.yaml
sudo -u tradebot cp /opt/tradebot/env.example /opt/tradebot/.env
sudo -u tradebot $EDITOR /opt/tradebot/.env
```

Generate a Caddy basic-auth bcrypt hash:

```bash
docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'
```

Put the hash in `.env` as `BASIC_AUTH_HASH`. **Note:** bcrypt hashes contain `$`; when editing `.env` manually, double each `$` (e.g. `$$2a$$14$$...`) so Docker Compose does not treat them as variable references. The deploy workflow escapes this automatically.

## GitHub secrets

Add these in **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `DEPLOY_HOST` | VM public IP or DNS name |
| `DEPLOY_USER` | SSH user (e.g. `tradebot` or your sudo user) |
| `DEPLOY_SSH_KEY` | Private key for deploy user |
| `DEPLOY_DOMAIN` | Public domain (e.g. `tradebot.example.com`) |
| `BASIC_AUTH_USER` | Dashboard username |
| `BASIC_AUTH_HASH` | Caddy bcrypt hash (not plaintext password) |
| `KITE_API_KEY` | Zerodha Kite API key |
| `KITE_API_SECRET` | Kite API secret |
| `KITE_ACCESS_TOKEN` | Daily Kite access token |
| `GOOGLE_API_KEY` | Vertex Express / Gemini key (`AQ.*` or `AIza*`) |
| `OPENAI_API_KEY` | Optional |
| `ANTHROPIC_API_KEY` | Optional |
| `BASIC_AUTH_PASSWORD` | Optional — enables public HTTPS smoke check in CI |

Runtime secrets are written into the VM `.env` on each deploy. You can also maintain `.env` manually on the VM; deploy upserts non-empty secret values.

## Deploy

Push to `main` or run **Actions → Deploy → Run workflow**.

The workflow:

1. Runs `ruff`, `mypy --strict`, `pytest -q`
2. Builds and pushes `ghcr.io/ashutosh2011/trader-ai:<sha>` and `:main`
3. SSHes to the VM, syncs `deploy/`, sets `IMAGE`, runs `docker compose pull && docker compose up -d`
4. Health-checks the dashboard inside the container

First deploy: ensure the deploy user can run Docker (`usermod -aG docker tradebot`) and can `docker login ghcr.io` (the workflow logs in via `GITHUB_TOKEN` during deploy).

## Firewall (per cloud)

Open **TCP 80** and **443** inbound to the VM. Examples:

### Google Cloud (GCE)

```bash
gcloud compute firewall-rules create tradebot-web \
  --allow=tcp:80,tcp:443 \
  --target-tags=tradebot \
  --description="HTTPS for tradebot dashboard"
```

Tag the VM: `gcloud compute instances add-tags INSTANCE --tags=tradebot --zone=ZONE`

### AWS (EC2)

Security group inbound rules:

- TCP 80 from `0.0.0.0/0`
- TCP 443 from `0.0.0.0/0`

### Linode / Hetzner / DigitalOcean

Use the cloud firewall UI: allow inbound TCP 80 and 443 to the VM.

## Operations

### Logs

```bash
cd /opt/tradebot
docker compose logs -f dashboard
docker compose logs -f caddy
```

### Rollback

```bash
cd /opt/tradebot
# Pin to a previous image tag (commit SHA from GHCR)
sed -i 's|^IMAGE=.*|IMAGE=ghcr.io/ashutosh2011/trader-ai:PREVIOUS_SHA|' .env
docker compose pull && docker compose up -d
```

### Enable paper trading

```bash
cd /opt/tradebot
docker compose --profile paper up -d
```

### Enable live trading (dangerous)

The `live` profile starts with `--dry-run`. To go live:

1. Confirm kill-switch is clear (`runtime/KILL` absent, `KILL_SWITCH=0`)
2. Refresh Kite token daily
3. Edit `docker-compose.yml` live service command — remove `--dry-run` only after explicit review
4. `docker compose --profile live up -d`

### Kite daily token

Kite tokens expire each trading day. Update GitHub secret `KITE_ACCESS_TOKEN` and redeploy, or edit `/opt/tradebot/.env` and restart:

```bash
docker compose restart dashboard paper live
```

## Switching clouds (e.g. GCP → Linode)

1. Provision a new Debian/Ubuntu VM
2. Run `bootstrap-vm.sh`
3. Repoint DNS `A` record to the new IP
4. Update GitHub secrets: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`
5. Optionally rsync `/opt/tradebot/runtime/` from the old VM to preserve DuckDB history
6. Run the deploy workflow

No code changes required.

## Local smoke checks

Before pushing:

```bash
ruff check .
mypy --strict .
pytest -q
docker build .
docker compose -f deploy/docker-compose.yml config
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Caddy won't get cert | DNS points to VM? Ports 80/443 open? `DEPLOY_DOMAIN` correct in `.env`? |
| 401 on dashboard | `BASIC_AUTH_USER` / `BASIC_AUTH_HASH` match your login? |
| Dashboard unhealthy | `docker compose logs dashboard`; Kite token valid? |
| GHCR pull denied | Deploy user logged into `ghcr.io`; package visibility allows the repo |
