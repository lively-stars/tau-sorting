#!/usr/bin/env bash
#
# deploy-to-1blu.sh — deploy the Q_rad webapp Docker image to the "1blu" VPS.
#
# The GHCR image is PRIVATE, so the VPS cannot pull it. Instead we build the
# amd64 image locally (Apple Silicon / OrbStack is arm64, the VPS is x86_64),
# stream it over SSH into the VPS's local Docker, and recreate the container
# with the compose file that already lives on the VPS.
#
# This script ONLY builds, ships, and `up -d`s. It never touches the VPS
# compose file, the mounted ODF/model inputs, or any other service. Safe to
# re-run.
#
set -euo pipefail
IFS=$'\n\t'

# ---- configuration (edit here) ---------------------------------------------
readonly IMAGE_TAG="ghcr.io/lively-stars/tau-sorting:latest"
readonly VPS_HOST="1blu"                       # alias configured in ~/.ssh/config
readonly REMOTE_DIR='~/docker/tausorting'      # holds docker-compose.yaml on the VPS
readonly CONTAINER="tausorting-tausorting-1"
readonly PUBLIC_URL="https://tausorting.mihac.de"
readonly HEALTH_URL="${PUBLIC_URL}/api/init"
readonly DOCKER_START_TIMEOUT=60               # seconds to wait for the local daemon
readonly HEALTH_TIMEOUT=180                    # seconds to wait for the app to serve 200

# ---- resolve repo root = this script's directory ---------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
readonly SCRIPT_DIR
cd "$SCRIPT_DIR"

die() { echo "❌ ERROR: $*" >&2; exit 1; }
phase() { echo; echo "==> $*"; }

START_TS=$(date +%s)

# ---- 1. preflight ----------------------------------------------------------
phase "Preflight: local Docker daemon"
if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon not responding; attempting to start OrbStack…"
    open -a OrbStack 2>/dev/null || die "could not launch OrbStack (open -a OrbStack failed)"
    deadline=$(( $(date +%s) + DOCKER_START_TIMEOUT ))
    until docker info >/dev/null 2>&1; do
        [ "$(date +%s)" -ge "$deadline" ] && \
            die "Docker daemon did not come up within ${DOCKER_START_TIMEOUT}s"
        sleep 3
    done
fi
echo "Docker daemon is up."

phase "Preflight: SSH to ${VPS_HOST} and remote dir ${REMOTE_DIR}"
ssh -o ConnectTimeout=15 "$VPS_HOST" "test -d ${REMOTE_DIR}" \
    || die "cannot reach ${VPS_HOST} or ${REMOTE_DIR} is missing (compose dir must exist)"
echo "SSH ok; remote compose dir present."

# ---- 2. build (amd64) ------------------------------------------------------
phase "Build amd64 image ${IMAGE_TAG}"
docker buildx build --platform linux/amd64 -t "$IMAGE_TAG" --load . \
    || die "docker buildx build failed"
echo "Build ok."

# ---- 3. ship over SSH (no registry) ----------------------------------------
phase "Ship image to ${VPS_HOST} via docker save | ssh docker load"
docker save "$IMAGE_TAG" | ssh "$VPS_HOST" 'docker load' \
    || die "docker save | ssh docker load failed"
echo "Image loaded on VPS."

# ---- 4. restart the service (recreate container only) ----------------------
phase "Recreate container on ${VPS_HOST} (docker compose up -d)"
ssh "$VPS_HOST" "cd ${REMOTE_DIR} && docker compose up -d" \
    || die "docker compose up -d failed on VPS"
echo "Container recreated."

# ---- 5. verify readiness ---------------------------------------------------
phase "Verify: poll ${HEALTH_URL} for HTTP 200 (app re-precomputes on startup)"
http_code="000"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while :; do
    http_code="$(ssh "$VPS_HOST" \
        "curl -s -o /dev/null -w '%{http_code}' ${HEALTH_URL}" 2>/dev/null || echo 000)"
    if [ "$http_code" = "200" ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "…still not ready after ${HEALTH_TIMEOUT}s (last HTTP ${http_code})."
        break
    fi
    echo "…HTTP ${http_code}, waiting…"
    sleep 6
done

container_status="$(ssh "$VPS_HOST" \
    "docker inspect ${CONTAINER} --format 'status={{.State.Status}} restarts={{.RestartCount}}'" \
    2>/dev/null || echo 'status=unknown restarts=?')"

# ---- summary ---------------------------------------------------------------
elapsed=$(( $(date +%s) - START_TS ))
echo
echo "=========================== DEPLOY SUMMARY ==========================="
echo "  Image:      ${IMAGE_TAG}"
echo "  VPS:        ${VPS_HOST}  (${REMOTE_DIR})"
echo "  Container:  ${container_status}"
echo "  Health:     HTTP ${http_code} @ ${HEALTH_URL}"
echo "  URL:        ${PUBLIC_URL}"
echo "  Elapsed:    ${elapsed}s"
if [ "$http_code" = "200" ]; then
    echo "  Result:     ✅ SUCCESS — app is serving."
else
    echo "  Result:     ❌ NOT VERIFIED — app did not return 200 in time."
fi
echo "======================================================================"

[ "$http_code" = "200" ] || exit 1
