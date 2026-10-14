#!/usr/bin/env bash
#
# deploy-to-oracle.sh — deploy the Q_rad webapp to the "oraclearm3" VPS.
#
# The VPS is the same arch as the build host, so unlike deploy-to-1blu.sh there
# is no local cross-build and no image streaming: we rsync this repo into the
# remote build context (./src, a plain copy, no .git) and rebuild there with
# `docker compose up -d --build tau`.
#
# Remote layout (/home/ubuntu/docker/tau):
#   docker-compose.yaml  — compose file (NEVER touched by this script)
#   src/                 — synced copy of this repo (the docker build context)
#   ODF_format.npy, continuumabs.dat, continuumabs.npy, data/ — large runtime
#     inputs mounted into the container as /data:ro via TAUSORT_DATA_DIR=/data
#     (NEVER synced, NEVER deleted; continuumabs.npy exists ONLY remotely)
#
# This script ONLY syncs src/, rebuilds, and `up -d`s. It never touches the
# compose file, the mounted inputs, or any other service. Safe to re-run.
#
set -euo pipefail
IFS=$'\n\t'

# ---- configuration (edit here) ---------------------------------------------
readonly REMOTE_HOST="oraclearm3"                  # alias configured in ~/.ssh/config
readonly REMOTE_DIR="/home/ubuntu/docker/tau"      # holds docker-compose.yaml on the VPS
readonly REMOTE_SRC="${REMOTE_DIR}/src"            # rsync target = docker build context
readonly COMPOSE_FILE="${REMOTE_DIR}/docker-compose.yaml"
readonly CONTAINER="tau"
readonly SERVICE="tau"
readonly PUBLIC_URL="https://tau.filetki.de"
readonly EXPECTED_MEM=8589934592                  # 8g; a lower cap OOM-loops the precompute
readonly MIN_TREES=10                              # /api/trees must serve at least this many
readonly HEALTH_TIMEOUT=600                        # seconds; precompute takes minutes on 2 CPUs
readonly POLL_INTERVAL=15                          # seconds between health polls

# ---- resolve repo root = this script's directory ---------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
readonly SCRIPT_DIR
cd "$SCRIPT_DIR"

die() { echo "❌ ERROR: $*" >&2; exit 1; }
phase() { echo; echo "==> $*"; }

START_TS=$(date +%s)
START_ISO="$(date -u +%FT%TZ)"
readonly START_ISO

# Run a python one-liner inside the app container (it has no curl, but it has
# python + stdlib urllib). $1 is python source; avoid single quotes in it.
remote_py() {
    ssh "$REMOTE_HOST" "docker exec ${CONTAINER} python -c '$1'"
}

# ---- 1. preflight ----------------------------------------------------------
phase "Preflight: local git state (advisory only)"
if [ -n "$(git status --porcelain 2>/dev/null || true)" ]; then
    echo "⚠️  working tree is dirty — uncommitted changes WILL be synced to src/."
    git status --short | head -20
else
    echo "Working tree is clean."
fi

phase "Preflight: rsync present"
command -v rsync >/dev/null 2>&1 || die "rsync not found locally"
echo "rsync ok."

phase "Preflight: SSH to ${REMOTE_HOST}, dir ${REMOTE_DIR}, compose file present"
ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
    "test -d ${REMOTE_DIR} && test -f ${COMPOSE_FILE}" \
    || die "cannot reach ${REMOTE_HOST}, or ${REMOTE_DIR}/${COMPOSE_FILE} is missing"
echo "SSH ok; remote compose dir + file present."

# ---- 2. ship: rsync ./ -> src/ ---------------------------------------------
# --delete keeps src/ a faithful copy, but the deletes are scoped by the
# filters below: excluded/protected paths on the receiver are NEVER deleted.
# continuumabs.npy exists ONLY remotely (gitignored, generated there) — the
# exclude keeps it from being deleted AND from being overwritten by local
# absence; the explicit protect is belt-and-braces. Same for the compose file,
# which must survive even if someone stages one locally one day.
phase "Ship: rsync ./ -> ${REMOTE_HOST}:${REMOTE_SRC}/"
rsync -a --delete \
    -e "ssh -o ConnectTimeout=15" \
    --filter='protect docker-compose.yaml' \
    --filter='protect continuumabs.npy' \
    --exclude='docker-compose.yaml' \
    --exclude='.git/' \
    --exclude='.github/' \
    --exclude='.jj/' \
    --exclude='.venv/' \
    --exclude='.ruff_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='.mplconfig/' \
    --exclude='__pycache__/' \
    --exclude='*.py[oc]' \
    --exclude='*.swp' \
    --exclude='.DS_Store' \
    --exclude='.python-version' \
    --exclude='.env' \
    --exclude='.env.*' \
    --exclude='webapp.log' \
    --exclude='.webapp.pid' \
    --exclude='ODF_format.npy' \
    --exclude='ODF_nc_format.nc' \
    --exclude='continuumabs.dat' \
    --exclude='continuumabs.npy' \
    --exclude='continuumall.dat' \
    --exclude='continuumall.npy' \
    --exclude='continuumscat.dat' \
    --exclude='continuumscat.npy' \
    --exclude='data/' \
    --exclude='runs/' \
    --exclude='runs_ab/' \
    --exclude='plots*/' \
    --exclude='*.log' \
    --exclude='verify5g_*' \
    --exclude='opt_*.*' \
    --exclude='ab_*.log' \
    --exclude='paper/ms.tex' \
    --exclude='paper/*.aux' \
    --exclude='paper/*.bbl' \
    --exclude='paper/*.blg' \
    --exclude='paper/*.fdb_latexmk' \
    --exclude='paper/*.fls' \
    --exclude='paper/*.log' \
    --exclude='paper/*.out' \
    --exclude='paper/*.toc' \
    --exclude='paper/*.pdf' \
    --exclude='paper/*.synctex.gz' \
    ./ "${REMOTE_HOST}:${REMOTE_SRC}/" \
    || die "rsync to ${REMOTE_HOST}:${REMOTE_SRC}/ failed"
echo "Sync ok (exit 0)."

phase "Ship: confirm remote-only files survived the --delete sync"
ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
    "test -f ${REMOTE_DIR}/continuumabs.npy && test -f ${COMPOSE_FILE} && ls -la ${REMOTE_DIR}/continuumabs.npy ${COMPOSE_FILE}" \
    || die "remote-only continuumabs.npy or docker-compose.yaml went missing after rsync"
echo "Protected files intact."

# ---- 3. build + restart remotely -------------------------------------------
phase "Build + recreate on ${REMOTE_HOST} (docker compose up -d --build ${SERVICE})"
restarts_before="$(ssh "$REMOTE_HOST" \
    "docker inspect ${CONTAINER} --format '{{.RestartCount}}'" 2>/dev/null || echo '?')"
ssh "$REMOTE_HOST" "cd ${REMOTE_DIR} && docker compose up -d --build ${SERVICE}" \
    || die "docker compose up -d --build failed on ${REMOTE_HOST}"
echo "Container recreated (pre-build restarts: ${restarts_before})."

# ---- 4. verify readiness ---------------------------------------------------
phase "Verify: memory cap is still 8g (lower values OOM-loop the precompute)"
mem_bytes="$(ssh "$REMOTE_HOST" \
    "docker inspect ${CONTAINER} --format '{{.HostConfig.Memory}}'")"
echo "mem_limit = ${mem_bytes} bytes (expected ${EXPECTED_MEM})."
[ "$mem_bytes" = "$EXPECTED_MEM" ] \
    || die "mem_limit regressed to ${mem_bytes}; refusing to call this healthy (OOM risk)"

phase "Verify: poll in-container /api/init for HTTP 200 (precompute takes minutes)"
http_code="000"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while :; do
    http_code="$(remote_py 'import urllib.request;print(urllib.request.urlopen("http://localhost:8771/api/init",timeout=10).status)' \
        2>/dev/null || echo 000)"
    if [ "$http_code" = "200" ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "…still not ready after ${HEALTH_TIMEOUT}s (last HTTP ${http_code})."
        break
    fi
    echo "…HTTP ${http_code}, waiting ${POLL_INTERVAL}s…"
    sleep "$POLL_INTERVAL"
done
[ "$http_code" = "200" ] || die "/api/init never returned 200 within ${HEALTH_TIMEOUT}s"

phase "Verify: /api/trees serves >= ${MIN_TREES} trees"
n_trees="$(remote_py 'import urllib.request,json;print(len(json.load(urllib.request.urlopen("http://localhost:8771/api/trees",timeout=30))["trees"]))' \
    2>/dev/null || echo 0)"
echo "trees served: ${n_trees} (minimum ${MIN_TREES})."
[ "$n_trees" -ge "$MIN_TREES" ] || die "/api/trees serves only ${n_trees} trees"

# The recreate itself kills the old container with SIGKILL (exitCode=137), so a
# bare 137 match is meaningless. OOM is `oomKilled=true` on a die event, or an
# `oom` action (a kernel OOM also bumps RestartCount, checked above).
phase "Verify: container running, restarts stable, no OOM-kill since deploy start"
container_state="$(ssh "$REMOTE_HOST" \
    "docker inspect ${CONTAINER} --format 'status={{.State.Status}} restarts={{.RestartCount}}'")"
echo "container: ${container_state}."
case "$container_state" in
    status=running*) ;;
    *) die "container is not running: ${container_state}" ;;
esac
restarts_after="${container_state##*restarts=}"
if [ "$restarts_before" != "?" ] && [ "$restarts_after" != "$restarts_before" ] \
    && [ "$restarts_after" != "0" ]; then
    die "restart counter grew during deploy (${restarts_before} -> ${restarts_after})"
fi
oom_hits="$(ssh "$REMOTE_HOST" \
    "docker events --since ${START_ISO} --until \$(date -u +%FT%TZ) --filter container=${CONTAINER} --filter event=die --format '{{.Actor.Attributes}}' 2>/dev/null" \
    | grep -E 'oomKilled=true' || true)"
[ -z "$oom_hits" ] || die "OOM-killed container since deploy start: ${oom_hits}"
oom_actions="$(ssh "$REMOTE_HOST" \
    "docker events --since ${START_ISO} --until \$(date -u +%FT%TZ) --filter container=${CONTAINER} --filter event=oom 2>/dev/null" \
    || true)"
[ -z "$oom_actions" ] || die "kernel OOM event since deploy start: ${oom_actions}"
echo "No restarts growth, no OOM-kill."

# ---- summary ---------------------------------------------------------------
elapsed=$(( $(date +%s) - START_TS ))
echo
echo "=========================== DEPLOY SUMMARY ==========================="
echo "  Host:       ${REMOTE_HOST}  (${REMOTE_SRC}/ <- repo root)"
echo "  Container:  ${container_state}"
echo "  mem_limit:  ${mem_bytes} bytes"
echo "  Health:     HTTP ${http_code} @ ${CONTAINER}:http://localhost:8771/api/init"
echo "  Trees:      ${n_trees} via /api/trees"
echo "  URL:        ${PUBLIC_URL}"
echo "  Elapsed:    ${elapsed}s"
echo "  Result:     ✅ SUCCESS — app is serving."
echo "======================================================================"
