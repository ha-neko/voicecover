#!/usr/bin/env bash
# Voicecover web stack manager. Interactive with no arguments; scriptable otherwise.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$ROOT/.dev-runtime"
LOG_DIR="$RUNTIME_DIR/logs"
LOCK_FILE="$RUNTIME_DIR/manager.lock"
PYTHON="$ROOT/.venv/bin/python"
WEB_HOST="${VOICECOVER_WEB_HOST:-0.0.0.0}"
WEB_PORT="${VOICECOVER_WEB_PORT:-8000}"
LLM_PORT="${VOICECOVER_LLM_PORT:-8081}"

SERVICES=(web llm tunnel)

if [[ -t 1 ]]; then
    RESET=$'\033[0m'; BOLD=$'\033[1m'; DIM=$'\033[2m'
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; CYAN=$'\033[36m'
else
    RESET=''; BOLD=''; DIM=''; GREEN=''; YELLOW=''; RED=''; CYAN=''
fi

mkdir -p -- "$LOG_DIR"

pid_file() { printf '%s/%s.pid' "$RUNTIME_DIR" "$1"; }
log_file() { printf '%s/%s.log' "$LOG_DIR" "$1"; }

service_label() {
    case "$1" in
        web) printf 'Voicecover Web' ;;
        llm) printf 'LLM Server' ;;
        tunnel) printf 'HTTPS Tunnel' ;;
    esac
}

tunnel_url() {
    local line url=''
    if [[ -f "$(log_file tunnel)" ]]; then
        while IFS= read -r line; do
            if [[ "$line" =~ https://[a-z0-9-]+\.trycloudflare\.com ]]; then
                url="${BASH_REMATCH[0]}"
            fi
        done < "$(log_file tunnel)"
    fi
    printf '%s' "${url:-waiting for URL}"
}

service_url() {
    case "$1" in
        web) printf 'http://127.0.0.1:%s' "$WEB_PORT" ;;
        llm) printf 'http://127.0.0.1:%s' "$LLM_PORT" ;;
        tunnel) tunnel_url ;;
    esac
}

read_pid() {
    local file pid _
    file="$(pid_file "$1")"
    [[ -f "$file" ]] || return 1
    read -r pid _ < "$file" || return 1
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s' "$pid"
}

process_start_time() {
    local pid="$1" stat rest fields
    [[ -r "/proc/$pid/stat" ]] || return 1
    IFS= read -r stat < "/proc/$pid/stat" || return 1
    rest="${stat##*) }"
    read -ra fields <<< "$rest"
    [[ "${fields[19]:-}" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "${fields[19]}"
}

write_pid_record() {
    local service="$1" pid="$2" started tmp
    started="$(process_start_time "$pid")" || return 1
    tmp="$(pid_file "$service").tmp.$$"
    printf '%s %s\n' "$pid" "$started" > "$tmp"
    mv -f -- "$tmp" "$(pid_file "$service")"
}

recorded_process_alive() {
    local service="$1" pid recorded_pid recorded_start current_start
    pid="$(read_pid "$service")" || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    read -r recorded_pid recorded_start < "$(pid_file "$service")" || return 1
    [[ "$recorded_pid" == "$pid" && "${recorded_start:-}" =~ ^[0-9]+$ ]] || return 1
    current_start="$(process_start_time "$pid")" || return 1
    [[ "$recorded_start" == "$current_start" ]]
}

process_matches_service() {
    local service="$1" pid="$2" command
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    case "$service" in
        web) [[ "$command" == *uvicorn*voicecover.web:app*"--port $WEB_PORT"* ]] ;;
        llm) [[ "$command" == *llama-server*"--port $LLM_PORT"* ]] ;;
        tunnel) [[ "$command" == *cloudflared*tunnel*"http://127.0.0.1:$WEB_PORT"* ]] ;;
        *) return 1 ;;
    esac
}

discover_service_pid() {
    local service="$1" path pid
    for path in /proc/[1-9]*/cmdline; do
        pid="${path#/proc/}"
        pid="${pid%/cmdline}"
        process_matches_service "$service" "$pid" && { printf '%s' "$pid"; return 0; }
    done
    return 1
}

managed_running() {
    local service="$1" pid
    recorded_process_alive "$service" || return 1
    pid="$(read_pid "$service")" || return 1
    process_matches_service "$service" "$pid"
}

listener_owned_by_web() {
    local pid pgid output candidate candidate_pgid
    managed_running web || return 1
    pid="$(read_pid web)" || return 1
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null)"
    pgid="${pgid//[[:space:]]/}"
    output="$(ss -H -ltnp "sport = :$WEB_PORT" 2>/dev/null)"
    while [[ "$output" =~ pid=([0-9]+) ]]; do
        candidate="${BASH_REMATCH[1]}"
        candidate_pgid="$(ps -o pgid= -p "$candidate" 2>/dev/null)"
        candidate_pgid="${candidate_pgid//[[:space:]]/}"
        [[ "$candidate_pgid" == "$pgid" ]] && return 0
        output="${output#*pid=$candidate}"
    done
    return 1
}

web_responding() {
    curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$WEB_PORT/" 2>/dev/null
}

llm_responding() {
    # llama-server answers /health with 503 until the model has finished loading
    curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$LLM_PORT/health" 2>/dev/null
}

tunnel_healthy() {
    managed_running tunnel && web_responding && [[ "$(tunnel_url)" != 'waiting for URL' ]]
}

service_healthy() {
    case "$1" in
        web) listener_owned_by_web && web_responding ;;
        llm) llm_responding ;;
        tunnel) tunnel_healthy ;;
    esac
}

service_responding() {
    case "$1" in
        web) web_responding ;;
        llm) llm_responding ;;
        tunnel) [[ "$(tunnel_url)" != 'waiting for URL' ]] && managed_running tunnel ;;
    esac
}

cleanup_stale_pid() {
    local service="$1"
    [[ -f "$(pid_file "$service")" ]] && ! managed_running "$service" && rm -f -- "$(pid_file "$service")"
    return 0
}

launch() {
    local service="$1" pid i
    shift
    rm -f -- "$(log_file "$service")"
    : > "$(log_file "$service")"
    nohup setsid "$@" 9>&- >> "$(log_file "$service")" 2>&1 < /dev/null &
    pid=$!
    for ((i=0; i<20; i++)); do
        write_pid_record "$service" "$pid" && return 0
        sleep .01
    done
    return 1
}

signal_managed() {
    local service="$1" signal="$2" pid pgid
    managed_running "$service" || return 1
    pid="$(read_pid "$service")" || return 1
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null)"
    pgid="${pgid//[[:space:]]/}"
    if [[ "$pgid" == "$pid" ]]; then
        kill "-$signal" -- "-$pid" 2>/dev/null
    else
        kill "-$signal" "$pid" 2>/dev/null
    fi
}

wait_ready() {
    local service="$1" i attempts=60
    # loading a multi-gigabyte model into memory can take a while
    [[ "$service" == llm ]] && attempts=240
    for ((i=0; i<attempts; i++)); do
        service_healthy "$service" && return 0
        recorded_process_alive "$service" || return 1
        sleep .25
    done
    return 1
}

cloudflared_bin() {
    command -v cloudflared 2>/dev/null || return 1
}

start_one() {
    local service="$1" label binary
    label="$(service_label "$service")"
    cleanup_stale_pid "$service"
    if managed_running "$service"; then
        printf '  %s●%s %-16s already running\n' "$GREEN" "$RESET" "$label"
        return 0
    fi
    if [[ "$service" == web ]] && web_responding; then
        printf '  %s◆%s %-16s port used by an external process\n' "$YELLOW" "$RESET" "$label"
        return 1
    fi
    if [[ "$service" == llm ]] && llm_responding; then
        printf '  %s◆%s %-16s port used by an external process\n' "$YELLOW" "$RESET" "$label"
        return 1
    fi
    if [[ "$service" == tunnel ]] && ! service_healthy web; then
        printf '  %s×%s %-16s web must be managed and healthy first\n' "$RED" "$RESET" "$label"
        return 1
    fi

    printf '  %s…%s starting %-16s' "$CYAN" "$RESET" "$label"
    case "$service" in
        web)
            [[ -x "$PYTHON" ]] || { printf '\r  %s×%s %-16s virtualenv missing\n' "$RED" "$RESET" "$label"; return 1; }
            PYTHONUNBUFFERED=1 \
            VOICECOVER_LOG_DIR="$LOG_DIR" \
            VOICECOVER_WEB_WORKERS="${VOICECOVER_WEB_WORKERS:-4}" \
            VOICECOVER_MAX_CONCURRENT_JOBS="${VOICECOVER_MAX_CONCURRENT_JOBS:-3}" \
            launch web "$PYTHON" -m uvicorn voicecover.web:app --host "$WEB_HOST" --port "$WEB_PORT"
            ;;
        llm)
            launch llm bash "$ROOT/scripts/llm-server.sh"
            ;;
        tunnel)
            binary="$(cloudflared_bin)" || { printf '\r  %s×%s %-16s cloudflared missing\n' "$RED" "$RESET" "$label"; return 1; }
            launch tunnel "$binary" tunnel --url "http://127.0.0.1:$WEB_PORT" --no-autoupdate
            ;;
    esac
    if wait_ready "$service"; then
        printf '\r  %s●%s %-16s %s\n' "$GREEN" "$RESET" "$label" "$(service_url "$service")"
    else
        printf '\r  %s×%s %-16s failed, see %s\n' "$RED" "$RESET" "$label" "$(log_file "$service")"
        signal_managed "$service" TERM || true
        rm -f -- "$(pid_file "$service")"
        return 1
    fi
}

adopt_one() {
    local service="$1" label pid source_log
    label="$(service_label "$service")"
    cleanup_stale_pid "$service"
    if managed_running "$service"; then
        printf '  %s●%s %-16s already managed\n' "$GREEN" "$RESET" "$label"
        return 0
    fi
    pid="$(discover_service_pid "$service")" || {
        printf '  %s×%s %-16s no matching process found\n' "$RED" "$RESET" "$label"
        return 1
    }
    write_pid_record "$service" "$pid" || return 1
    source_log="$(readlink -f "/proc/$pid/fd/1" 2>/dev/null)" || source_log=''
    if [[ -f "$source_log" && "$source_log" != "$(log_file "$service")" ]]; then
        rm -f -- "$(log_file "$service")"
        ln -s -- "$source_log" "$(log_file "$service")"
    fi
    if service_healthy "$service"; then
        printf '  %s●%s %-16s adopted pid %s · %s\n' "$GREEN" "$RESET" "$label" "$pid" "$(service_url "$service")"
    else
        rm -f -- "$(pid_file "$service")"
        printf '  %s×%s %-16s process is not healthy\n' "$RED" "$RESET" "$label"
        return 1
    fi
}

stop_one() {
    local service="$1" label i
    label="$(service_label "$service")"
    cleanup_stale_pid "$service"
    if ! managed_running "$service"; then
        printf '  %s○%s %-16s already stopped\n' "$DIM" "$RESET" "$label"
        return 0
    fi
    [[ "$service" == web ]] && managed_running tunnel && stop_one tunnel
    printf '  %s…%s stopping %-16s' "$YELLOW" "$RESET" "$label"
    signal_managed "$service" TERM || true
    for ((i=0; i<40; i++)); do
        recorded_process_alive "$service" || break
        sleep .25
    done
    recorded_process_alive "$service" && signal_managed "$service" KILL || true
    if recorded_process_alive "$service"; then
        printf '\r  %s×%s %-16s could not terminate\n' "$RED" "$RESET" "$label"
        return 1
    fi
    rm -f -- "$(pid_file "$service")"
    printf '\r  %s○%s %-16s stopped\n' "$DIM" "$RESET" "$label"
}

validate_service() {
    [[ "$1" == all || "$1" == web || "$1" == llm || "$1" == tunnel ]] || {
        printf 'unknown service: %s\nvalid services: all, web, llm, tunnel\n' "$1" >&2
        return 2
    }
}

start_target() {
    local target="${1:-all}"
    validate_service "$target" || return
    if [[ "$target" == all ]]; then
        printf '\n%sStarting Voicecover web stack%s\n' "$BOLD" "$RESET"
        # the LLM server warms up in parallel; web jobs that need it can wait
        start_one llm || true
        start_one web && start_one tunnel
    else
        start_one "$target"
    fi
}

adopt_target() {
    local target="${1:-all}" rc=0
    validate_service "$target" || return
    printf '\n%sAdopting Voicecover web stack%s\n' "$BOLD" "$RESET"
    if [[ "$target" == all ]]; then
        adopt_one web || rc=1
        adopt_one llm || rc=1
        adopt_one tunnel || rc=1
    else
        adopt_one "$target" || rc=1
    fi
    return "$rc"
}

stop_target() {
    local target="${1:-all}" rc=0
    validate_service "$target" || return
    if [[ "$target" == all ]]; then
        printf '\n%sStopping Voicecover web stack%s\n' "$BOLD" "$RESET"
        stop_one tunnel || rc=1
        stop_one web || rc=1
        stop_one llm || rc=1
        return "$rc"
    fi
    stop_one "$target"
}

restart_target() {
    local target="${1:-all}" restore_tunnel=0 rc=0
    managed_running tunnel && [[ "$target" == all || "$target" == web ]] && restore_tunnel=1
    stop_target "$target" || rc=$?
    start_target "$target" || rc=$?
    ((restore_tunnel)) && ! managed_running tunnel && start_one tunnel || true
    return "$rc"
}

status_line() {
    local service="$1" state detail
    if managed_running "$service"; then
        service_healthy "$service" && state="${GREEN}RUNNING${RESET}" || state="${YELLOW}DEGRADED${RESET}"
        detail="pid $(read_pid "$service") · $(service_url "$service")"
    elif service_responding "$service"; then
        state="${YELLOW}EXTERNAL${RESET}"
        detail="responding, not managed by dev.sh"
    else
        state="${DIM}STOPPED${RESET}"
        detail="$(service_url "$service")"
    fi
    printf '  %-16s %-18b %s\n' "$(service_label "$service")" "$state" "$detail"
}

show_status() {
    printf '\n%sVoicecover web stack%s\n\n' "$BOLD" "$RESET"
    status_line web
    status_line llm
    status_line tunnel
    if web_responding && [[ -x "$PYTHON" ]]; then
        local summary
        summary="$(curl -fsS --max-time 2 "http://127.0.0.1:$WEB_PORT/api/status" 2>/dev/null | \
            "$PYTHON" -c 'import json, sys; d=json.load(sys.stdin); print("{} running · {} queued · {} completed · {} failed".format(d.get("running", 0), d.get("queued", 0), d.get("completed", 0), d.get("failed", 0)))' 2>/dev/null)" || true
        [[ -n "$summary" ]] && printf '  %-16s %s\n' "Jobs" "$summary"
    fi
    printf '\n  logs: %s\n' "$LOG_DIR"
}

show_logs() {
    local service="${1:-web}"
    validate_service "$service" || return
    if [[ "$service" == all ]]; then
        show_logs web
        show_logs llm
        show_logs tunnel
        return
    fi
    touch "$(log_file "$service")"
    printf '%sFollowing %s. Ctrl-C returns.%s\n' "$DIM" "$(log_file "$service")" "$RESET"
    tail -n 80 -f "$(log_file "$service")"
}

with_manager_lock() {
    (
        flock -w 60 9 || { printf 'could not acquire manager lock\n' >&2; return 75; }
        "$@"
    ) 9>"$LOCK_FILE"
}

tui() {
    if [[ ! -t 0 || ! -t 1 ]]; then
        printf 'interactive dashboard requires a terminal; use ./dev.sh status\n' >&2
        return 1
    fi
    exec "$PYTHON" "$ROOT/dev_tui.py"
}

usage() {
    cat <<'EOF'
Usage:
  ./dev.sh                         interactive TUI
  ./dev.sh start [web|llm|tunnel|all]
  ./dev.sh adopt [all|web|llm|tunnel]
  ./dev.sh stop [all|web|llm|tunnel]
  ./dev.sh restart [all|web|llm|tunnel]
  ./dev.sh status
  ./dev.sh logs [all|web|llm|tunnel]
EOF
}

case "${1:-tui}" in
    tui) tui ;;
    start) with_manager_lock start_target "${2:-web}" ;;
    adopt) with_manager_lock adopt_target "${2:-all}" ;;
    stop) with_manager_lock stop_target "${2:-all}" ;;
    restart) with_manager_lock restart_target "${2:-all}" ;;
    status) show_status ;;
    logs) show_logs "${2:-web}" ;;
    help|-h|--help) usage ;;
    *) usage; exit 2 ;;
esac
