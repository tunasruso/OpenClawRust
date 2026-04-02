#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/tunasruso/OpenClawRust.git}"
INSTALL_ROOT="${INSTALL_ROOT:-$HOME/.local/share/openclawrust}"
RUST_DIR="$INSTALL_ROOT/rust"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"
PROFILE_FILE="${PROFILE_FILE:-$HOME/.profile}"
BUILD_PROFILE="${BUILD_PROFILE:-release}"

log() {
  printf '[install] %s\n' "$*"
}

fail() {
  printf '[install] error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif need_cmd sudo; then
    sudo "$@"
  else
    fail "need root privileges for: $*"
  fi
}

detect_package_manager() {
  if need_cmd apt-get; then
    echo apt
    return
  fi
  if need_cmd dnf; then
    echo dnf
    return
  fi
  if need_cmd yum; then
    echo yum
    return
  fi
  if need_cmd pacman; then
    echo pacman
    return
  fi
  if need_cmd zypper; then
    echo zypper
    return
  fi
  echo unknown
}

install_system_packages() {
  local manager
  manager="$(detect_package_manager)"

  case "$manager" in
    apt)
      run_as_root apt-get update
      run_as_root apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        pkg-config \
        libssl-dev \
        unzip
      ;;
    dnf)
      run_as_root dnf install -y \
        gcc \
        gcc-c++ \
        make \
        ca-certificates \
        curl \
        git \
        pkgconf-pkg-config \
        openssl-devel \
        unzip
      ;;
    yum)
      run_as_root yum install -y \
        gcc \
        gcc-c++ \
        make \
        ca-certificates \
        curl \
        git \
        pkgconfig \
        openssl-devel \
        unzip
      ;;
    pacman)
      run_as_root pacman -Sy --noconfirm \
        base-devel \
        ca-certificates \
        curl \
        git \
        pkgconf \
        openssl \
        unzip
      ;;
    zypper)
      run_as_root zypper --non-interactive install \
        gcc \
        gcc-c++ \
        make \
        ca-certificates \
        curl \
        git \
        pkg-config \
        libopenssl-devel \
        unzip
      ;;
    *)
      fail "unsupported package manager; install git curl build tools pkg-config openssl headers unzip manually"
      ;;
  esac
}

ensure_rustup() {
  if need_cmd cargo && need_cmd rustc; then
    log "Rust toolchain already installed"
    return
  fi

  log "Installing Rust toolchain via rustup"
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y
}

load_rust_env() {
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  need_cmd cargo || fail "cargo not found after rustup install"
}

sync_repo() {
  mkdir -p "$INSTALL_ROOT"

  if [ -d "$INSTALL_ROOT/.git" ]; then
    log "Updating existing repository in $INSTALL_ROOT"
    git -C "$INSTALL_ROOT" fetch --depth=1 origin main
    git -C "$INSTALL_ROOT" reset --hard origin/main
  else
    log "Cloning repository into $INSTALL_ROOT"
    rm -rf "$INSTALL_ROOT"
    git clone "$REPO_URL" "$INSTALL_ROOT"
  fi
}

build_binary() {
  log "Building claw (${BUILD_PROFILE})"
  if [ "$BUILD_PROFILE" = "release" ]; then
    cargo build --release -p claw-cli --manifest-path "$RUST_DIR/Cargo.toml"
  else
    cargo build -p claw-cli --manifest-path "$RUST_DIR/Cargo.toml"
  fi
}

install_binary() {
  local source_bin
  if [ "$BUILD_PROFILE" = "release" ]; then
    source_bin="$RUST_DIR/target/release/claw"
  else
    source_bin="$RUST_DIR/target/debug/claw"
  fi

  [ -x "$source_bin" ] || fail "expected binary not found: $source_bin"

  mkdir -p "$BIN_DIR"
  install -m 0755 "$source_bin" "$BIN_DIR/claw"
  log "Installed binary to $BIN_DIR/claw"
}

ensure_path_hint() {
  case ":$PATH:" in
    *":$BIN_DIR:"*)
      return
      ;;
  esac

  if [ -f "$PROFILE_FILE" ] && grep -Fqs "$BIN_DIR" "$PROFILE_FILE"; then
    return
  fi

  log "Adding $BIN_DIR to PATH in $PROFILE_FILE"
  mkdir -p "$(dirname "$PROFILE_FILE")"
  {
    printf '\n# OpenClawRust\n'
    printf 'export PATH="%s:$PATH"\n' "$BIN_DIR"
  } >>"$PROFILE_FILE"
}

write_env_example() {
  local env_file="$INSTALL_ROOT/.env.example"
  cat >"$env_file" <<'EOF'
# Anthropic-compatible backend
# export ANTHROPIC_API_KEY="your-key"
# export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# OpenAI-compatible backend
# export OPENAI_API_KEY="your-key"
# export OPENAI_BASE_URL="https://api.openai.com/v1"

# xAI-compatible backend
# export XAI_API_KEY="your-key"
# export XAI_BASE_URL="https://api.x.ai/v1"

# Experimental ChatGPT OAuth flow for `claw login openai-chatgpt`
# export OPENAI_CHATGPT_CLIENT_ID="..."
# export OPENAI_CHATGPT_AUTHORIZE_URL="..."
# export OPENAI_CHATGPT_TOKEN_URL="..."
# export OPENAI_CHATGPT_CALLBACK_PORT="4545"
# export OPENAI_CHATGPT_SCOPES="openid profile email"
EOF
  log "Wrote environment template to $env_file"
}

print_next_steps() {
  cat <<EOF

Install complete.

Repo:      $INSTALL_ROOT
Binary:    $BIN_DIR/claw
Env file:  $INSTALL_ROOT/.env.example

Next steps:
  1. Reload your shell: source "$PROFILE_FILE"
  2. Export provider credentials
  3. Run: claw --help
  4. Start a session: claw
  5. One-shot prompt: claw prompt "summarize this repository"
EOF
}

main() {
  log "Installing Linux dependencies"
  install_system_packages
  ensure_rustup
  load_rust_env
  sync_repo
  build_binary
  install_binary
  ensure_path_hint
  write_env_example
  print_next_steps
}

main "$@"
