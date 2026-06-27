#!/bin/sh
# FUSION installer — downloads the right prebuilt binary for your platform and
# installs it onto your PATH so you can run `fusion` from anywhere.
#
#   curl -fsSL https://raw.githubusercontent.com/teddytennant/FUSION/master/install.sh | sh
#
# Environment overrides:
#   FUSION_INSTALL_DIR   install location (default: /usr/local/bin if writable,
#                        otherwise ~/.local/bin)
#   FUSION_VERSION       release tag to install (default: latest)
set -eu

REPO="teddytennant/FUSION"

err() {
	printf 'error: %s\n' "$1" >&2
	exit 1
}

# --- detect platform -------------------------------------------------------
os=$(uname -s)
arch=$(uname -m)

case "$os" in
Linux) os_part=linux ;;
Darwin) os_part=macos ;;
*) err "unsupported OS '$os' (FUSION ships Linux and macOS binaries)" ;;
esac

case "$arch" in
x86_64 | amd64) arch_part=x86_64 ;;
aarch64 | arm64) arch_part=aarch64 ;;
*) err "unsupported architecture '$arch'" ;;
esac

asset="fusion-${os_part}-${arch_part}"

# --- resolve download URL --------------------------------------------------
version="${FUSION_VERSION:-latest}"
if [ "$version" = latest ]; then
	url="https://github.com/${REPO}/releases/latest/download/${asset}"
else
	url="https://github.com/${REPO}/releases/download/${version}/${asset}"
fi

# --- choose an install dir on PATH -----------------------------------------
if [ -n "${FUSION_INSTALL_DIR:-}" ]; then
	install_dir="$FUSION_INSTALL_DIR"
elif [ -w /usr/local/bin ] 2>/dev/null; then
	install_dir="/usr/local/bin"
else
	install_dir="$HOME/.local/bin"
fi
mkdir -p "$install_dir"

target="${install_dir}/fusion"

# --- download --------------------------------------------------------------
printf 'Downloading %s -> %s\n' "$asset" "$target"
if command -v curl >/dev/null 2>&1; then
	curl -fsSL "$url" -o "$target" || err "download failed from $url"
elif command -v wget >/dev/null 2>&1; then
	wget -qO "$target" "$url" || err "download failed from $url"
else
	err "need curl or wget to download the binary"
fi
chmod +x "$target"

printf 'Installed fusion to %s\n' "$target"

# --- PATH check ------------------------------------------------------------
case ":${PATH}:" in
*":${install_dir}:"*)
	printf 'Run `fusion --onboard` to get started.\n'
	;;
*)
	printf '\n%s is not on your PATH. Add it, e.g.:\n' "$install_dir"
	printf '  echo '\''export PATH="%s:$PATH"'\'' >> ~/.profile\n' "$install_dir"
	printf 'Then restart your shell and run `fusion --onboard`.\n'
	;;
esac
