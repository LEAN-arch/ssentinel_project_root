#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Environments
# This script sets up a Python virtual environment and installs dependencies.
# PLATINUM STANDARD - SME OPTIMIZED

# --- Configuration ---
set -e # Exit immediately if a command exits with a non-zero status.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_NAME=".venv"
VENV_DIR="${PROJECT_ROOT}/${VENV_NAME}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
PACKAGES_FILE="${PROJECT_ROOT}/scripts/packages.txt"
PYTHON_EXEC="${SENTINEL_PYTHON_EXEC:-python3.9}"

# --- Helper Functions ---
log_info() { echo -e "\033[34m[INFO]\033[0m $1"; }
log_warn() { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1" >&2; }
log_success() { echo -e "\033[32m[SUCCESS]\033[0m $1"; }

exit_on_error() {
    log_error "$1"
    log_error "Setup aborted."
    exit "${2:-1}"
}

# --- Prerequisite Checks ---
log_info "Starting Sentinel Health Co-Pilot environment setup..."
log_info "Project Root: ${PROJECT_ROOT}"

log_info "Checking for Python interpreter: ${PYTHON_EXEC}..."
if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
    exit_on_error "Python interpreter '${PYTHON_EXEC}' not found. Please install Python 3.9+ or set SENTINEL_PYTHON_EXEC."
fi
log_info "Found: $(${PYTHON_EXEC} -V)"

log_info "Checking for Python 'venv' module..."
if ! "${PYTHON_EXEC}" -m venv -h >/dev/null 2>&1; then
    exit_on_error "'venv' module not found for ${PYTHON_EXEC}. On Debian/Ubuntu, try: sudo apt-get install python3.9-venv"
fi
log_info "'venv' module found."

# --- System Dependency Guidance ---
if [ -f "$PACKAGES_FILE" ]; then
    log_info "System-level libraries may be required by some Python packages."
    log_info "Refer to '${PACKAGES_FILE}' for details."
    if command -v apt-get >/dev/null 2>&1; then
        log_info "Example for Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y \$(cat ${PACKAGES_FILE})"
    elif command -v yum >/dev/null 2>&1; then
        log_info "Example for CentOS/RHEL: sudo yum install -y \$(cat ${PACKAGES_FILE})"
    fi
else
    log_warn "System dependency list '${PACKAGES_FILE}' not found. Some packages might require manual installation of system libraries."
fi

# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating Python virtual environment in ${VENV_DIR}..."
    "${PYTHON_EXEC}" -m venv "${VENV_DIR}" || exit_on_error "Failed to create virtual environment."
    log_success "Virtual environment created."
else
    log_info "Virtual environment already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate" || exit_on_error "Failed to activate virtual environment."
log_info "Virtual environment activated: $(which python)"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip..."
python -m pip install --upgrade pip || exit_on_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE}..."
    python -m pip install -r "${REQUIREMENTS_FILE}" || exit_on_error "Failed to install dependencies from ${REQUIREMENTS_FILE}."
    log_success "Python dependencies installed successfully."
else
    exit_on_error "Requirements file '${REQUIREMENTS_FILE}' not found."
fi

# --- Post-Setup Information ---
echo
log_success "=========================================================="
log_success "Python Environment Setup for Sentinel is Complete!"
log_success "=========================================================="
echo
log_info "To activate this environment in a new terminal, run:"
echo -e "  \033[1msource \"${VENV_DIR}/bin/activate\"\033[0m"
echo
log_info "To run the Streamlit application, run:"
echo -e "  \033[1mstreamlit run \"${PROJECT_ROOT}/app.py\"\033[0m"
echo

# --- .env File Check ---
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"
ACTUAL_ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ACTUAL_ENV_FILE" ]; then
    log_warn "IMPORTANT: Found '${ENV_EXAMPLE_FILE}'. Copy it to '.env' and add your Mapbox token:"
    echo -e "  \033[1mcp \"${ENV_EXAMPLE_FILE}\" \"${ACTUAL_ENV_FILE}\"\033[0m"
fi

exit 0
