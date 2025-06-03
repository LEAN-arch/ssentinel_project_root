#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies
# for Web Dashboards, backend services, and Dev/Simulation.

echo "======================================================================"
echo "Setting up Sentinel Health Co-Pilot Python Virtual Environment..."
echo "Target: Python Web Dashboards, Dev/Simulation Environment."
echo "======================================================================"
echo

# --- Configuration ---
# Determine Project Root: Assume script is in sentinel_project_root/scripts/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" || { echo "Failed to determine script directory"; exit 1; }
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")" # Should resolve to sentinel_project_root

VENV_NAME="${SENTINEL_VENV_NAME:-.venv_sentinel_py}" # Customizable via SENTINEL_VENV_NAME
VENV_DIR="${PROJECT_ROOT}/${VENV_NAME}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
SYSTEM_PACKAGES_FILE="${PROJECT_ROOT}/packages.txt"
PYTHON_EXEC="${SENTINEL_PYTHON_EXEC:-python3}" # Customizable via SENTINEL_PYTHON_EXEC
MIN_PYTHON_VERSION="3.8"

# --- Helper Functions ---
log_info() { echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2; }

exit_on_error() {
    # $1: Error message
    # $2: Exit code (default: 1)
    local exit_code="${2:-1}"
    log_error "$1"
    log_error "Setup aborted. Exit code: $exit_code"
    # Deactivate venv if active
    if [ -n "$VIRTUAL_ENV" ]; then
        log_info "Deactivating virtual environment..."
        deactivate 2>/dev/null
    fi
    exit "$exit_code"
}

# --- Pre-requisite Checks ---
log_info "Using Project Root: ${PROJECT_ROOT}"
log_info "Checking for Python interpreter: ${PYTHON_EXEC}..."
if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
    exit_on_error "Python interpreter '${PYTHON_EXEC}' not found. Install Python ${MIN_PYTHON_VERSION}+."
fi

# Get Python version
PYTHON_VERSION=$("${PYTHON_EXEC}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") || \
    exit_on_error "Failed to retrieve Python version."
log_info "Found Python version: ${PYTHON_VERSION}"

# Check minimum Python version
if ! "${PYTHON_EXEC}" -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    log_warn "Python ${PYTHON_VERSION} is older than recommended ${MIN_PYTHON_VERSION}. Dependencies may fail."
    if [ "${SENTINEL_IGNORE_PY_VERSION:-0}" != "1" ]; then
        exit_on_error "Python version too old. Set SENTINEL_IGNORE_PY_VERSION=1 to bypass."
    fi
fi

log_info "Checking for Python 'venv' module..."
if ! "${PYTHON_EXEC}" -m venv -h >/dev/null 2>&1; then
    exit_on_error "'venv' module not found for ${PYTHON_EXEC}. Install 'python3-venv' package."
fi
log_info "'venv' module found."

# --- System Dependency Guidance ---
if [ -f "$SYSTEM_PACKAGES_FILE" ]; then
    log_info "----------------------------------------------------------------------"
    log_info "System-level libraries may be required (see ${SYSTEM_PACKAGES_FILE})."
    if [ -f /etc/debian_version ]; then
        log_info "On Debian/Ubuntu, install with: sudo apt-get update && sudo apt-get install -y $(grep -v '^#' "${SYSTEM_PACKAGES_FILE}")"
    elif [ -f /etc/redhat-release ]; then
        log_info "On CentOS/RHEL, install with: sudo yum install -y $(grep -v '^#' "${SYSTEM_PACKAGES_FILE}")"
    elif [ "$(uname)" = "Darwin" ]; then
        log_info "On macOS, install with: brew install $(grep -v '^#' "${SYSTEM_PACKAGES_FILE}")"
    else
        log_info "Consult ${SYSTEM_PACKAGES_FILE} for system dependencies and install manually."
    fi
    log_info "----------------------------------------------------------------------"
else
    log_warn "${SYSTEM_PACKAGES_FILE} not found. System dependencies may be required."
fi

# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating Python virtual environment in ${VENV_DIR}..."
    "${PYTHON_EXEC}" -m venv "${VENV_DIR}" || exit_on_error "Failed to create virtual environment at '${VENV_DIR}'."
    log_info "Virtual environment created successfully."
else
    log_info "Virtual environment ${VENV_DIR} already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
log_info "Activating Python virtual environment: source ${VENV_DIR}/bin/activate"
# shellcheck disable=SC1091
if ! source "${VENV_DIR}/bin/activate"; then
    exit_on_error "Failed to activate virtual environment. Try manually: source ${VENV_DIR}/bin/activate"
fi

# Verify activation
if [ -z "$VIRTUAL_ENV" ] || [ "$(cd "$VIRTUAL_ENV" && pwd)" != "$(cd "$VENV_DIR" && pwd)" ]; then
    exit_on_error "Virtual environment activation failed. Expected VIRTUAL_ENV=${VENV_DIR}, got ${VIRTUAL_ENV:-Not set}."
fi
log_info "Virtual environment activated: $VIRTUAL_ENV"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
python -m pip install --upgrade pip || exit_on_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE}..."
    if grep -qi "geopandas" "$REQUIREMENTS_FILE"; then
        log_warn "----------------------------------------------------------------------"
        log_warn "'geopandas' found in ${REQUIREMENTS_FILE}."
        log_warn "This project has removed GeoPandas dependency. Consider removing it."
        log_warn "Continuing installation, but issues may arise."
        log_warn "----------------------------------------------------------------------"
    fi

    if grep -qE "fiona|pyproj|shapely|rtree|rasterio" "$REQUIREMENTS_FILE"; then
        log_info "----------------------------------------------------------------------"
        log_info "Geospatial libraries detected (e.g., fiona, pyproj, shapely, rasterio)."
        log_info "Ensure system dependencies (e.g., GEOS, PROJ) are installed."
        log_info "----------------------------------------------------------------------"
    fi

    python -m pip install -r "${REQUIREMENTS_FILE}" || exit_on_error "Failed to install dependencies from ${REQUIREMENTS_FILE}."
    log_info "Python dependencies installed successfully."
else
    exit_on_error "Requirements file '${REQUIREMENTS_FILE}' not found."
fi

# --- Post-Setup Information ---
echo
log_info "=========================================================================="
log_info "Python Environment Setup for Sentinel Health Co-Pilot Complete!"
log_info "=========================================================================="
log_info "Virtual Environment Path: ${VENV_DIR}"
echo
log_info "To activate this environment: source ${VENV_DIR}/bin/activate"
log_info "To run the Streamlit app: cd ${PROJECT_ROOT} && streamlit run app.py"
log_info "To deactivate the environment: deactivate"
echo

# Check for .env.example
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"
ACTUAL_ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ACTUAL_ENV_FILE" ]; then
    log_warn "----------------------------------------------------------------------"
    log_warn "Found ${ENV_EXAMPLE_FILE}. Copy it to ${ACTUAL_ENV_FILE} and customize:"
    log_warn "  cp ${ENV_EXAMPLE_FILE} ${ACTUAL_ENV_FILE}"
    log_warn "Edit ${ACTUAL_ENV_FILE} with environment variables (e.g., MAPBOX_ACCESS_TOKEN)."
    log_warn "----------------------------------------------------------------------"
fi

exit 0
