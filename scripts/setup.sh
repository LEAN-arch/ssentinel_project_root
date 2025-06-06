#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies.

echo "======================================================================"
echo "Setting up Sentinel Health Co-Pilot Python Virtual Environment..."
echo "======================================================================"
echo

# --- Configuration ---
SCRIPT_DIR_RAW="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to determine script directory. Aborting." >&2
    exit 1
fi
SCRIPT_DIR="${SCRIPT_DIR_RAW}"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VENV_NAME="${SENTINEL_VENV_NAME:-.venv_sentinel_py}"
VENV_DIR="${PROJECT_ROOT}/${VENV_NAME}"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
PACKAGES_FILE="${PROJECT_ROOT}/packages.txt" # Corrected variable name
PYTHON_EXEC="${SENTINEL_PYTHON_EXEC:-python3}"
MIN_PYTHON_VERSION_MAJOR=3
MIN_PYTHON_VERSION_MINOR=8

# --- Helper Functions ---
log_info() { echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2; }

exit_on_error() {
    local msg="${1:-"Unknown error occurred"}"
    local exit_code="${2:-1}"
    log_error "$msg"
    log_error "Setup aborted. Exit code: $exit_code"
    if [ -n "$VIRTUAL_ENV" ]; then
        log_info "Attempting to deactivate virtual environment..."
        deactivate >/dev/null 2>&1
    fi
    exit "$exit_code"
}

# --- Pre-requisite Checks ---
log_info "Using Project Root: ${PROJECT_ROOT}"
log_info "Checking for Python interpreter: ${PYTHON_EXEC}..."
if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
    exit_on_error "Python interpreter '${PYTHON_EXEC}' not found. Please install Python ${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR} or newer."
fi

PYTHON_VERSION_FULL=$("${PYTHON_EXEC}" -V 2>&1)
PYTHON_VERSION_MAJOR=$("${PYTHON_EXEC}" -c "import sys; print(sys.version_info.major)")
PYTHON_VERSION_MINOR=$("${PYTHON_EXEC}" -c "import sys; print(sys.version_info.minor)")

if [ -z "$PYTHON_VERSION_MAJOR" ] || [ -z "$PYTHON_VERSION_MINOR" ]; then
    exit_on_error "Failed to retrieve Python version components from ${PYTHON_VERSION_FULL}."
fi
log_info "Found Python version: ${PYTHON_VERSION_FULL} (Major: ${PYTHON_VERSION_MAJOR}, Minor: ${PYTHON_VERSION_MINOR})"

if [ "$PYTHON_VERSION_MAJOR" -lt "$MIN_PYTHON_VERSION_MAJOR" ] || \
   ( [ "$PYTHON_VERSION_MAJOR" -eq "$MIN_PYTHON_VERSION_MAJOR" ] && \
     [ "$PYTHON_VERSION_MINOR" -lt "$MIN_PYTHON_VERSION_MINOR" ] ); then
    log_warn "Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} is older than recommended ${MIN_PYTHON_VERSION_MAJOR}.${MIN_PYTHON_VERSION_MINOR}. Dependencies may fail."
    if [ "${SENTINEL_IGNORE_PY_VERSION:-0}" != "1" ]; then
        exit_on_error "Python version too old. Set SENTINEL_IGNORE_PY_VERSION=1 to bypass this check."
    else
        log_warn "SENTINEL_IGNORE_PY_VERSION=1 is set. Proceeding with older Python version at your own risk."
    fi
fi

log_info "Checking for Python 'venv' module..."
if ! "${PYTHON_EXEC}" -m venv -h >/dev/null 2>&1; then
    # Attempt to guide user for common systems
    distro_info=""
    if [ -f /etc/os-release ]; then
        distro_info=$(grep PRETTY_NAME /etc/os-release | cut -d'=' -f2 | tr -d '"')
    fi
    exit_on_error "'venv' module not found for ${PYTHON_EXEC}. On ${distro_info:-your system}, you might need to install a package like 'python3-venv' or 'python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}-venv'."
fi
log_info "'venv' module found."

# --- System Dependency Guidance ---
# CORRECTED: Use the correct variable name ${PACKAGES_FILE}
if [ -f "$PACKAGES_FILE" ]; then
    log_info "----------------------------------------------------------------------"
    log_info "System-level libraries may be required for some Python packages."
    log_info "Refer to the contents of '${PACKAGES_FILE}' for details."
    log_info "Example installation commands based on detected OS:"
    if grep -qE "(Debian|Ubuntu|Mint)" /etc/os-release 2>/dev/null; then
        log_info "  Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y \$(cat ${PACKAGES_FILE} | grep -v '^#')"
    elif grep -qE "(CentOS|RHEL|Fedora)" /etc/os-release 2>/dev/null; then # Fedora uses dnf
        log_info "  CentOS/RHEL:   sudo yum install -y \$(cat ${PACKAGES_FILE} | grep -v '^#')"
        log_info "  Fedora:        sudo dnf install -y \$(cat ${PACKAGES_FILE} | grep -v '^#')"
    elif [ "$(uname)" = "Darwin" ]; then
        log_info "  macOS (Homebrew): brew install \$(cat ${PACKAGES_FILE} | grep -v '^#')"
    else
        log_info "  Please consult '${PACKAGES_FILE}' and install dependencies manually for your OS."
    fi
    log_info "----------------------------------------------------------------------"
else
    log_warn "System dependency list file '${PACKAGES_FILE}' not found. Some Python packages might require manual installation of system libraries."
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
ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
log_info "Activating Python virtual environment: source ${ACTIVATE_SCRIPT}"
# shellcheck disable=SC1090
if ! source "${ACTIVATE_SCRIPT}"; then
    exit_on_error "Failed to activate virtual environment. Try manually: source ${ACTIVATE_SCRIPT}"
fi

# Verify activation robustly
if [ -z "$VIRTUAL_ENV" ] || [ "$(cd "$VIRTUAL_ENV" && pwd)" != "$(cd "$VENV_DIR" && pwd)" ]; then
    exit_on_error "Virtual environment activation check failed. Expected VIRTUAL_ENV='${VENV_DIR}', got '${VIRTUAL_ENV:-Not set}'."
fi
log_info "Virtual environment successfully activated: $VIRTUAL_ENV"
log_info "Python interpreter in use: $(command -v python)"


# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
python -m pip install --upgrade pip || exit_on_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE}..."
    
    # Specific check for GeoPandas (which was removed)
    if grep -qiE "geopandas|fiona" "$REQUIREMENTS_FILE"; then # Also check for fiona as it's a heavy dep of geopandas
        log_warn "----------------------------------------------------------------------"
        log_warn "WARNING: 'geopandas' or 'fiona' found in '${REQUIREMENTS_FILE}'."
        log_warn "The Sentinel project has been refactored to remove these dependencies."
        log_warn "Consider removing them from '${REQUIREMENTS_FILE}' if they are no longer needed."
        log_warn "Proceeding with installation, but this might indicate an outdated requirements file."
        log_warn "----------------------------------------------------------------------"
    fi
    
    # General check for other potentially heavy geospatial libraries
    if grep -qE "pyproj|shapely|rtree|rasterio" "$REQUIREMENTS_FILE"; then
        log_info "----------------------------------------------------------------------"
        log_info "Geospatial libraries (e.g., pyproj, shapely, rtree, rasterio) detected in requirements."
        log_info "Ensure corresponding system dependencies (e.g., GEOS, PROJ) are installed if these are used."
        log_info "----------------------------------------------------------------------"
    fi

    python -m pip install -r "${REQUIREMENTS_FILE}" || exit_on_error "Failed to install dependencies from ${REQUIREMENTS_FILE}."
    log_info "Python dependencies installed successfully."
else
    exit_on_error "Requirements file '${REQUIREMENTS_FILE}' not found. Cannot install dependencies."
fi

# --- Post-Setup Information ---
echo
log_info "=========================================================================="
log_info "Python Environment Setup for Sentinel Health Co-Pilot Complete!"
log_info "=========================================================================="
log_info "Virtual Environment Path: ${VENV_DIR}"
echo
log_info "To activate this environment in a new terminal: source ${VENV_DIR}/bin/activate"
log_info "To run the Streamlit app (from project root '${PROJECT_ROOT}'):"
log_info "  cd ${PROJECT_ROOT} && streamlit run app.py"
log_info "To deactivate the environment (when done): deactivate"
echo

# Check for .env.example
ENV_EXAMPLE_FILE="${PROJECT_ROOT}/.env.example"
ACTUAL_ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ACTUAL_ENV_FILE" ]; then
    log_warn "----------------------------------------------------------------------"
    log_warn "IMPORTANT: Found '${ENV_EXAMPLE_FILE}'. You should copy it to '${ACTUAL_ENV_FILE}' and customize it:"
    log_warn "  cp \"${ENV_EXAMPLE_FILE}\" \"${ACTUAL_ENV_FILE}\""
    log_warn "Then, edit '${ACTUAL_ENV_FILE}' to set necessary environment variables (e.g., MAPBOX_ACCESS_TOKEN)."
    log_warn "----------------------------------------------------------------------"
elif [ ! -f "$ENV_EXAMPLE_FILE" ] && [ ! -f "$ACTUAL_ENV_FILE" ]; then
    log_info "No '.env.example' or '.env' file found. Some features like Mapbox maps might require API keys set as environment variables."
fi

exit 0
