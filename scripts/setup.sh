# sentinel_project_root/scripts/setup.sh
#!/bin/bash
# setup.sh - For Sentinel Health Co-Pilot Python Backend/Development Environments
# This script sets up a Python virtual environment and installs dependencies
# primarily for Web Dashboards, backend services (if any), and Dev/Simulation.

echo "======================================================================"
echo "Setting up Sentinel Health Co-Pilot Python Virtual Environment..."
echo "Target: Python Web Dashboards, Dev/Simulation Environment."
echo "======================================================================"
echo

# --- Configuration ---
# Determine Project Root: Assume script is in sentinel_project_root/scripts/
# So, project root is one level up from the script's directory.
SCRIPT_DIR_SETUP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT_DIR_SETUP="$(dirname "$SCRIPT_DIR_SETUP")" # Should resolve to sentinel_project_root

VENV_NAME_CONFIG_PY="${VENV_NAME_PY:-.venv_sentinel_py}" # Renamed to avoid conflict if multiple venvs for project
VENV_DIR_PATH="${PROJECT_ROOT_DIR_SETUP}/${VENV_NAME_CONFIG_PY}"
# requirements.txt should be at the project root.
REQUIREMENTS_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/requirements.txt"
# packages.txt for system dependencies (used by Streamlit Cloud, good for reference)
SYSTEM_PACKAGES_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/packages.txt"
PYTHON_EXECUTABLE="${PYTHON_CMD:-python3}" # Allow override, e.g., python3.9

# --- Helper Functions ---
log_info() { echo "[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_warn() { echo "[WARN] $(date +'%Y-%m-%d %H:%M:%S') - $1"; }
log_error() { echo "[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1" >&2; }

exit_on_critical_error() {
    # $1 is the error message string
    # $2 is the exit code (optional, defaults to 1)
    local exit_code=${2:-1}
    if [ $? -ne 0 ] || [ "$1" != "SKIP_EXIT_CHECK" ] ; then # Allow manual trigger of exit too
        log_error "Failed step: $1"
        log_error "Setup aborted due to a critical error. Exit code: $exit_code"
        # Attempt to deactivate venv if active (best effort)
        if [ -n "$VIRTUAL_ENV" ]; then
            log_info "Attempting to deactivate virtual environment (if active)..."
            deactivate &>/dev/null
        fi
        exit "$exit_code"
    fi
}

# --- Pre-requisite Checks ---
log_info "Using Project Root: ${PROJECT_ROOT_DIR_SETUP}"
log_info "Checking for Python interpreter: ${PYTHON_EXECUTABLE}..."
if ! command -v ${PYTHON_EXECUTABLE} &> /dev/null; then
    log_error "${PYTHON_EXECUTABLE} command not found. Please install Python 3 (3.8+ recommended)."
    exit_on_critical_error "Python interpreter check failed." "SKIP_EXIT_CHECK" # Manual exit
fi
PYTHON_VERSION_INFO=$(${PYTHON_EXECUTABLE} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
log_info "Found Python version: ${PYTHON_VERSION_INFO}"

# Check Python version (e.g., >= 3.8)
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=8
CURRENT_PYTHON_MAJOR=$(echo "$PYTHON_VERSION_INFO" | cut -d. -f1)
CURRENT_PYTHON_MINOR=$(echo "$PYTHON_VERSION_INFO" | cut -d. -f2)

if [ "$CURRENT_PYTHON_MAJOR" -lt "$MIN_PYTHON_MAJOR" ] || \
   ( [ "$CURRENT_PYTHON_MAJOR" -eq "$MIN_PYTHON_MAJOR" ] && [ "$CURRENT_PYTHON_MINOR" -lt "$MIN_PYTHON_MINOR" ] ); then
    log_warn "Current Python version ${PYTHON_VERSION_INFO} is older than recommended ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+. Some dependencies might encounter issues."
    # Decide if this is a critical error or just a warning
    # exit_on_critical_error "Python version too old." "SKIP_EXIT_CHECK" # Example if critical
fi


log_info "Checking for Python 'venv' module..."
if ! ${PYTHON_EXECUTABLE} -m venv -h &> /dev/null; then
    log_error "'venv' module not found for ${PYTHON_EXECUTABLE}. This is usually part of a standard Python installation."
    log_error "Please verify your Python setup (e.g., ensure 'python3-venv' or 'python${CURRENT_PYTHON_MAJOR}.${CURRENT_PYTHON_MINOR}-venv' package is installed)."
    exit_on_critical_error "'venv' module check failed." "SKIP_EXIT_CHECK"
fi
log_info "'venv' module found."

# --- System Dependency Guidance (from packages.txt) ---
if [ -f "$SYSTEM_PACKAGES_FILE_PATH" ]; then
    log_info "----------------------------------------------------------------------"
    log_info "NOTE: This project may require system-level libraries."
    log_info "Consult 'packages.txt' for a list of APT packages (Debian/Ubuntu)."
    log_info "Example system dependencies often include:"
    log_info "  build-essential, libgdal-dev, libproj-dev, libgeos-dev, pkg-config, etc."
    log_info "Ensure these are installed on your system, especially if Python package installations fail."
    log_info "For example, on Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y \$(cat ${SYSTEM_PACKAGES_FILE_PATH} | grep -v '^#')"
    log_info "----------------------------------------------------------------------"
else
    log_warn "'packages.txt' not found. System dependencies might be required but are not listed."
fi


# --- Virtual Environment Setup ---
if [ ! -d "${VENV_DIR_PATH}" ]; then
    log_info "Creating Python virtual environment in ${VENV_DIR_PATH}..."
    ${PYTHON_EXECUTABLE} -m venv "${VENV_DIR_PATH}"
    exit_on_critical_error "Failed to create virtual environment at '${VENV_DIR_PATH}'. Check permissions and Python 'venv' module status."
    log_info "Virtual environment created successfully."
else
    log_info "Python virtual environment ${VENV_DIR_PATH} already exists. Skipping creation."
fi

# --- Activate Virtual Environment ---
log_info "Activating Python virtual environment: source \"${VENV_DIR_PATH}/bin/activate\""
# shellcheck source=/dev/null # Suppress SC1090/SC1091 for dynamic path
if ! source "${VENV_DIR_PATH}/bin/activate"; then
    log_error "Failed to activate the virtual environment by sourcing from script."
    log_error "Please try activating manually in your shell: source \"${VENV_DIR_PATH}/bin/activate\""
    exit_on_critical_error "Virtual environment activation script source failed." "SKIP_EXIT_CHECK"
fi

# Robust check for activation using VIRTUAL_ENV variable
# Use readlink -f to get canonical path for comparison, handles symlinks.
# Ensure VENV_DIR_PATH is also canonical if it might be a symlink.
CANONICAL_VENV_DIR_PATH=$(readlink -f "${VENV_DIR_PATH}")
if [ -z "$VIRTUAL_ENV" ] || [ "$(readlink -f "$VIRTUAL_ENV")" != "$CANONICAL_VENV_DIR_PATH" ]; then
    log_error "Virtual environment activation check failed or pointed to an unexpected location."
    log_error "Expected VIRTUAL_ENV='${CANONICAL_VENV_DIR_PATH}', but found VIRTUAL_ENV='${VIRTUAL_ENV:-Not set}'."
    log_error "Please ensure activation was successful or try activating manually: source \"${VENV_DIR_PATH}/bin/activate\""
    exit_on_critical_error "VIRTUAL_ENV variable check failed after activation attempt." "SKIP_EXIT_CHECK"
fi
log_info "Virtual environment successfully activated: $VIRTUAL_ENV"

# --- Pip Upgrade and Dependency Installation ---
log_info "Upgrading pip within the virtual environment..."
python -m pip install --upgrade pip
exit_on_critical_error "Failed to upgrade pip."

if [ -f "$REQUIREMENTS_FILE_PATH" ]; then
    log_info "Installing Python dependencies from ${REQUIREMENTS_FILE_PATH}..."
    # Check for GeoPandas explicitly since it was a major point. If found, error.
    if grep -qi "geopandas" "$REQUIREMENTS_FILE_PATH"; then
        log_error "----------------------------------------------------------------------"
        log_error "CRITICAL ERROR: 'geopandas' found in ${REQUIREMENTS_FILE_PATH}."
        log_error "This project has been refactored to REMOVE GeoPandas dependency."
        log_error "Please remove 'geopandas' from your requirements.txt file."
        log_error "----------------------------------------------------------------------"
        exit_on_critical_error "'geopandas' must be removed from requirements." "SKIP_EXIT_CHECK"
    fi
    
    # Check for other common geospatial libraries that might imply system deps.
    # Since GeoPandas is removed, the direct need for some of these might be gone,
    # but they could be used independently.
    if grep -qE "fiona|pyproj|shapely|rtree|rasterio" "$REQUIREMENTS_FILE_PATH"; then
        log_info "----------------------------------------------------------------------"
        log_info "NOTE: Geospatial-related libraries (like fiona, pyproj, shapely, rasterio) detected in requirements."
        log_info "These might still require system-level C libraries (e.g., GEOS, PROJ)."
        log_info "If installation fails for these, ensure their system dependencies are met."
        log_info "Consult individual library documentation or 'packages.txt' for system package names."
        log_info "----------------------------------------------------------------------"
    fi
    
    python -m pip install -r "${REQUIREMENTS_FILE_PATH}"
    if [ $? -eq 0 ]; then
        log_info "Python dependencies installed successfully from ${REQUIREMENTS_FILE_PATH}."
    else
        log_warn "Some Python dependencies may have failed to install from ${REQUIREMENTS_FILE_PATH}."
        log_warn "Please review the error messages above carefully."
        log_warn "Common causes include missing system libraries or package version conflicts."
        log_warn "Consult project documentation or specific package installation guides for troubleshooting."
        # For a CI/CD or production setup, this might be a critical error.
        # exit_on_critical_error "Python dependency installation failed." # Uncomment if this should be fatal
    fi
else
    log_error "Requirements file '${REQUIREMENTS_FILE_PATH}' not found. Cannot install Python dependencies."
    log_error "A 'requirements.txt' file at the project root is essential."
    exit_on_critical_error "requirements.txt not found." "SKIP_EXIT_CHECK"
fi

# --- Post-Setup Information & Guidance ---
echo
log_info "=========================================================================="
log_info "Python Environment Setup for Sentinel Health Co-Pilot is Complete!"
log_info "=========================================================================="
log_info "Virtual Environment Path: ${VENV_DIR_PATH}"
echo
log_info "To use this environment in your current shell session (if not already active):"
log_info "  source \"${VENV_DIR_PATH}/bin/activate\""
log_info "  (Your prompt should change to indicate the active venv, e.g., '(${VENV_NAME_CONFIG_PY}) user@host:...$')"
echo
log_info "To run the Streamlit application (example, from project root):"
log_info "  cd \"${PROJECT_ROOT_DIR_SETUP}\""
log_info "  streamlit run app.py"
echo
log_info "Native PED/Hub Applications (if applicable to project scope):"
log_info " - These require separate development environments (e.g., Android Studio for Android PEDs)."
log_info " - Edge AI models (.tflite) must be converted and bundled with native apps separately."
echo
log_info "To deactivate this Python virtual environment later, simply type: deactivate"
echo

# Check for .env.example file for user guidance on environment variables
ENV_EXAMPLE_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/.env.example" # Assuming .env.example exists
ACTUAL_ENV_FILE_PATH="${PROJECT_ROOT_DIR_SETUP}/.env"
if [ -f "$ENV_EXAMPLE_FILE_PATH" ] && [ ! -f "$ACTUAL_ENV_FILE_PATH" ]; then
    log_warn "----------------------------------------------------------------------"
    log_warn "IMPORTANT: An example environment file '.env.example' was found."
    log_warn "It's recommended to copy it to '.env' in the project root and customize it:"
    log_warn "  cp \"${ENV_EXAMPLE_FILE_PATH}\" \"${ACTUAL_ENV_FILE_PATH}\""
    log_warn "Then, edit '${ACTUAL_ENV_FILE_PATH}' with your actual environment variables"
    log_warn "(e.g., MAPBOX_ACCESS_TOKEN, database credentials, API keys if used)."
    log_warn "The application might rely on these environment variables for full functionality."
    log_warn "----------------------------------------------------------------------"
fi

exit 0
