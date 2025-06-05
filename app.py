# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import sys # sys must be imported absolutely first for path manipulation
from pathlib import Path # Pathlib for robust path operations
import logging # Import logging early for potential setup issues
import html 
import importlib.util # For checking dependencies like plotly

# --- CRITICAL PATH SETUP ---
# This script (app.py) is assumed to be in the project root: ssentinel_project_root/app.py
# Therefore, Path(__file__).resolve().parent IS the project root.
_this_file_path_app = Path(__file__).resolve() # Specific to app.py's location
_project_root_dir_app = _this_file_path_app.parent    

# Initial diagnostic prints to stderr (visible in server logs)
print(f"DEBUG_APP_PY (L19): __file__ in app.py = {__file__}", file=sys.stderr)
print(f"DEBUG_APP_PY (L20): _this_file_path_app = {_this_file_path_app}", file=sys.stderr)
print(f"DEBUG_APP_PY (L21): Calculated _project_root_dir_app = {_project_root_dir_app}", file=sys.stderr)
print(f"DEBUG_APP_PY (L22): Initial sys.path before any modification in app.py = {sys.path}", file=sys.stderr)

# Ensure the project root is the first entry in sys.path
project_root_str_app = str(_project_root_dir_app)
if project_root_str_app not in sys.path:
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L29): Added project root '{project_root_str_app}' to sys.path.", file=sys.stderr)
elif sys.path[0] != project_root_str_app:
    try:
        sys.path.remove(project_root_str_app)
        print(f"DEBUG_APP_PY (L34): Removed '{project_root_str_app}' from interior of sys.path.", file=sys.stderr)
    except ValueError:
        print(f"DEBUG_APP_PY (L36): '{project_root_str_app}' reported in sys.path but not found by remove(). Odd.", file=sys.stderr)
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L39): Moved '{project_root_str_app}' to start of sys.path.", file=sys.stderr)
else:
    print(f"DEBUG_APP_PY (L42): Project root '{project_root_str_app}' is already at start of sys.path.", file=sys.stderr)

print(f"DEBUG_APP_PY (L45): sys.path JUST BEFORE 'from config import settings' =File 56 (Corrected `app.py` - Path logic and stray text removed):
```python
# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import sys # sys must be imported absolutely first for path manipulation
from pathlib import Path # Pathlib for robust path operations
# Import other standard libraries AFTER path setup if they might be affected by PYTHONPATH issues
# (though less likely for standard libraries like logging, html, importlib.util)

# --- CRITICAL PATH SETUP ---
# This section MUST execute correctly before almost any other import from the project.
# Goal: Ensure `_project_root_dir` is correctly `ssentinel_project_root`
#       and it's the first entry in `sys.path`.

_this_file_path_app = Path(__file__).resolve() # Absolute path to this app.py
_project_root_dir_app = _this_file_path_app.parent    # If app.py is in project root, its parent dir IS the project root.

# Initial diagnostic prints to stderr (visible in server logs)
# These use print because logging might not be configured yet.
print(f"DEBUG_APP_PY (L19): __file__ in app.py = {__file__}", file=sys.stderr)
print(f"DEBUG_APP_PY (L20): _this_file_path_app = {_this_file_path_app}", file=sys.stderr)
print(f"DEBUG_APP_PY (L21): Calculated _project_root_dir_app = {_project_root_dir_app}", file=sys.stderr)
print(f"DEBUG_APP_PY (L22): Initial sys.path before any modification in app.py = {sys.path}", file=sys.stderr)

# Add/Prioritize project root in sys.path
project_root_str_app = str(_project_root_dir_app)
if project_root_str_app not in sys.path:
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L29): Added project root '{project_root_str_app}' to sys.path.", file=sys.stderr)
elif sys.path[0] != project_root_str_app:
    try: # Ensure it's removed before re-inserting at index 0
        sys.path.remove(project_root_str_app)
        print(f"DEBUG_APP_PY (L34): Removed '{project_root_str_app}' from interior of sys.path for re-ordering.", file=sys.stderr)
    except ValueError:
        print(f"DEBUG_APP_PY (L36): '{project_root_str_app}' reported in sys.path but not found by remove(). Will be inserted at [0].", file=sys.stderr)
    sys.path.insert(0, project_root_str_app)
    print(f"DEBUG_APP_PY (L39): Moved '{project_root_str_app}' to start of sys.path.", file=sys.stderr)
else:
    print(f"DEBUG_APP_PY (L42): Project root '{project_root_str_app}' is already at start of sys.path.", file=sys.stderr)

print(f"DEBUG_APP_PY (L45): sys.path JUST BEFORE 'from config import settings' = {sys.path}", file=sys.stderr)

# --- Import Settings (This is the critical point) ---
try:
    from config import settings 
    # If this works, then 'config' package was found correctly under _project_root_dir_app
except ImportError as e_cfg_app_final_v2:
    print(f"FATAL_APP_PY (L52): FAILED to import 'config.settings': {e_cfg_app_final_v2}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    print(f"Verify: Does '{_project_root_dir_app / 'config'}' exist? {(_project_root_dir_app / 'config').is_dir()}", file=sys.stderr)
    print(f"Verify: Does '{_project_root_dir_app / 'config' / '__init__.py'}' exist? {sys.path}", file=sys.stderr)

# --- Import Settings (This is the critical point) ---
try:
    from config import settings 
except ImportError as e_cfg_app_final_corrected:
    print(f"FATAL_APP_PY (L52): STILL FAILED to import config.settings: {e_cfg_app_final_corrected}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    # Detailed check for config package structure
    config_package_path = _project_root_dir_app / 'config'
    config_init_path = config_package_path / '__init__.py'
    config_settings_path = config_package_path / 'settings.py'
    print(f"Check: Does '{config_package_path}' exist and is it a directory? {config_package_path.is_dir()}", file=sys.stderr)
    print(f"Check: Does '{config_init_path}' exist? {config_init_path.is_file()}", file=sys.stderr)
    print(f"Check: Does '{config_settings_path}' exist? {config_settings_path.is_file()}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings_final_corrected: 
    print(f"FATAL_APP_PY (L63): AttributeError on 'config.settings' (likely circular import OR settings.py has an error OR sys.path issue still): {e_attr_settings_final_corrected}", file=sys.stderr)
    print(f"FINAL sys.path at attribute error: {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e_genericFile 56 (Corrected `app.py`):
```python
# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import sys # sys must be imported absolutely first for path manipulation
from pathlib import Path # Pathlib for robust path operations

# --- CRITICAL PATH SETUP ---
# This section MUST execute correctly before almost any other import from the project.
# This script (app.py) is assumed to be in the project root: ssentinel_project_root/app.py
# Therefore, Path(__file__).resolve().parent IS the project root.
_this_app_file_path = Path(__file__).resolve() # Absolute path to this app.py
_project_root_dir = _this_app_file_path.parent    # If app.py is in project root, its parent dir IS the project root.

# Initial diagnostic prints to stderr (visible in server logs)
# These use direct print to stderr as logging might not be configured yet.
print(f"DEBUG_APP_PY (L19): __file__ = {__file__}", file=sys.stderr)
print(f"DEBUG_APP_PY (L20): _this_app_file_path = {_this_app_file_path}", file=sys.stderr)
print(f"DEBUG_APP_PY (L21): Calculated _project_root_dir = {_project_root_dir}", file=sys.stderr)
print(f"DEBUG_APP_PY (L22): Initial sys.path before any modification = {sys.path}", file=sys.stderr)

# Ensure the project root is the first entry in sys.path
project_root_str = str(_project_root_dir)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
    print(f"DEBUG_APP_PY (L29): Added project root '{project_root_str}' to sys.path.", file=sys.stderr)
elif sys.path[0] != project_root_str: # If already in path, ensure it's first
    try:
        sys.path.remove(project_root_str)
        print(f"DEBUG_APP_PY (L34): Removed '{project_root_str}' from interior of sys.path.", file=sys.stderr)
    except ValueError:
        # This case (in sys.path but not found by remove) is unlikely but handled.
        print(f"DEBUG_APP_PY (L37): '{project_root_str}' reported in sys.path but not found by remove(). Will insert at [0].", file=sys.stderr)
    sys.path.insert(0, project_root_str)
    print(f"DEBUG_APP_PY (L40): Moved '{project_root_str}' to start of sys.path.", file=sys.stderr)
else:
    print(f"DEBUG_APP_PY (L43): Project root '{project_root_str}' is already at start of sys.path.", file=sys.stderr)

print(f"DEBUG_APP_PY (L46): sys.path JUST BEFORE 'from config import settings' = {sys.path}", file=sys.stderr)

# --- Import Settings (This is the critical point) ---
try:
    from config import settings 
except ImportError as e_cfg_app_final_attempt: # Renamed exception var
    print(f"FATAL_APP_PY (L52): STILL FAILED to import 'config.settings': {e_cfg_app_final_attempt}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    # Detailed checks for directory structure
    config_package_path = _project_root_dir / 'config'
    settings_module_path = config_package_path / 'settings.py'
    init_in_config_path = config_package {(_project_root_dir_app / 'config' / '__init__.py').is_file()}", file=sys.stderr)
    print(f"Verify: Does '{_project_root_dir_app / 'config' / 'settings.py'}' exist? {(_project_root_dir_app / 'config' / 'settings.py').is_file()}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings_final_v2: 
    print(f"FATAL_APP_PY (L60): AttributeError on 'config.settings' (likely circular import OR settings.py has an internal error): {e_attr_settings_final_v2}", file=sys.stderr)
    print(f"FINAL sys.path at attribute error: {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg_final_v2: # Catch any other exception
    print(f"FATAL_APP_PY (L65): Generic error during 'config.settings' import: {e_generic_cfg_final_v2}", file=sys.stderr)
    print(f"FINAL sys.path at generic error: {sys.path}", file=sys.stderr)
    sys.exit(1)

# --- Now that settings is imported, configure logging and import other modules ---
import streamlit as st # Standard Streamlit import
import logging # Standard logging import
import html 
import importlib.util

# --- Global Logging Configuration ---
valid_log_levels_app_final_cfg_v2 = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str_final_cfg_v2 = str(settings.LOG_LEVEL).upper()
if log_level_app_str_final_cfg_v2 not in valid_log_levels_app_final_cfg_v2:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str_final_cfg_v2}'. Using INFO.", file=sys.stderr); log_level_app_str_final_cfg_v2 = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str_final_cfg_v2, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) 

logger.info(f"INFO (app.py): Successfully imported and accessed config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    from packaging import version 
    st_version = version.parse(st.__version__)
    if st_version >= version.parse("1.30.0"): STREAMLIT_VERSION_GE_1_30 = True
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True # page_link available earlier, but use_container_width later
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {st.__version__} < 1.30.0. Some UI features like st.container(border=True) or page_link(use_container_width=True) might use fallbacks or not work.")
except Exception as e_st_ver_final_cfg_v2: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver_final_cfg_v2}")

if not importlib.util.find_spec("plotly"): logger.warning("Plotly library not found. Visualization features dependent on Plotly may fail.")

# --- Page Configuration ---
page_icon_path_obj_app_main_cfg_final_v2 = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_main_cfg_final_v2: str = str(page_icon_path_obj_app_main_cfg_final_v2) if page_icon_path_obj_app_main_cfg_final_v2.is_file() else "🌍"
if final_page_icon_str_app_main_cfg_final_v2 == "🌍": logger.warning(f"Page icon not found: '{page_icon_path_obj_app_main_cfg_final_v2}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_main_cfg_final_v2,
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Plotly Theme & CSS ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme(); logger.debug("Sentinel Plotly theme applied.")
except Exception as e_theme_main_app_cfg_final_v2: logger.error(f"Error applying Plotly theme: {e_theme_main_app_cfg_final_v2}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app_final_cfg_val_v2(css_path_str_app_final_cfg_val_v2: str):
    css_path_app_final_cfg_val_v2 = Path(css_path_str_app_final_cfg_val_v2)
    if css_path_app_final_cfg_val_v2.is_file():
        try:
            with open(css_path_app_final_cfg_val_v2, "r", encoding="utf-8") as f_css_app_final_cfg_val_v2: st.markdown(f'<style>{f_css_app_final_cfg_val_v2.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_final_cfg_val_v2}")
        except Exception as e_css_main_app_final_cfg_val_v2: logger.error(f"Error applying CSS {css_path_app_final_cfg_val_v2}: {e_css_main_app_final_cfg_val_v2}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_final_cfg_val_v2}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_final_cfg_val_v2(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_final_cfg_val_ui_v3 = st.columns([0.12, 0.88])
with header_cols_app_ui_final_cfg_val_ui_v3[0]:
    l_logo_path_app_final_cfg_val_v2 = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_final_cfg_val_v2 = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_final_cfg_val_v2.is_file(): st.image(str(l_logo_path_app_final_cfg_val_v2), width=100)
    elif s_logo_path_app_final_cfg_val_v2.is_file(): st.image(str(s_logo_path_app_final_cfg_val_v2), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_final_cfg_val_v2}', S: '{s_logo_path_app_final_cfg_val_v2}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_final_cfg_val_ui_v3[1]: st.title(html.escape(settings.APP_NAME)); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
core_principles_main_app_v5__cfg_final_corrected: # Catch any other exception during settings import
    print(f"FATAL_APP_PY (L68): Generic error during 'config.settings' import: {e_generic_cfg_final_corrected}", file=sys.stderr)
    print(f"FINAL sys.path at generic error: {sys.path}", file=sys.stderr)
    sys.exit(1)

# --- Imports that depend on settings ---
import streamlit as st # Moved Streamlit import after settings if it might use it indirectly
# logging already imported, but get a logger for app.py after basicConfig
# html, importlib.util already imported

# --- Global Logging Configuration ---
valid_log_levels_app_final_cfg_v2 = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str_final_cfg_v2 = str(settings.LOG_LEVEL).upper()
if log_level_app_str_final_cfg_v2 not in valid_log_levels_app_final_cfg_v2:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str_final_cfg_v2}'. Using INFO.", file=sys.stderr); log_level_app_str_final_cfg_v2 = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str_final_cfg_v2, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) # Logger for this app.py

logger.info(f"INFO (app.py): Successfully imported and accessed config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30_APP = False 
STREAMLIT_PAGE_LINK_AVAILABLE_APP = False
try:
    from packaging import version 
    if version.parse(st.__version__) >= version.parse("1.30.0"): STREAMLIT_VERSION_GE_1_30_APP = True
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE_APP = True
    if not STREAMLIT_VERSION_GE_1_30_APP: logger.warning(f"Streamlit version {st.__version__} < 1.30.0. Some UI features might use fallbacks.")
except Exception as e_st_ver_final_cfg_val_app: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver_final_cfg_val_app}")

if not importlib.util.find_spec("plotly"): logger.warning("Plotly not installed. Visualization features may fail.")

# --- Page Configuration ---
page_icon_path_obj_app_main_cfg_final_val = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_main_cfg_final_val: str = str(page_icon_path_obj_app_main_cfg_final_val) if page_icon_path_obj_app_main_cfg_final_val.is_file() else "🌍"
if final_page_icon_str_app_main_cfg_final_val == "🌍": logger.warning(f"Page icon not found: '{page_icon_path_obj_app_main_cfg_final_val}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_main_cfg_final_val,
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Plotly Theme & CSS ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme(); logger.debug("Sentinel Plotly theme applied.")
except Exception as e_theme_main_app_cfg_final_val_app: logger.error(f"Error applying Plotly theme: {e_theme_main_app_cfg_final_val_app}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app_final_cfg_val_ui_app(css_path_str_app_final_cfg_val_ui_app: str):
    css_path_app_final_cfg_val_ui_app = Path(css_path_str_app_final_cfg_val_ui_app)
    if css_path_app_final_cfg_val_ui_app.is_file():
        try:
            with open(css_path_app_final_cfg_val_ui_app, "r", encoding="utf-8") as f_css_app_final_cfg_val_ui_app: st.markdown(f'<style>{f_css_app_final_cfg_val_ui_app.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_final_cfg_val_ui_app}")
        except Exception as e_css_main__path / '__init__.py'
    print(f"Project Root check: '{_project_root_dir}'", file=sys.stderr)
    print(f"  config dir exists? ({config_package_path}): {config_package_path.is_dir()}", file=sys.stderr)
    print(f"  config/__init__.py exists? ({init_in_config_path}): {init_in_config_path.is_file()}", file=sys.stderr)
    print(f"  config/settings.py exists? ({settings_module_path}): {settings_module_path.is_file()}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings_final_attempt: # Renamed exception var
    print(f"FATAL_APP_PY (L64): AttributeError on 'config.settings' (likely circular import OR settings.py has an issue OR sys.path still incorrect): {e_attr_settings_final_attempt}", file=sys.stderr)
    print(f"FINAL sys.path at attribute error: {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg_final_attempt: # Renamed exception var
    print(f"FATAL_APP_PY (L69): Generic error during 'config.settings' import: {e_generic_cfg_final_attempt}", file=sys.stderr)
    print(f"FINAL sys.path at generic error: {sys.path}", file=sys.stderr)
    sys.exit(1)

# --- Imports that depend on settings ---
import streamlit as st # Import Streamlit after settings if it's not needed for path setup
import logging # Import logging after settings for its config values
import html 
import importlib.util

# --- Global Logging Configuration ---
# This must come AFTER settings is successfully imported.
valid_log_levels_app_final_cfg_val_v2 = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} # Renamed var
log_level_app_str_final_cfg_val_v2 = str(settings.LOG_LEVEL).upper()
if log_level_app_str_final_cfg_val_v2 not in valid_log_levels_app_final_cfg_val_v2:
    # Use print as logger might not be fully up if settings.LOG_LEVEL was bad
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str_final_cfg_val_v2}'. Using INFO.", file=sys.stderr)
    log_level_app_str_final_cfg_val_v2 = "INFO"

logging.basicConfig(level=getattr(logging, log_level_app_str_final_cfg_val_v2, logging.INFO), 
                    format=settings.LOG_FORMAT,
app_final_cfg_val_ui_app: logger.error(f"Error applying CSS {css_path_app_final_cfg_val_ui_app}: {e_css_main_app_final_cfg_val_ui_app}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_final_cfg_val_ui_app}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_final_cfg_val_ui_app(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_final_cfg_val_ui_val_app = st.columns([0.12, 0.88])
with header_cols_app_ui_final_cfg_val_ui_val_app[0]:
    l_logo_path_app_final_cfg_val_app = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_final_cfg_val_app = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_final_cfg_val_app.is_file(): st.image(str(l_logo_path_app_final_cfg_val_app), width=100)
    elif s_logo_path_app_final_cfg_val_app.is_file(): st.image(str(s_logo_path_app_final_cfg_val_app), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_final_cfg_val_app}', S: '{s_logo_path_app_final_cfg_val_app}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_final_cfg_val_ui_val_app[1]: st.title(html.escape(settings.APP_NAME)); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description (Full content from prompt used here) ---
st.markdown(f"""## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
core_principles_main_app_v5_val_app = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v5_val_app = min(len(core_principles_main_app_v5_val_app), 2)
if num_cols_core_principles_v5_val_app > 0:
    cols_core_principles_ui_v5_val_app = st.columns(num_cols_core_principles_v5_val_app)
    for idx_core_v5_val_app, (title_core_v5_val_app, desc_core_v5_val_app) in enumerate(core_principles_main_app_v5_val_app):
        with cols_core_principles_ui_v5_val_app[idx_core_v5_val_app % num_cols_core_principles_v5_val_app]:
            st.markdown(f"##### {html.escape(title_core_v5_val_app)}"); st.markdown(f"<small>{html.escape(desc_core_v5_val_app)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their Personal Edge Device (PED), tailored for their specific operational context.")
st.info("💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's data processing capabilities and the types of aggregated views available to management and strategic personnel.")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app_final_cfg_val_app = _project_root_dir_app / "pages" 
role_navigation_config_app_final_cfg_list_val_app = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] 

num_nav_cols_final_app_cfg_val_ui_v4_app = min(len(role_navigation_config_app_final_cfg_list_val), 2)
if num_nav_cols_final_app_cfg_val_ui_v4_app > 0:
    nav_cols_ui_final_app_cfg_val_ui_v4_app = st.columns(num_nav_cols_final_app_cfg_val_ui_v4_app)
    current_col_idx_nav_final_cfg_val_ui_v4_app = 0
    for nav_item_final_app_cfg_item_val_v4_app in role_navigation_config_app_final_cfg_list_val:
        page_link_target_app_cfg_item_val_v4_app = nav_item_final_app_cfg_item_val_v4_app['page_filename'] 
        physical_page_full_path_app_cfg_item_val_v4_app = pages_directory_obj_app_final_cfg_val_app / nav_item_final_app_cfg_item_val_v4_app["page_filename"]
        if not physical_page_full_path_app_cfg_item_val_v4_app.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg_item_val_v4_app['title']}' not found: {physical_page_full_path_app_cfg_item_val_v4_app}")
            continue
        with nav_cols_ui_final_app_cfg_val_ui_v4_app[current_col_idx_nav_final_cfg_val_ui_v4_app % num_nav_cols_final_app_cfg_val_ui_v4_app]:
            container_args_final_app_cfg_val_v4_app = {"border": True} if STREAMLIT_VERSION_GE_1_30_APP else {}
            with st.container(**container_args_final_app_cfg_val_v4_app):
                st.subheader(f"{nav_item_final_app_cfg_item_val_v4_app['icon']} {html.escape(nav_item_final_app_cfg_item_val_v4_app['title'])}")
                st.markdown(f"<small>{nav_item_final_app_cfg_item_val_v4_app['desc']}</small>", unsafe_allow_html=True)
                link_label_final_app_cfg_val_v4_app = f"Explore {nav_item_final_app_cfg_item_val_v4_app['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE_APP:
                    link_kwargs_final_app_cfg_val_v4_app = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30_APP else {}
                    st.page_link(page_link_target_app_cfg_item_val_v4_app, label=link_label_final_app_cfg_val_v4_app, icon="➡️", **link_kwargs_final_app_cfg_val_v4_app)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_cfg_item_val_v4_app["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_cfg_val_v4_app} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_cfg_val_ui_v4_app += 1
st.divider()

st.header(f"{html.escape(settings.APP_NAME)} - Key Capabilities Reimagined")
capabilities_data_app_final_cfg_full_v4_app = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_cfg_val_final_v4_app = min(len(capabilities_data_app_final_cfg_full_v4_app), 3)
if num_cap_cols_final_app_cfg_val_final_v4_app > 0:
    cap_cols_ui_final_app_cfg_val_final_v4_app = st.columns(num_cap_cols_final_app_cfg_val_final_v4_app)
    for i_cap_final_cfg_final_v4_app, (cap_t_final_cfg_final_v4_app, cap_d_final_cfg_final_v4_app) in enumerate(capabilities_data_app_final_cfg_full_v4_app):
        with cap_cols_ui_final_app_cfg_val_final_v4_app[i_cap_final_cfg_final_v4_app % num_cap_cols_final_app_cfg_val_final_v4_app]: 
            st.markdown(f"##### {html.escape(cap_t_final_cfg_final_v4_app)}"); st.markdown(f"<small>{html.escape(cap_d_final_cfg_final_v4_app)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header(f"{html.escape(settings.APP_NAME)} v{settings.APP_VERSION}")
st.sidebar.divider(); st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()
glossary_filename_sidebar_cfg_final_val_v4_app = "05_glossary_page.py" 
glossary_link_target_sidebar_cfg_final_val_v4_app = glossary_filename_sidebar_cfg_final_val_v4_app 
glossary_physical_path_final_sb_cfg_final_val_v4_app = pages_directory_obj_app_cfg_val_app / glossary_filename_sidebar_cfg_final_val_v4_app
if glossary_physical_path_final_sb_cfg_final_val_v4_app.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE_APP: st.sidebar.page_link(glossary_link_target_sidebar_cfg_final_val_v4_app, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_cfg_final_val_v4_app}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page for sidebar (expected: {glossary_physical_path_final_sb_cfg_final_val_v4_app}) not found.")
st.sidebar.divider()
st.sidebar.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**"); st.sidebar.markdown(f"Support: [{html.escape(settings.SUPPORT_CONTACT_INFO)}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(html.escape(settings.APP_FOOTER_TEXT))
logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
