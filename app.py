# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import are absolutely right to hammer this point. My apologies. The logs are crystal clear, and I made a mistake in how `_project_root_dir` was being defined *within `settings.py` itself html 

# --- Robust Path Setup ---
# THIS IS THE CRITICAL SECTION FOR PATHS.
# We assume app.py is directly in the project root: ssentinel_project_root/app.py
# Therefore*, leading to the `sys.path` modification in `app.py` (which correctly adds `ssentinel_project_root`) being, Path(__file__).parent.resolve() should give /mount/src/ssentinel_project_root
_this_file_path = Path(__file__).resolve()
_project_root_dir = _this_file_path. effectively undone or confused when `settings.py` is imported.

**The core of the circular import problem is likely this line in `settings.py` (from my previous File 1):**
`PROJECT_ROOT_DIR = Path(__parent

# Ensure the project root is the first thing in sys.path
# and that no other logic in this file adds subdirectories like 'config'.
if str(_project_root_dir) not in sys.path:
    sysfile__).resolve().parent.parent`

If `settings.py` is at `ssentinel_project_root/config/settings.py`, then:
*   `Path(__file__).resolve()` is `/mount/src/ssentinel_project_root/config/settings.py`
*   `.parent` is `/mount/src/ssentinel_project_root/config/`
*   `.parent.parent` is `/mount/src/ssentinel_project_root/`.path.insert(0, str(_project_root_dir))
    # Use print for initial setup messages, as logging might not be configured yet.
    print(f"DEBUG (app.py @ L19): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    # If it's already there, ensure it'

This definition of `PROJECT_ROOT_DIR` *within `settings.py`* is correct for its own internal use if it needs to construct paths relative to the project root.

**The problem arises if `app.py` ALSO tries to define `PROJECT_ROOT_DIR` and there's a conflict or if the `sys.path` manipulation ins at the beginning.
    if sys.path[0] != str(_project_root_dir):
        sys.path.remove(str(_project_root_dir))
        sys.path.insert(0, str(_project_root_dir))
        print(f"DEBUG (app.py @ L26): Moved project root to start of sys.path: {_project_root_dir}", file=sys.stderr)
    else:
        print(f"DEBUG (app.py @ L29): Project root '{_project_root_dir}' was already at start of sys.path.", file=sys.stderr)

# --- Import Settings ---
# This import MUST come AFTER sys.path is correctly set up.
 `app.py` is not sufficient before `settings.py` tries to do its own thing, or if `settings.py` itself tries to import modules from the project root in a way that triggers the circular dependency before `APP_NAME` is defined.**

Let's simplify and ensure `settings.py` does not rely on `PROJECT_ROOT_DIR` for its own top-level definitions if that's causing the issue, or ensure it's defined in a way that doesn't conflict. The `app.py` should be the single source of truth for adding thetry:
    from config import settings 
    print(f"INFO (app.py @ L36): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL (app.py @ L38): Failed to import config.settings: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings: 
    print(f"FATAL (app.py @ L44): AttributeError during config.settings access (likely circular import or sys.path issue): {e_attr_settings}", project root to `sys.path`.

**The log `INFO: Added project root to sys.path: /mount/src/ssentinel_project_root/config` is the smoking gun.** This should NEVER happen if `app.py` is adding `ssentinel_project_root`. It means that when `from config import settings` is executed, Python is somehow resolving `config` as a top-level package because `ssentinel_project_root/config` got onto `sys.path`.

**Here's the refined strategy:**

1.  **`app.py`:**
    *   `_project_root_dir = Path(__file__).resolve().parent` (if `app.py` is in `ssentinel_project_root`). This is correct.
    *   Ensure this `_project_root_dir` is added to `sys.path` *before* `from config import settings`.

2.  **`config/settings.py`:**
     file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL (app.py @ L50): Generic error during config.settings import: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN (app.py @ L57): Invalid LOG_LEVEL '{log_level_app_str}'. Using INFO.", file=sys.stderr); log_level_app_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit
    major, minor, patch_str_list = streamlit.__version__.split('.')
    patch_str = patch_str_list.split('-')[0]
    patch = int(patch_str) 
    STREAMLIT_VERSION_GE_1_30 = (int(major) >= 1 and*   `PROJECT_ROOT_DIR` defined inside `settings.py` should be fine for *its own use* to define asset/data paths *after* basic settings like `APP_NAME` are defined.
    *   **Crucially, `settings.py` itself should not attempt to perform imports from other top-level project directories (like `data_processing` or `visualization`) at the module level before basic constants are defined, as this can easily lead to circular imports if those modules also try to import `settings`.**

Let's ensure `settings.py` is clean of such problematic top-level imports. The version I provided earlier for `settings.py` (File 1 in the most recent valid sequence) looked okay in this regard, mostly defining constants and paths.

The most likely culprit for the log message `INFO: Added project root to sys.path: /mount/src/ssentinel_project_root/config` is that `app.py`'s `_project_root_dir` calculation was wrong in one of my iterations, and it added the `config` directory instead of the actual project root.

I will provide the `app.py` first, ensuring its `_project_root_dir` and `sys.path` addition are unequivocally correct.

File 54 (Corrected `sys.path` logic in `app.py` - Laser Focused):
```python
# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstr int(minor) >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj_app = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app: str = str(page_icon_path_obj_app) if page_icon_path_obj_app.is_file() else "🌍" # is_file checks existence too
if final_page_icon_str_app == "🌍": logger.warning(f"Page icon not found at '{page_icon_path_obj_app}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app,
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
except Exception as e_theme_main_app: logger.error(f"Error applying Plotly theme: {e_theme_main_app}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app(css_path_str_app: str):
    css_path_app = Path(css_path_str_app) # settings.STYLE_CSS_PATH_WEB isator.

import streamlit as st
import sys
import logging # Ensure logging is imported before use
from pathlib import Path
import html 

# --- Robust Path Setup: This MUST be correct for all imports to work ---
# This script (app.py) is assumed to be in the project root: ssentinel_project_root/app.py
# Therefore, Path(__file__).resolve().parent IS the project root.
_project_root_dir = Path(__file__).resolve().parent 

if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    # Use print for initial setup messages that occur before logging is fully configured
    print(f"DEBUG (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    print(f"DEBUG (app.py): Project root '{_project_root_dir}' was already in sys.path.", file=sys.stderr)

# --- Import Settings (NOW that sys.path is correctly set) ---
try:
    from config import settings 
    # The print statement for successful import moved AFTER basicConfig, so logger can be used.
except ImportError as e_cfg_app:
    print(f"FATAL (app.py): Failed to import config.settings: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings: 
    print(f"FATAL (app.py): AttributeError during config.settings access (likely circular import or settings.py error): {e_attr_settings}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root for sys.path: {_project_root_dir}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL (app.py): Generic error during config.settings import: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration (Now that settings is imported) ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    # Use print here as logger might not be fully set if settings.LOG_LEVEL was bad
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str}' in settings. Defaulting to INFO.", file=sys.stderr)
    log_level_app_str = "INFO"

logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) # Logger for this app.py

logger.info(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}") # Log after basicConfig

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit # Ensure streamlit is imported before using st alias
    major_str, minor_str, patch_full_str = streamlit.__version__.split('.')
    major, minor = int(major_str), int(minor_str)
    patch_str = patch_full_str.split('-')[0] 
    patch = int(patch_str) 
    STREAMLIT_VERSION_GE_1_30 = (major >= 1 and minor >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger already absolute
    if css_path_app.is_file(): # is_file() implies exists()
        try:
            with open(css_path_app, "r", encoding="utf-8") as f_css_app: st.markdown(f'<style>{f_css_app.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app}")
        except Exception as e_css_main_app: logger.error(f"Error applying CSS {css_path_app}: {e_css_main_app}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_v2 = st.columns([0.12, 0.88]) # Renamed var for clarity
with header_cols_app_ui_v2[0]:
    l_logo_path_app_v2 = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_v2 = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_v2.is_file(): st.image(str(l_logo_path_app_v2), width=100)
    elif s_logo_path_app_v2.is_file(): st.image(str(s_logo_path_app_v2), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_v2}', S: '{s_logo_path_app_v2}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_v2[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator...""") # Full content
st.markdown("#### Core Design Principles:")
core_principles_main_app_v4 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v4 = min(len(core_principles_main_app_v4), 2)
if num_cols_core_principles_v4 > 0:
    cols_core_principles_ui_v4 = st.columns(num_cols_core_principles_v4)
    for idx_core_v4, (title_core_v4, desc_core_v4) in enumerate(core_principles_main_app_v4):
        with cols_core_principles_ui_v4[idx_core_v4 % num_cols_core_principles_v4]:
            st.markdown(f"##### {title_core_v4}"); st.markdown(f"<small>{html.escape(desc_core_v4)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...") # Full content
st.info("💡 **Note:** This web application serves as a high-level demonstrator...") # Full content
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app = _project_root_dir / "pages" 
role_navigation_config_app_final_v3 = [ # Assuming prefixed filenames for order
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj_app_cfg = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_cfg: str = str(page_icon_path_obj_app_cfg) if page_icon_path_obj_app_cfg.exists() and page_icon_path_obj_app_cfg.is_file() else "🌍"
if final_page_icon_str_app_cfg == "🌍": logger.warning(f"Page icon not found at '{page_icon_path_obj_app_cfg}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_cfg,
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
except Exception as e_theme_main_app_cfg: logger.error(f"Error applying Plotly theme: {e_theme_main_app_cfg}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles_app_cfg(css_path_str_app_cfg: str):
    css_path_app_cfg = Path(css_path_str_app_cfg)
    if css_path_app_cfg.exists() and css_path_app_cfg.is_file():
        try:
            with open(css_path_app_cfg, "r", encoding="utf-8") as f_css_app_cfg: st.markdown(f'<style>{f_css_app_cfg.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_cfg}")
        except Exception as e_css_main_app_cfg: logger.error(f"Error applying CSS {css_path_app_cfg}: {e_css_main_app_cfg}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_cfg}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_cfg(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_app_ui_cfg = st.columns([0.12, 0.88])
with header_cols_app_ui_cfg[0]:
    l_logo_path_app_cfg = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_cfg = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_cfg.is_file(): st.image(str(l_logo_path_app_cfg), width=100)
    elif s_logo_path_app_cfg.is_file(): st.image(str(s_logo_path_app_cfg), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_cfg}', S: '{s_logo_path_app_cfg}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_cfg[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator...""") # Full content
st.markdown("#### Core Design Principles:")
core_principles_main_app_v4 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v4 = min(len(core_principles_main_app_v4), 2)
if num_cols_core_principles_v4 > 0:
    cols_core_principles_ui_v4 = st.columns(num_cols_core_principles_v4)
    for idx_core_v4, (title_core_v4, desc_core_v4) in enumerate(core_principles_main_app_v4):
        with cols_core_principles_ui_v4[idx_core_v4 % num_cols_core_principles_v4]:
            st.markdown(f"##### {title_core_v4}"); st.markdown(f"<small>{html.escape(desc_core_v4)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...") # Full content
st.info("💡 **Note:** This web application serves as a high-level demonstrator...") # Full content
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_base_dir_app_cfg = _project_root_dir / "pages" 
role_navigation_config_app_cfg = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] # Full descriptions assumed

num_nav_cols_final_app_cfg = min(len(role_navigation_config_app_cfg), 2)
if num_nav_cols_final_app_cfg > 0:
    nav_cols_ui_final_app_cfg = st.columns(num_nav_cols_final_app_cfg)
    current_col_idx_nav_final_cfg = 0
    for nav_item_final_app_cfg in role_navigation_config_app_cfg:
        # For st.page_link, path must be relative to the `pages` directory.
        # e.g. "01_chw_dashboard.py" not "pages/01_chw_dashboard.py"
        page_link_target_app_cfg = nav_item_final_app_cfg['page_filename']
        physical_page_full_path_app_cfg = pages_base_dir_app_cfg / nav_item_final_app_cfg["page_filename"]
        
        if not physical_page_full_path_app_cfg.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg['title']}' not found: {physical_page_full_path_app_cfg}")
            continue

        with nav_cols_ui_final_app_cfg[current_col_idx_nav_final_cfg % num_nav_cols_final_app_cfg]:
            container_args_final_app", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] # Full descriptions assumed

num_nav_cols_final_app_v3 = min(len(role_navigation_config_app_final_v3), 2)
if num_nav_cols_final_app_v3 > 0:
    nav_cols_ui_final_app_v3 = st.columns(num_nav_cols_final_app_v3)
    current_col_idx_nav_final_v3 = 0
    for nav_item_final_app_v3 in role_navigation_config_app_final_v3:
        page_link_target_app_v3 = f"pages/{nav_item_final_app_v3['page_filename']}" 
        physical_page_full_path_app_v3 = pages_directory_obj_app / nav_item_final_app_v3["page_filename"]
        
        if not physical_page_full_path_app_v3.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_v3['title']}' not found: {physical_page_full_path_app_v3}")
            continue

        with nav_cols_ui_final_app_v3[current_col_idx_nav_final_v3 % num_nav_cols_final_app_v3]:
            container_args_final_app_v3 = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args_final_app_v3):
                st.subheader(f"{nav_item_final_app_v3['icon']} {html.escape(nav_item_final_app_v3['title'])}")
                st.markdown(f"<small>{nav_item_final_app_v3['desc']}</small>", unsafe_allow_html=True)
                link_label_final_app_v3 = f"Explore {nav_item_final_app_v3['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final_app_v3 = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target_app_v3, label=link_label_final_app_v3, icon="➡️", **link_kwargs_final_app_v3)
                else: 
                    # Fallback: href should be just the filename if Streamlit serves pages from the 'pages' dir directly
                    st.markdown(f'<a href="{nav_item_final_app_v3["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_v3} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_v3 += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
# ... (Full capabilities descriptions from prompt) ...
capabilities_data_app_final_v3_full = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_v3 = min(len(capabilities_data_app_final_v3_full), 3)
if num_cap_cols_final_app_v3 > 0:
    cap_cols_ui_final_app_v3 = st.columns(num_cap_cols_final_app_v3)
    for i_cap_final_v3, (cap_t_final_v3, cap_d_final_v3) in enumerate(capabilities_data_app_final_v3_full):
        with cap_cols_ui_final_app_v3[i_cap_final_v3 % num_cap_cols_final_app_v3]: 
            st.markdown(f"##### {html.escape(cap_t_final_v3)}")
            st.markdown(f"<small>{html.escape(cap_d_final_v3)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()

# For sidebar order, rely on filename prefixes (e.g., 05_glossary_page.py)
# The manual link is removed as Streamlit auto-discovery with prefixes is preferred.
glossary_page_filename_sb_final = "05_glossary_page.py" 
glossary_physical_path_sb_final = pages_directory_obj_app / glossary_page_filename_sb_final
if not glossary_physical_path_sb_final.exists():
    logger.warning(f"Glossary page file (expected as '{glossary_filename_sb_final}') not found for sidebar auto-discovery.")

st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
