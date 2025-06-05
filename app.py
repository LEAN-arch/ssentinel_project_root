# sentinel_project_root/config/settings.py
# Centralized Configuration for "Sentinel Health Co-Pilot"

import os # Standard library, fine
import logging # Standard library, fine
from datetime import datetime # Standard library, fine
from pathlib import Path # Standard library, fine
import sys # THIS IS A RED FLAG if used to modify sys.path here.

# --- Base Project Directory ---
# This assumes settings.py is in sentinel_project_root/config/
# So, parent of 'config' is 'sentinel_project_root'
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
# REMOVE ANY DEBUG PRINTS THAT MODIFY SYS.PATH OR COULD CAUSE ISSUES BEFORE APP_NAME IS DEFINED
# For example, if there was a print statement here that tried to use another module from project root,
# and that module imported settings, it could be an issue.
# The print statement "print(f"DEBUG settings.py: PROJECT_ROOT_DIR resolved to: {PROJECT_ROOT_DIR}", file=sys.stderr)"
# is fine as it doesn't import other project modules.

# --- Logger for Path Validation ---
settings_logger = logging.getLogger(__name__) # Fine

# --- Path Validation Helper ---
# This function itself is fine.
def validate_path(path_obj: Path, description: str, is_dir: bool = False) -> Path:
    abs_path = path_obj.resolve()
    if not abs_path.exists():
        settings_logger.warning(f"{description} not found at resolved absolute path: {abs_path}")
    # ... rest of validate_path ...
    return abs_path

# --- I. Core System & Directory Configuration ---
# These are direct assignments, no problematic imports.
APP_NAME = "Sentinel Health Co-Pilot"
APP_VERSION = "4.0.2"
# ... other direct assignments ...

# Directories (now resolved to absolute paths)
# These use PROJECT_ROOT_DIR defined above and validate_path. This should be fine.
ASSETS_DIR = validate_path(PROJECT_ROOT_DIR / "assets", "Assets directory", is_dir=True)
DATA_SOURCES_DIR = validate_path(PROJECT_ROOT_DIR / "data_sources", "Data sources directory", is_dir=True)
# ... other path definitions ...

# --- Ensure LOG_LEVEL is defined before any logging calls that might depend on it if they were in settings.py ---
# This is already handled as LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT are derived from os.getenv first.

# CRITICAL CHECK: Ensure settings.py does NOT have any top-level imports from other parts of 
# your ssentinel_project_root (e.g., `from data_processing import something`)
# because those modules would then try to import `settings` causing a circular dependency.
# The current version of settings.py you provided seems clean of such top-level project-internal imports.

# The `sys.path` addition of `/mount/src/ssentinel_project_root/config`
# is extremely puzzling if `app.py` is adding `/mount/src/ssentinel_project_root`.
# One possibility: Is there an `__init__.py` in `ssentinel_project_root/config/`
# that is manipulating `sys.path`? This is unconventional but possible.

# Let's try an aggressive approach in `app.py` to ensure the `config` directory is NOT in `sys.path`
# right before importing `settings`.

# The current settings.py file content seems okay and shouldn't cause the path pollution by itself.
# The issue is almost certainly how the execution environment (Streamlit run) and `app.py`'s initial `sys.path` setup interact.

I will provide the `app.py` again, with an explicit check and removal of the `config` directory from `sys.path` if it's found, right before importing `settings`. This is a diagnostic and corrective measure.

File 54 (Corrected `app.py` with aggressive `sys.path` cleanup before importing `settings`):
```python
# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html 

# --- Robust Path Setup: This MUST be correct for all imports to work ---
_this_file_path = Path(__file__).resolve()
_project_root_dir = _this_file_path.parent 

print(f"DEBUG (app.py initial): _project_root_dir = {_project_root_dir}", file=sys.stderr)
print(f"DEBUG (app.py initial): Current sys.path = {sys.path}", file=sys.stderr)

# Ensure the project root is the first thing in sys.path
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    if sys.path[0] != str(_project_root_dir):
        try: sys.path.remove(str(_project_root_dir))
        except ValueError: pass 
        sys.path.insert(0, str(_project_root_dir))
        print(f"DEBUG (app.py): Moved project root to start of sys.path: {_project_root_dir}", file=sys.stderr)
    else:
        print(f"DEBUG (app.py): Project root '{_project_root_dir}' was already at start of sys.path.", file=sys.stderr)

# AGGRESSIVE CLEANUP: Explicitly remove <project_root>/config from sys.path if present
# This is to counteract whatever might be adding it.
config_dir_in_path_to_remove = str(_project_root_dir / "config")
if config_dir_in_path_to_remove in sys.path:
    print(f"WARN (app.py): Found '{config_dir_in_path_to_remove}' in sys.path. Removing it before importing settings.", file=sys.stderr)
    sys.path.remove(config_dir_in_path_to_remove)
    print(f"DEBUG (app.py): sys.path after removing config dir = {sys.path}", file=sys.stderr)

# --- Import Settings (NOW that sys.path is hopefully pristine) ---
try:
    from config import settings 
except ImportError as e_cfg_app:
    print(f"FATAL (app.py): Failed to import config.settings AFTER aggressive cleanup: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    sys.exit(1) 
except AttributeError as e_attr_settings: 
    print(f"FATAL (app.py): AttributeError during config.settings access AFTER aggressive cleanup (likely circular import persists): {e_attr_settings}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL (app.py): Generic error during config.settings import AFTER aggressive cleanup: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str}'. Using INFO.", file=sys.stderr); log_level_app_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)
logger.info(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}")


# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit 
    major_str, minor_str, patch_full_str = streamlit.__version__.split('.')
    major, minor = int(major_str), int(minor_str)
    patch_str_numeric = "".join(filter(str.isdigit, patch_full_str.split('-')[0]))
    patch = int(patch_str_numeric) if patch_str_numeric else 0
    STREAMLIT_VERSION_GE_1_30 = (major >= 1 and minor >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj_app_cfg = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_cfg: str = str(page_icon_path_obj_app_cfg) if page_icon_path_obj_app_cfg.is_file() else "🌍"
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
    if css_path_app_cfg.is_file():
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

# --- Welcome & System Description (Full content from prompt assumed) ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator...""")
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
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...")
st.info("💡 **Note:** This web application serves as a high-level demonstrator...")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app_cfg = _project_root_dir / "pages" 
role_navigation_config_app_cfg = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
]

num_nav_cols_final_app_cfg_val = min(len(role_navigation_config_app_cfg), 2)
if num_nav_cols_final_app_cfg_val > 0:
    nav_cols_ui_final_app_cfg_val = st.columns(num_nav_cols_final_app_cfg_val)
    current_col_idx_nav_final_cfg_val = 0
    for nav_item_final_app_cfg_item in role_navigation_config_app_cfg:
        page_link_target_app_cfg_item = nav_item_final_app_cfg_item['page_filename'] 
        physical_page_full_path_app_cfg_item = pages_directory_obj_app_cfg / nav_item_final_app_cfg_item["page_filename"]
        if not physical_page_full_path_app_cfg_item.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg_item['title']}' not found: {physical_page_full_path_app_cfg_item}")
            continue
        with nav_cols_ui_final_app_cfg_val[current_col_idx_nav_final_cfg_val % num_nav_cols_final_app_cfg_val]:
            container_args_final_app_cfg = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args_final_app_cfg):
                st.subheader(f"{nav_item_final_app_cfg_item['icon']} {html.escape(nav_item_final_app_cfg_item['title'])}")
                st.markdown(f"<small>{nav_item_final_app_cfg_item['desc']}</small>", unsafe_allow_html=True)
                link_label_final_app_cfg = f"Explore {nav_item_final_app_cfg_item['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final_app_cfg = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target_app_cfg_item, label=link_label_final_app_cfg, icon="➡️", **link_kwargs_final_app_cfg)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_cfg_item["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_cfg} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_cfg_val += 1
st.divider()

st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app_final_cfg_full = [ # Full descriptions assumed
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring..."), ("🌍 Offline-First Edge AI", "On-device intelligence..."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations..."), ("🤝 Human-Centered & Accessible UX", "Pictogram UIs..."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing..."), ("🌱 Scalable & Interoperable Architecture", "Modular design...")
]
num_cap_cols_final_app_cfg_val_ui = min(len(capabilities_data_app_final_cfg_full), 3)
if num_cap_cols_final_app_cfg_val_ui > 0:
    cap_cols_ui_final_app_cfg_val_ui = st.columns(num_cap_cols_final_app_cfg_val_ui)
    for i_cap_final_cfg_ui, (cap_t_final_cfg_ui, cap_d_final_cfg_ui) in enumerate(capabilities_data_app_final_cfg_full):
        with cap_cols_ui_final_app_cfg_val_ui[i_cap_final_cfg_ui % num_cap_cols_final_app_cfg_val_ui]: 
            st.markdown(f"##### {html.escape(cap_t_final_cfg_ui)}"); st.markdown(f"<small>{html.escape(cap_d_final_cfg_ui)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider(); st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()
glossary_filename_sidebar_cfg_val = "05_glossary_page.py" 
glossary_link_target_sidebar_cfg_val = glossary_filename_sidebar_cfg_val 
glossary_physical_path_final_sb_cfg_val = pages_directory_obj_app_cfg / glossary_filename_sidebar_cfg_val
if glossary_physical_path_final_sb_cfg_val.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE: st.sidebar.page_link(glossary_link_target_sidebar_cfg_val, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_cfg_val}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page for sidebar (expected: {glossary_physical_path_final_sb_cfg_val}) not found.")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**"); st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)
logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
