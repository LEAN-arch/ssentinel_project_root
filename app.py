# sentinel_project_root/app.py

import sys # sys must be imported first for path manipulation
from pathlib import Path # Pathlib for robust path operations

# --- CRITICAL PATH SETUP ---
# This section MUST execute correctly before almost any other import from the project.
# Goal: Ensure `_project_root_dir` is correctly `ssentinel_project_root`
#       and it's the first entry in `sys.path`.

_this_app_file_path = Path(__file__).resolve() # Absolute path to this app.py
_project_root_dir = _this_file_path.parent    # If app.py is in project root, its parent dir IS the project root.

# Initial diagnostic prints
print(f"DEBUG (app.py @ initial): __file__ = {__file__}", file=sys.stderr)
print(f"DEBUG (app.py @ initial): _this_app_file_path = {_this_app_file_path}", file=sys.stderr)
print(f"DEBUG (app.py @ initial): Calculated _project_root_dir = {_project_root_dir}", file=sys.stderr)
print(f"DEBUG (app.py @ initial): Initial sys.path = {sys.path}", file=sys.stderr)

# Add/Prioritize project root in sys.path
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py @ sys.path insert): Added '{_project_root_dir}' to sys.path.", file=sys.stderr)
elif sys.path[0] != str(_project_root_dir):
    try:
        sys.path.remove(str(_project_root_dir))
    except ValueError: # Not found, which is unexpected if it's in sys.path but not at [0]
        print(f"WARN (app.py @ sys.path reorder): '{_project_root_dir}' reported in sys.path but not found by remove().", file=sys.stderr)
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py @ sys.path reorder): Moved '{_project_root_dir}' to start of sys.path.", file=sys.stderr)
else:
    print(f"DEBUG (app.py @ sys.path check): Project root '{_project_root_dir}' is already at start of sys.path.", file=sys.stderr)

print(f"DEBUG (app.py @ final pre-settings-import): sys.path = {sys.path}", file=sys.stderr)

# --- Import Settings (This is the critical point) ---
try:
    # Now, 'from config import settings' should resolve correctly if:
    # 1. _project_root_dir is indeed '/mount/src/ssentinel_project_root'
    # 2. '/mount/src/ssentinel_project_root' is now at sys.path[0]
    # 3. There is a 'config' directory under _project_root_dir
    # 4. 'config' directory has an '__init__.py'
    # 5. 'config' directory has 'settings.py'
    from config import settings 
except ImportError as e_cfg_app_final_attempt:
    print(f"FATAL (app.py): STILL FAILED to import config.settings: {e_cfg_app_final_attempt}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    print(f"Project Root check: Is '{_project_root_dir / 'config' / '__init__.py'}' a file? {( _project_root_dir / 'config' / '__init__.py').is_file()}", file=sys.stderr)
    print(f"Project Root check: Is '{_project_root_dir / 'config' / 'settings.py'}' a file? {( _project_root_dir / 'config' / 'settings.py').is_file()}", file=sys.stderr)
    sys.exit(1)
except AttributeError as e_attr_settings_final_attempt: 
    print(f"FATAL (app.py): AttributeError (likely circular import) on config.settings: {e_attr_settings_final_attempt}", file=sys.stderr)
    print(f"FINAL sys.path at attribute error: {sys.path}", file=sys.stderr)
    sys.exit(1)

# --- Imports that depend on settings ---
import streamlit as st # Moved Streamlit import after settings if it might use it indirectly via logging
import logging # Moved logging import after settings if it uses LOG_LEVEL etc.
import html 
import importlib.util
# from pathlib Path already imported

# --- Global Logging Configuration ---
valid_log_levels_app_final_cfg = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str_final_cfg = str(settings.LOG_LEVEL).upper()
if log_level_app_str_final_cfg not in valid_log_levels_app_final_cfg:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str_final_cfg}'. Using INFO.", file=sys.stderr); log_level_app_str_final_cfg = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str_final_cfg, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) 
logger.info(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    from packaging import version 
    if version.parse(st.__version__) >= version.parse("1.30.0"): STREAMLIT_VERSION_GE_1_30 = True
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {st.__version__} < 1.30.0. Some UI features might use fallbacks.")
except Exception as e_st_ver_final_cfg: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver_final_cfg}")

if not importlib.util.find_spec("plotly"): logger.warning("Plotly not installed. Visualization features may fail.")

# --- Page Configuration ---
page_icon_path_obj_app_main_cfg_val = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon_str_app_main_cfg_val: str = str(page_icon_path_obj_app_main_cfg_val) if page_icon_path_obj_app_main_cfg_val.is_file() else "🌍"
if final_page_icon_str_app_main_cfg_val == "🌍": logger.warning(f"Page icon not found: '{page_icon_path_obj_app_main_cfg_val}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str_app_main_cfg_val,
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
except Exception as e_theme_final_cfg: logger.error(f"Error applying Plotly theme: {e_theme_final_cfg}", exc_info=True); st.error("Error applying visualization theme.")
@st.cache_resource
def load_global_css_styles_app_final_cfg_val(css_path_str_app_final_cfg_val: str):
    css_path_app_final_cfg_val = Path(css_path_str_app_final_cfg_val)
    if css_path_app_final_cfg_val.is_file():
        try:
            with open(css_path_app_final_cfg_val, "r", encoding="utf-8") as f_css_app_final_cfg_val: st.markdown(f'<style>{f_css_app_final_cfg_val.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path_app_final_cfg_val}")
        except Exception as e_css_main_app_final_cfg_val: logger.error(f"Error applying CSS {css_path_app_final_cfg_val}: {e_css_main_app_final_cfg_val}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path_app_final_cfg_val}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles_app_final_cfg_val(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
# ... (rest of the app.py content from File 54, ensuring html.escape for user-facing strings from settings) ...
# For brevity, I will not repeat the entire UI part if it was unchanged from the previous correct version.
# The critical part was the sys.path and settings import.

# Ensure the content below uses html.escape for settings values displayed in markdown
header_cols_app_ui_final_cfg_val_ui = st.columns([0.12, 0.88])
with header_cols_app_ui_final_cfg_val_ui[0]:
    l_logo_path_app_final_cfg_val = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_app_final_cfg_val = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_app_final_cfg_val.is_file(): st.image(str(l_logo_path_app_final_cfg_val), width=100)
    elif s_logo_path_app_final_cfg_val.is_file(): st.image(str(s_logo_path_app_final_cfg_val), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_app_final_cfg_val}', S: '{s_logo_path_app_final_cfg_val}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_app_ui_final_cfg_val_ui[1]: st.title(html.escape(settings.APP_NAME)); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

st.markdown(f"""## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
# ... (core principles loop with html.escape) ...
core_principles_main_app_v5 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_core_principles_v5 = min(len(core_principles_main_app_v5), 2)
if num_cols_core_principles_v5 > 0:
    cols_core_principles_ui_v5 = st.columns(num_cols_core_principles_v5)
    for idx_core_v5, (title_core_v5, desc_core_v5) in enumerate(core_principles_main_app_v5):
        with cols_core_principles_ui_v5[idx_core_v5 % num_cols_core_principles_v5]:
            st.markdown(f"##### {html.escape(title_core_v5)}"); st.markdown(f"<small>{html.escape(desc_core_v5)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their Personal Edge Device (PED), tailored for their specific operational context.")
st.info(f"💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's data processing capabilities and the types of aggregated views available to management and strategic personnel.")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_directory_obj_app_final_cfg_val = _project_root_dir / "pages" 
role_navigation_config_app_final_cfg_list_val = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] 

num_nav_cols_final_app_cfg_val_ui_v3 = min(len(role_navigation_config_app_final_cfg_list_val), 2)
if num_nav_cols_final_app_cfg_val_ui_v3 > 0:
    nav_cols_ui_final_app_cfg_val_ui_v3 = st.columns(num_nav_cols_final_app_cfg_val_ui_v3)
    current_col_idx_nav_final_cfg_val_ui_v3 = 0
    for nav_item_final_app_cfg_item_val_v3 in role_navigation_config_app_final_cfg_list_val:
        page_link_target_app_cfg_item_val_v3 = nav_item_final_app_cfg_item_val_v3['page_filename'] 
        physical_page_full_path_app_cfg_item_val_v3 = pages_directory_obj_app_final_cfg_val / nav_item_final_app_cfg_item_val_v3["page_filename"]
        if not physical_page_full_path_app_cfg_item_val_v3.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app_cfg_item_val_v3['title']}' not found: {physical_page_full_path_app_cfg_item_val_v3}")
            continue
        with nav_cols_ui_final_app_cfg_val_ui_v3[current_col_idx_nav_final_cfg_val_ui_v3 % num_nav_cols_final_app_cfg_val_ui_v3]:
            container_args_final_app_cfg_val_v3 = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args_final_app_cfg_val_v3):
                st.subheader(f"{nav_item_final_app_cfg_item_val_v3['icon']} {html.escape(nav_item_final_app_cfg_item_val_v3['title'])}")
                st.markdown(f"<small>{nav_item_final_app_cfg_item_val_v3['desc']}</small>", unsafe_allow_html=True) # Assuming desc is safe HTML
                link_label_final_app_cfg_val_v3 = f"Explore {nav_item_final_app_cfg_item_val_v3['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final_app_cfg_val_v3 = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target_app_cfg_item_val_v3, label=link_label_final_app_cfg_val_v3, icon="➡️", **link_kwargs_final_app_cfg_val_v3)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app_cfg_item_val_v3["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final_app_cfg_val_v3} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_cfg_val_ui_v3 += 1
st.divider()

st.header(f"{html.escape(settings.APP_NAME)} - Key Capabilities Reimagined")
# ... (Full capabilities descriptions) ...
capabilities_data_app_final_cfg_full_v3 = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app_cfg_val_final_v3 = min(len(capabilities_data_app_final_cfg_full_v3), 3)
if num_cap_cols_final_app_cfg_val_final_v3 > 0:
    cap_cols_ui_final_app_cfg_val_final_v3 = st.columns(num_cap_cols_final_app_cfg_val_final_v3)
    for i_cap_final_cfg_final_v3, (cap_t_final_cfg_final_v3, cap_d_final_cfg_final_v3) in enumerate(capabilities_data_app_final_cfg_full_v3):
        with cap_cols_ui_final_app_cfg_val_final_v3[i_cap_final_cfg_final_v3 % num_cap_cols_final_app_cfg_val_final_v3]: 
            st.markdown(f"##### {html.escape(cap_t_final_cfg_final_v3)}"); st.markdown(f"<small>{html.escape(cap_d_final_cfg_final_v3)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

st.sidebar.header(f"{html.escape(settings.APP_NAME)} v{settings.APP_VERSION}")
st.sidebar.divider(); st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()
glossary_filename_sidebar_cfg_final_val_v3 = "05_glossary_page.py" 
glossary_link_target_sidebar_cfg_final_val_v3 = glossary_filename_sidebar_cfg_final_val_v3 
glossary_physical_path_final_sb_cfg_final_val_v3 = pages_directory_obj_app_cfg / glossary_filename_sidebar_cfg_final_val_v3
if glossary_physical_path_final_sb_cfg_final_val_v3.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE: st.sidebar.page_link(glossary_link_target_sidebar_cfg_final_val_v3, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_cfg_final_val_v3}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page for sidebar (expected: {glossary_physical_path_final_sb_cfg_final_val_v3}) not found.")
st.sidebar.divider()
st.sidebar.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**"); st.sidebar.markdown(f"Support: [{html.escape(settings.SUPPORT_CONTACT_INFO)}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(html.escape(settings.APP_FOOTER_TEXT))
logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
