# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html 

# --- Robust Path Setup ---
# app.py is in the project root (ssentinel_project_root)
_project_root_dir = Path(__file__).parent.resolve() 

if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    # Use print for initial setup messages before logging is configured
    print(f"INFO (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
else:
    print(f"INFO (app.py): Project root already in sys.path: {_project_root_dir}", file=sys.stderr)


# --- Import Settings ---
try:
    from config import settings # This should now work correctly
    print(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL (app.py): Failed to import config.settings: {e_cfg_app}", file=sys.stderr)
    print(f"PYTHONPATH for app.py: {sys.path}", file=sys.stderr)
    print(f"Calculated project root: {_project_root_dir}", file=sys.stderr)
    sys.exit(1) 
except Exception as e_generic_cfg:
    print(f"FATAL (app.py): Generic error during config.settings import: {e_generic_cfg}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_app_str}'. Using INFO.", file=sys.stderr); log_level_app_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__) # Logger for this app.py

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit
    major, minor, patch_str = streamlit.__version__.split('.')
    patch = int(patch_str.split('-')[0]) 
    STREAMLIT_VERSION_GE_1_30 = (int(major) >= 1 and int(minor) >= 30)
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError: logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver: logger.warning(f"Could not accurately determine Streamlit version/features: {e_st_ver}")

# --- Page Configuration ---
page_icon_path_obj = Path(settings.APP_LOGO_SMALL_PATH) # settings.APP_LOGO_SMALL_PATH is already absolute string
final_page_icon_str: str = str(page_icon_path_obj) if page_icon_path_obj.exists() and page_icon_path_obj.is_file() else "🌍"
if final_page_icon_str == "🌍": logger.warning(f"Page icon not found at '{page_icon_path_obj}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", page_icon=final_page_icon_str,
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
except Exception as e_theme_app_main: logger.error(f"Error applying Plotly theme: {e_theme_app_main}", exc_info=True); st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles(css_path_str: str):
    css_path = Path(css_path_str) # settings.STYLE_CSS_PATH_WEB is already absolute
    if css_path.exists() and css_path.is_file():
        try:
            with open(css_path, "r", encoding="utf-8") as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path}")
        except Exception as e_css_app_main: logger.error(f"Error applying CSS {css_path}: {e_css_app_main}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path}"); st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols_main = st.columns([0.12, 0.88])
with header_cols_main[0]:
    # Paths from settings are already absolute
    l_logo_path_main_app = Path(settings.APP_LOGO_LARGE_PATH)
    s_logo_path_main_app = Path(settings.APP_LOGO_SMALL_PATH)
    if l_logo_path_main_app.is_file(): st.image(str(l_logo_path_main_app), width=100)
    elif s_logo_path_main_app.is_file(): st.image(str(s_logo_path_main_app), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path_main_app}', S: '{s_logo_path_main_app}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols_main[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {settings.APP_NAME} Demonstrator...""") # Content unchanged, truncated for brevity
st.markdown("#### Core Design Principles:")
core_principles_main_app = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality..."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses..."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users..."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling...")] # Full descriptions from prompt assumed
num_cols_core_principles = min(len(core_principles_main_app), 2)
if num_cols_core_principles > 0:
    cols_core_principles_ui = st.columns(num_cols_core_principles)
    for idx_core, (title_core, desc_core) in enumerate(core_principles_main_app):
        with cols_core_principles_ui[idx_core % num_cols_core_principles]:
            st.markdown(f"##### {title_core}"); st.markdown(f"<small>{html.escape(desc_core)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...") # Content unchanged
st.info("💡 **Note:** This web application serves as a high-level demonstrator...") # Content unchanged
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_dir_app_final_val = _project_root_dir / "pages" 
# Ensure filenames here match the *actual* filenames in your pages directory (including prefixes if you added them for ordering)
role_nav_config_app_final = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2)...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Presents a strategic dashboard for District Health Officers (DHOs)...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "A view designed for detailed epidemiological and health systems analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
] # Full descriptions from prompt assumed

num_nav_cols_final = min(len(role_nav_config_app_final), 2)
if num_nav_cols_final > 0:
    nav_cols_ui_final = st.columns(num_nav_cols_final)
    current_col_idx_nav_final_val = 0
    for nav_item_final in role_navigation_config_app_final:
        # For st.page_link, the path should be 'pages/actual_filename_in_pages_dir.py'
        # If pages are in 'ssentinel_project_root/pages/', and app.py is in 'ssentinel_project_root/'
        page_link_target = f"pages/{nav_item_final['page_filename']}"
        physical_page_full_path = pages_dir_app_final_val / nav_item_final["page_filename"]
        
        if not physical_page_full_path.exists():
            logger.warning(f"Navigation page file for '{nav_item_final['title']}' not found: {physical_page_full_path}")
            continue

        with nav_cols_ui_final[current_col_idx_nav_final_val % num_nav_cols_final]:
            container_args = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args):
                st.subheader(f"{nav_item_final['icon']} {html.escape(nav_item_final['title'])}")
                st.markdown(f"<small>{nav_item_final['desc']}</small>", unsafe_allow_html=True)
                link_label_val = f"Explore {nav_item_final['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_args = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_target, label=link_label_val, icon="➡️", **link_args)
                else: 
                    st.markdown(f'<a href="{nav_item_final["page_filename"]}" target="_self" style="text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;display:block;">{link_label_val} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final_val += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app_final = [("🛡️ Frontline Worker Safety & Support", "Real-time vitals..."), ("🌍 Offline-First Edge AI", "On-device intelligence..."),
                               ("⚡ Actionable, Contextual Insights", "Raw data to clear recommendations..."), ("🤝 Human-Centered UX", "Pictogram UIs..."),
                               ("📡 Resilient Data Sync", "Flexible data sharing..."), ("🌱 Scalable Architecture", "Modular design...")] # Full descriptions
num_cap_cols_final = min(len(capabilities_data_app_final), 3)
if num_cap_cols_final > 0:
    cap_cols_ui_final = st.columns(num_cap_cols_final)
    for i_cap, (cap_t, cap_d) in enumerate(capabilities_data_app_final):
        with cap_cols_ui_final[i_cap % num_cap_cols_final]: st.markdown(f"##### {html.escape(cap_t)}"); st.markdown(f"<small>{html.escape(cap_d)}</small>", unsafe_allow_html=True); st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Content & Glossary Link ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:"); st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()

# Glossary link - assuming it's named "05_glossary_page.py" for ordering
glossary_filename_sidebar_final = "05_glossary_page.py" 
glossary_link_target_sidebar = f"pages/{glossary_filename_sidebar_final}"
glossary_physical_path_final_sb = pages_dir_app_final_val / glossary_filename_sidebar_final

if glossary_physical_path_final_sb.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE: st.sidebar.page_link(glossary_link_target_sidebar, label="📜 System Glossary", icon="📚")
    else: st.sidebar.markdown(f'<a href="{glossary_filename_sidebar_final}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else: logger.warning(f"Glossary page for sidebar not found: {glossary_physical_path_final_sb}")

st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
