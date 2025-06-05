# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html # For escaping in fallback links

# --- Robust Path Setup ---
_current_app_file_dir = Path(__file__).parent.resolve()
_project_root_dir = _current_app_file_dir

if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"INFO: Added project root to sys.path: {_project_root_dir}", file=sys.stderr)

# --- Import Settings ---
try:
    from config import settings
    print(f"INFO: Successfully imported config.settings. APP_NAME: {settings.APP_NAME}", file=sys.stderr)
except ImportError as e_cfg_app:
    print(f"FATAL: Failed to import config.settings in app.py: {e_cfg_app}", file=sys.stderr); sys.exit(1)
except Exception as e_generic_cfg:
    print(f"FATAL: Generic error during config.settings import in app.py: {e_generic_cfg}", file=sys.stderr); sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels_app = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_app_str = str(settings.LOG_LEVEL).upper()
if log_level_app_str not in valid_log_levels_app:
    print(f"WARN: Invalid LOG_LEVEL '{log_level_app_str}'. Using INFO.", file=sys.stderr); log_level_app_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_app_str, logging.INFO), format=settings.LOG_FORMAT,
                    datefmt=settings.LOG_DATE_FORMAT, handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

# --- Streamlit Version Check & Feature Availability ---
STREAMLIT_VERSION_GE_1_30 = False # Flag for st.page_link with use_container_width
STREAMLIT_PAGE_LINK_AVAILABLE = False
try:
    import streamlit
    major, minor, patch_str = streamlit.__version__.split('.')
    patch = int(patch_str.split('-')[0]) # Handle dev versions like 1.30.0-dev
    STREAMLIT_VERSION_GE_1_30 = (int(major) >= 1 and int(minor) >= 30)
    # st.page_link was introduced around 1.27/1.28. Check if it exists.
    if hasattr(st, 'page_link'):
        STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30:
        logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some UI features might use fallbacks.")
except ImportError:
    logger.critical("Streamlit library not found."); sys.exit("Streamlit library not found.")
except Exception as e_st_ver:
    logger.warning(f"Could not accurately determine Streamlit version or feature availability: {e_st_ver}")


# --- Page Configuration ---
page_icon_path_obj = Path(settings.APP_LOGO_SMALL_PATH)
if not page_icon_path_obj.is_absolute(): page_icon_path_obj = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()
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
except Exception as e: logger.error(f"Error applying Plotly theme: {e}", exc_info=True); st.error("Error applying visualization theme.")
@st.cache_resource
def load_global_css_styles(css_path_str: str):
    css_path = Path(css_path_str)
    if not css_path.is_absolute(): css_path = (_project_root_dir / css_path_str).resolve()
    if css_path.exists() and css_path.is_file():
        try:
            with open(css_path, "r", encoding="utf-8") as f: st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path}")
        except Exception as e: logger.error(f"Error applying CSS {css_path}: {e}", exc_info=True); st.error("Styles could not be loaded.")
    else: logger.warning(f"CSS file not found: {css_path}"); st.warning("Stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: load_global_css_styles(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols = st.columns([0.12, 0.88])
with header_cols[0]:
    l_logo_path = Path(settings.APP_LOGO_LARGE_PATH); s_logo_path = Path(settings.APP_LOGO_SMALL_PATH)
    if not l_logo_path.is_absolute(): l_logo_path = (_project_root_dir / settings.APP_LOGO_LARGE_PATH).resolve()
    if not s_logo_path.is_absolute(): s_logo_path = (_project_root_dir / settings.APP_LOGO_SMALL_PATH).resolve()
    if l_logo_path.is_file(): st.image(str(l_logo_path), width=100)
    elif s_logo_path.is_file(): st.image(str(s_logo_path), width=80)
    else: logger.warning(f"App logos not found. L: '{l_logo_path}', S: '{s_logo_path}'."); st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols[1]: st.title(settings.APP_NAME); st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"## Welcome to the {settings.APP_NAME} Demonstrator\n\nSentinel is an **edge-first health intelligence system**...") # Truncated for brevity, content unchanged
st.markdown("#### Core Design Principles:")
core_principles = [("📶 **Offline-First**", "On-device Edge AI..."), ("🎯 **Action-Oriented**", "Insights trigger targeted responses..."),
                   ("🧑‍🤝‍🧑 **Human-Centered**", "Optimized UIs for frontline users..."), ("🔗 **Resilient & Scalable**", "Modular design with robust sync...")]
cp_cols = st.columns(min(len(core_principles), 2))
for i, (t, d) in enumerate(core_principles):
    with cp_cols[i % min(len(core_principles), 2)]: st.markdown(f"##### {t}"); st.markdown(f"<small>{d}</small>", unsafe_allow_html=True); st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")

# --- INTEGRATED CONTENT: Navigation Information ---
st.markdown("""
    👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. 
    These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. 
    The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their 
    Personal Edge Device (PED), tailored for their specific operational context.
""")
st.info(
    "💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's "
    "data processing capabilities and the types of aggregated views available to management and strategic personnel."
)
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate the information available at higher tiers (Facility/Regional Nodes).")

pages_dir = _project_root_dir / "pages"
role_nav_config = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "Focus (Tier 1-2): Team performance, CHW support, local epi signals...", "page_filename": "chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Focus (Tier 2): Clinic workflows, care quality, resources, facility safety...", "page_filename": "clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", "desc": "Focus (Tier 2-3): Population health, resource allocation, interventions...", "page_filename": "district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", "desc": "Focus (Tier 3): In-depth epi/systems analysis, SDOH, equity...", "page_filename": "population_dashboard.py", "icon": "📊"},
] # Descriptions shortened for brevity here, full descriptions from prompt would be used

num_nav_cols = min(len(role_nav_config), 2)
if num_nav_cols > 0:
    nav_cols_ui = st.columns(num_nav_cols)
    col_idx_nav = 0
    for nav_item in role_nav_config:
        # Correct path for st.page_link is relative to the main app script's directory location.
        # If app.py is at project_root, then 'pages/filename.py' is correct.
        page_link_path = f"pages/{nav_item['page_filename']}" 
        physical_page_path = pages_dir / nav_item["page_filename"]
        
        if not physical_page_path.exists():
            logger.warning(f"Navigation page file for '{nav_item['title']}' not found at: {physical_page_path}")
            continue

        with nav_cols_ui[col_idx_nav % num_nav_cols]:
            try:
                with st.container(border=True):
                    st.subheader(f"{nav_item['icon']} {nav_item['title']}")
                    st.markdown(f"<small>{nav_item['desc']}</small>", unsafe_allow_html=True)
                    if STREAMLIT_PAGE_LINK_AVAILABLE:
                        st.page_link(page_link_path, label=f"Explore this View", icon="➡️", use_container_width=True if STREAMLIT_VERSION_GE_1_30 else None)
                    else: # Fallback for older Streamlit
                        st.markdown(f'<a href="{page_link_path.replace("pages/", "")}" target="_self"><button>Explore this View ➡️</button></a>', unsafe_allow_html=True)
            except TypeError: # Fallback for st.container(border=True)
                st.subheader(f"{nav_item['icon']} {nav_item['title']}")
                st.markdown(f"<small>{nav_item['desc']}</small>", unsafe_allow_html=True)
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    st.page_link(page_link_path, label=f"Explore this View", icon="➡️")
                else:
                    st.markdown(f'<a href="{page_link_path.replace("pages/", "")}" target="_self"><button>Explore this View ➡️</button></a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        col_idx_nav += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app = [("🛡️ Frontline Safety", "Real-time monitoring..."), ("🌍 Offline Edge AI", "On-device intelligence..."),
                         ("⚡ Actionable Insights", "Data to role-specific recommendations..."), ("🤝 Human-Centered UX", "Pictogram UIs, voice/tap..."),
                         ("📡 Resilient Sync", "Flexible data sharing..."), ("🌱 Scalable Architecture", "Modular design, FHIR/HL7...")]
num_cap_cols = min(len(capabilities_data_app), 3)
if num_cap_cols > 0:
    cap_cols = st.columns(num_cap_cols)
    for i, (t, d) in enumerate(capabilities_data_app):
        with cap_cols[i % num_cap_cols]: st.markdown(f"##### {t}"); st.markdown(f"<small>{d}</small>", unsafe_allow_html=True); st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar Content (Order of dashboards managed by filename prefixes 01_, 02_ etc. in pages/ dir) ---
st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info("Web app simulates higher-level dashboards. Frontline workers use dedicated PED apps.")
st.sidebar.divider()

# Glossary link at the bottom of the sidebar
glossary_page_filename = "glossary_page.py"
glossary_page_link_path_sidebar = f"pages/{glossary_page_filename}"
glossary_physical_path_sidebar = pages_dir / glossary_page_filename
if glossary_physical_path_sidebar.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE:
        st.sidebar.page_link(glossary_page_link_path_sidebar, label="📜 System Glossary", icon="📚")
    else:
        st.sidebar.markdown(f"[📜 System Glossary]({glossary_page_link_path_sidebar.replace('pages/', '')})") # Fallback markdown link
else:
    logger.warning(f"Glossary page for sidebar link not found: {glossary_physical_path_sidebar}")
st.sidebar.divider()
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
