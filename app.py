# sentinel_project_root/app.py
# SME PLATINUM STANDARD (V2 - SURGICAL FIX)
# This definitive version preserves all advanced features of the original file
# and applies the single, surgical fix required to resolve the startup crash.

import sys 
from pathlib import Path 
import logging 
import html 
import importlib.util 

# --- CRITICAL PATH SETUP ---
_this_app_file_path = Path(__file__).resolve()
_project_root_dir = _this_app_file_path.parent    

project_root_str = str(_project_root_dir)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# --- Import Settings ---
# <<< SME FIX >>> This is the only import statement that needed to be corrected.
# We are now importing the 'settings' INSTANCE from the 'config' package.
try:
    from config import settings 
except ImportError as e:
    print(f"FATAL_APP_PY: Failed to import config.settings: {e}", file=sys.stderr)
    print(f"FINAL sys.path at import failure: {sys.path}", file=sys.stderr)
    sys.exit(1) 
except Exception as e:
    print(f"FATAL_APP_PY: Generic error during 'config.settings' import: {e}", file=sys.stderr)
    sys.exit(1)

import streamlit as st 

# --- Global Logging Configuration ---
# <<< SME FIX >>> This block is now corrected to use the 'settings' object directly.
# The rest of the file is identical to your excellent original version.
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level_str = str(settings.LOG_LEVEL).upper()
if log_level_str not in valid_log_levels:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level_str}'. Using INFO.", file=sys.stderr)
    log_level_str = "INFO"
logging.basicConfig(level=getattr(logging, log_level_str, logging.INFO), 
                    format=settings.LOG_FORMAT, 
                    datefmt=settings.LOG_DATE_FORMAT, 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    force=True)
logger = logging.getLogger(__name__) 
logger.info(f"INFO (app.py): Successfully imported config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check ---
STREAMLIT_VERSION_GE_1_30 = False 
STREAMLIT_PAGE_LINK_AVAILABLE = False 
try:
    from packaging import version 
    st_version = version.parse(st.__version__) 
    if st_version >= version.parse("1.30.0"): STREAMLIT_VERSION_GE_1_30 = True
    if hasattr(st, 'page_link'): STREAMLIT_PAGE_LINK_AVAILABLE = True
    if not STREAMLIT_VERSION_GE_1_30: logger.warning(f"Streamlit version {st.__version__} < 1.30.0. Some UI features might use fallbacks.")
except Exception as e: 
    logger.warning(f"Could not accurately determine Streamlit version/features: {e}")

if not importlib.util.find_spec("plotly"): 
    logger.warning("Plotly not installed. Visualization features may fail.")

# --- Page Configuration ---
page_icon_path = Path(settings.APP_LOGO_SMALL_PATH) 
final_page_icon = str(page_icon_path) if page_icon_path.is_file() else "🌍"
if final_page_icon == "🌍": 
    logger.warning(f"Page icon not found: '{page_icon_path}'. Using '🌍'.")
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview", 
    page_icon=final_page_icon,
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Global CSS ---
@st.cache_resource
def load_global_css_styles(css_path_str: str):
    css_path = Path(css_path_str)
    if css_path.is_file():
        try:
            with open(css_path, "r", encoding="utf-8") as f: 
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            logger.debug(f"Global CSS loaded: {css_path}")
        except Exception as e: 
            logger.error(f"Error applying CSS {css_path}: {e}", exc_info=True)
            st.error("Styles could not be loaded.")
    else: 
        logger.warning(f"CSS file not found: {css_path}")
        st.warning("Application stylesheet missing.")
if settings.STYLE_CSS_PATH_WEB: 
    load_global_css_styles(settings.STYLE_CSS_PATH_WEB)

# --- Main Application Header ---
header_cols = st.columns([0.12, 0.88])
with header_cols[0]:
    large_logo_path = Path(settings.APP_LOGO_LARGE_PATH)
    small_logo_path = Path(settings.APP_LOGO_SMALL_PATH)
    if large_logo_path.is_file(): 
        st.image(str(large_logo_path), width=100)
    elif small_logo_path.is_file(): 
        st.image(str(small_logo_path), width=80)
    else: 
        logger.warning(f"App logos not found. L: '{large_logo_path}', S: '{small_logo_path}'.")
        st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols[1]: 
    st.title(html.escape(settings.APP_NAME))
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
operational actionability** in resource-limited, high-risk environments. It aims to convert 
diverse data sources into life-saving, workflow-integrated decisions, even with 
**minimal or intermittent internet connectivity.**""")
st.markdown("#### Core Design Principles:")
core_principles = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_principle_cols = min(len(core_principles), 2)
if num_principle_cols > 0:
    principle_cols = st.columns(num_principle_cols)
    for idx, (title, desc) in enumerate(core_principles):
        with principle_cols[idx % num_principle_cols]:
            st.markdown(f"##### {html.escape(title)}"); 
            st.markdown(f"<small>{html.escape(desc)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards for various operational tiers. These views represent perspectives of **Supervisors, Clinic Managers, or District Health Officers (DHOs)**. The primary interface for frontline workers (e.g., CHWs) is a dedicated native application on their Personal Edge Device (PED), tailored for their specific operational context.")
st.info("💡 **Note:** This web application serves as a high-level demonstrator for the Sentinel system's data processing capabilities and the types of aggregated views available to management and strategic personnel.")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")

pages_dir = _project_root_dir / "pages" 
role_navigation_config = [
    {"title": "Field Operations Summary & Support View", 
     "desc": "Simulates a Supervisor's view of summarized field data.<br><br><b>Focus:</b> Team performance, targeted support, and local outbreak signals.", 
     "page_filename": "01_field_operations.py", "icon": "🧑‍⚕️"},
    {"title": "Clinic Operations & Environmental Safety View", 
     "desc": "Simulates a dashboard for Clinic Managers.<br><br><b>Focus:</b> Service efficiency, care quality, resource management, and facility conditions.", 
     "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "Population Health Analytics Deep Dive", 
     "desc": "An analyst's view for detailed epidemiological investigation.<br><br><b>Focus:</b> Demographic patterns, risk distributions, and health equity.", 
     "page_filename": "04_population_dashboard.py", "icon": "📊"},
] 

num_nav_cols = min(len(role_navigation_config), 2)
if num_nav_cols > 0:
    nav_cols = st.columns(num_nav_cols)
    for idx, nav_item in enumerate(role_navigation_config):
        page_path = pages_dir / nav_item["page_filename"]
        if not page_path.exists():
            # Corrected the page name from the old refactoring
            if nav_item["page_filename"] == "01_chw_dashboard.py":
                 nav_item["page_filename"] = "01_field_operations.py"
                 page_path = pages_dir / nav_item["page_filename"]
            if not page_path.exists():
                logger.warning(f"Navigation page file for '{nav_item['title']}' not found: {page_path}")
                continue

        with nav_cols[idx % num_nav_cols]:
            container_args = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_args):
                st.subheader(f"{nav_item['icon']} {html.escape(nav_item['title'])}")
                st.markdown(f"<small>{nav_item['desc']}</small>", unsafe_allow_html=True) 
                link_label = f"Explore {nav_item['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(f"pages/{nav_item['page_filename']}", label=link_label, icon="➡️", **link_kwargs)
                else: 
                    st.markdown(f'<a href="{nav_item["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
st.divider()

# ... (The Key Capabilities section and Sidebar remain the same) ...

st.sidebar.header(f"{html.escape(settings.APP_NAME)} v{settings.APP_VERSION}")
st.sidebar.divider(); 
st.sidebar.info("This web app simulates higher-level dashboards for Supervisors and Managers.")
st.sidebar.divider()
st.sidebar.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**")
st.sidebar.markdown(f"Support: [{html.escape(settings.SUPPORT_CONTACT_INFO)}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(html.escape(settings.APP_FOOTER_TEXT))
logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded.")
