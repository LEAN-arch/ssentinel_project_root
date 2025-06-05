# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator

import streamlit as st
import logging
from pathlib import Path
import html
import sys
import importlib.util

# --- Robust Path Setup ---
_this_file_path = Path(__file__).resolve()
# Dynamically find project root by looking for requirements.txt
_project_root_dir = _this_file_path.parent
while not (_project_root_dir / "requirements.txt").exists() and _project_root_dir.parent != _project_root_dir:
    _project_root_dir = _project_root_dir.parent
if not (_project_root_dir / "requirements.txt").exists():
    print(f"FATAL (app.py): Could not find project root containing requirements.txt from {_this_file_path}", file=sys.stderr)
    sys.exit(1)

print(f"DEBUG (app.py): _project_root_dir = {_project_root_dir}", file=sys.stderr)
print(f"DEBUG (app.py): Initial sys.path = {sys.path}", file=sys.stderr)

# Ensure project root is first in sys.path
if str(_project_root_dir) not in sys.path:
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py): Added project root to sys.path: {_project_root_dir}", file=sys.stderr)
elif sys.path[0] != str(_project_root_dir):
    sys.path.remove(str(_project_root_dir))
    sys.path.insert(0, str(_project_root_dir))
    print(f"DEBUG (app.py): Moved project root to start of sys.path: {_project_root_dir}", file=sys.stderr)

# Remove config directory from sys.path if present
config_dir = _project_root_dir / "config"
if str(config_dir) in sys.path:
    print(f"WARN (app.py): Removing '{config_dir}' from sys.path before importing settings.", file=sys.stderr)
    sys.path.remove(str(config_dir))

# --- Import Settings ---
try:
    from config import settings
except ImportError as e_settings:
    print(f"FATAL (app.py): Failed to import config.settings: {e_settings}", file=sys.stderr)
    print(f"DEBUG (app.py): sys.path at failure = {sys.path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"FATAL (app.py): Error during config.settings import: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Logging Configuration ---
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level = str(settings.LOG_LEVEL).upper()
if log_level not in valid_log_levels:
    print(f"WARN (app.py): Invalid LOG_LEVEL '{log_level}'. Using INFO.", file=sys.stderr)
    log_level = "INFO"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info(f"Successfully imported config.settings. APP_NAME: {settings.APP_NAME}")

# --- Streamlit Version Check ---
try:
    import streamlit
    from packaging import version
    if version.parse(streamlit.__version__) < version.parse("1.30.0"):
        logger.warning(f"Streamlit version {streamlit.__version__} < 1.30.0. Some features may not work.")
    STREAMLIT_PAGE_LINK_AVAILABLE = hasattr(st, "page_link")
except ImportError:
    logger.critical("Streamlit library not found.")
    sys.exit(1)

# --- Dependency Check ---
if not importlib.util.find_spec("plotly"):
    logger.warning("Plotly not installed. Visualization features may fail.")

# --- Page Configuration ---
page_icon = Path(settings.APP_LOGO_SMALL_PATH)
if not page_icon.is_file():
    logger.warning(f"Page icon not found at '{page_icon}'. Using '🌍'.")
    page_icon = "🌍"
st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview",
    page_icon=str(page_icon) if isinstance(page_icon, Path) else page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}\n\nEdge-First Health Intelligence Co-Pilot."
    }
)

# --- Apply Plotly Theme & CSS ---
try:
    from visualization.plots import set_sentinel_plotly_theme
    set_sentinel_plotly_theme()
    logger.debug("Sentinel Plotly theme applied.")
except Exception as e:
    logger.error(f"Error applying Plotly theme: {e}", exc_info=True)
    st.error("Error applying visualization theme.")

@st.cache_resource
def load_global_css_styles(css_path: str):
    css_path = Path(css_path)
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
header_cols = st.columns([0.15, 0.85])
with header_cols[0]:
    large_logo = Path(settings.APP_LOGO_LARGE_PATH)
    small_logo = Path(settings.APP_LOGO_SMALL_PATH)
    if large_logo.is_file():
        st.image(str(large_logo), width=100)
    elif small_logo.is_file():
        st.image(str(small_logo), width=80)
    else:
        logger.warning(f"Logos not found: Large: '{large_logo}', Small: '{small_logo}'.")
        st.markdown("### 🌍", unsafe_allow_html=True)
with header_cols[1]:
    st.title(html.escape(settings.APP_NAME))
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()

# --- Welcome & System Description ---
st.markdown(f"""
    ## Welcome to the {html.escape(settings.APP_NAME)} Demonstrator
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk environments. It aims to convert 
    diverse data sources into life-saving, workflow-integrated decisions, even with 
    **minimal or intermittent internet connectivity.**
""")
st.markdown("#### Core Design Principles:")
core_principles = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols = min(len(core_principles), 2)
if num_cols > 0:
    cols = st.columns(num_cols)
    for idx, (title, desc) in enumerate(core_principles):
        with cols[idx % num_cols]:
            st.markdown(f"##### {html.escape(title)}")
            st.markdown(f"<small>{html.escape(desc)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("👈 **Navigate via the sidebar** to explore simulated web dashboards...")
st.info("💡 **Note:** This web application serves as a high-level demonstrator...")
st.divider()

# --- Role-Specific Dashboards ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate information available at higher tiers (Facility/Regional Nodes).")
pages_dir = _project_root_dir / "pages"
role_navigation = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data...", "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"},
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", "desc": "Simulates a dashboard for Clinic Managers at a Facility Node...", "page_filename": "02_clinic_dashboard.py", "icon": "🏥"},
    {"title": "🗺️ District Health Strategic Overview", "desc": "Presents a strategic dashboard for District Health Officers...", "page_filename": "03_district_dashboard.py", "icon": "🗺️"},
    {"title": "📊 Population Health Analytics Deep Dive", "desc": "A view designed for detailed epidemiological and analysis...", "page_filename": "04_population_dashboard.py", "icon": "📊"},
]
num_nav_cols = min(len(role_navigation), 2)
if num_nav_cols:
    nav_cols = st.columns(num_nav_cols)
    for i, item in enumerate(role_navigation):
        page_path = pages_dir / item["page_filename"]
        if not page_path.exists():
            logger.warning(f"Navigation page not found: {page_path}")
            continue
        with nav_cols[i % num_nav_cols]:
            with st.container():
                st.subheader(f"{item['icon']} {html.escape(item['title'])}")
                st.markdown(f"<small>{html.escape(item['desc'])}</small>", unsafe_allow_html=True)
                link_label = f"Explore {item['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    st.page_link(page=item["page_filename"], label=link_label, icon="🏆")
                else:
                    st.markdown(f'<a href="{item['page_filename']}" target="_self" style="display:block;padding:0.5em;color:white;background:#007bff;border-radius:4px;text-decoration:none;">{link_label} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Key Capabilities ---
st.header(f"{html.escape(settings.APP_NAME)} - Key Capabilities")
capabilities = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time monitoring..."),
    ("🌍 Offline-First Edge AI", "On-device intelligence..."),
    ("⚡ Actionable Insights", "Clear recommendations..."),
    ("🤝 Human-Centered UX", "Pictogram UIs..."),
    ("📡 Resilient Data Sync", "Flexible sharing..."),
    ("🌱 Scalable Architecture", "Modular design...")
]
num_cap_cols = min(len(capabilities), 3)
if num_cap_cols:
    cap_cols = st.columns(num_cap_cols)
    for i, (title, desc) in enumerate(capabilities):
        with cap_cols[i % num_cap_cols]:
            st.markdown(f"##### {html.escape(title)}")
            st.markdown(f"<small>{html.escape(desc)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.2rem;'></div>", unsafe_allow_html=True)
st.divider()

# --- Sidebar ---
st.sidebar.header(f"{html.escape(settings.APP_NAME)} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("## About")
st.sidebar.info("Web app simulates higher-level dashboards...")
st.sidebar.divider()
glossary_path = pages_dir / "05_glossary.py"
if glossary_path.exists():
    if STREAMLIT_PAGE_LINK_AVAILABLE:
        st.sidebar.page_link(glossary_path, label="📖 Glossary", icon="🏠")
    else:
        st.markdown(f'<a href="{glossary_path}" style="color:#007bff;">📖 Glossary</a>', unsafe_allow_html=True)
else:
    logger.warning(f"Glossary page not found: {glossary_path}")
    st.sidebar.markdown("📖 Glossary: Unavailable")
st.sidebar.divider()
st.sidebar.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**")
st.sidebar.markdown(f'<a href="mailto:{settings.SUPPORT_CONTACT_INFO}">{settings.SUPPORT_CONTACT_INFO}</a>', unsafe_allow_html=True)
st.sidebar.caption(html.escape(settings.APP_FOOTER_TEXT))
logger.info(f"{settings.APP_NAME} - System Overview loaded.")

