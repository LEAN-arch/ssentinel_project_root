# sentinel_project_root/app.py
# SME PLATINUM STANDARD - APPLICATION ENTRY POINT (V6 - FINAL)

import logging
import sys
from pathlib import Path
import html

try:
    _project_root = Path(__file__).resolve().parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    import streamlit as st
    from config import settings
    from visualization import load_and_inject_css, set_plotly_theme
    
except ImportError as e:
    print(f"FATAL ERROR in app.py: A core module failed to import.", file=sys.stderr)
    print("1. Ensure you have run 'scripts/setup.sh' to install dependencies.", file=sys.stderr)
    print("2. Run the app from the project root: `cd sentinel_project_root && streamlit run app.py`", file=sys.stderr)
    print(f"\nPython Path: {sys.path}\nOriginal ImportError: {e}", file=sys.stderr)
    sys.exit(1)

# --- Global Configuration ---
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# --- SME FIX: Tame noisy third-party loggers for a cleaner console ---
# This sets the log level for these specific libraries to WARNING,
# so their INFO and DEBUG messages are suppressed.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


st.set_page_config(
    page_title=f"{settings.APP_NAME} - Overview",
    page_icon=str(settings.APP_LOGO_SMALL_PATH),
    layout="wide", initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{settings.SUPPORT_CONTACT_INFO}",
        "Report a bug": f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        "About": f"### {settings.APP_NAME} (v{settings.APP_VERSION})\n{settings.APP_FOOTER_TEXT}"
    }
)

load_and_inject_css(settings.STYLE_CSS_PATH)
set_plotly_theme()

# --- Application Header and Body ---
header_cols = st.columns([0.1, 0.9])
with header_cols[0]:
    st.image(str(settings.APP_LOGO_LARGE_PATH), width=110)
with header_cols[1]:
    st.title(settings.APP_NAME)
    st.subheader("Actionable Intelligence for Resilient Health Systems")
st.divider()

st.markdown(f"### Welcome to the {html.escape(settings.APP_NAME)}")
st.info("""
**💡 How to Use This Demo:** The dashboards linked below simulate the views available to **Supervisors, Clinic Managers, and Health Analysts**. 
""", icon="ℹ️")
st.divider()

st.header("Explore Simulated Role-Specific Dashboards")

pages_dir = _project_root / "pages"
if pages_dir.is_dir():
    page_files = sorted(pages_dir.glob("[0-9]*.py"))
    
    nav_cols = st.columns(min(len(page_files), 3))
    col_idx = 0
    for page_path in page_files:
        page_name = page_path.stem[3:].replace("_", " ")
        with nav_cols[col_idx % len(nav_cols)]:
            with st.container(border=True):
                st.subheader(page_name)
                st.page_link(str(page_path.relative_to(_project_root)), label=f"Explore {page_name} View", use_container_width=True, icon="➡️")
        col_idx += 1
else:
    st.warning("`pages` directory not found. Cannot display navigation links.")

st.divider()

with st.sidebar:
    st.header(f"{settings.APP_NAME}")
    st.caption(f"v{settings.APP_VERSION}")
    st.divider()
    st.info("Navigate between simulated dashboards using the links above or the list on the left.")
    st.divider()
    st.markdown(f"**{html.escape(settings.ORGANIZATION_NAME)}**")
    st.markdown(f"Contact: <a href='mailto:{settings.SUPPORT_CONTACT_INFO}'>{settings.SUPPORT_CONTACT_INFO}</a>", unsafe_allow_html=True)
    st.caption(settings.APP_FOOTER_TEXT)

logger.info("Main application page loaded successfully.")
