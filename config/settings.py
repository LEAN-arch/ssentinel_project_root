# sentinel_project_root/app.py
# Main Streamlit application for Sentinel Health Co-Pilot Demonstrator.

import streamlit as st
import sys
import logging
from pathlib import Path
import html 

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
    else: logger.warning(f"CSS file not found: {css_path}"); st.warning("Application stylesheet missing.")
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
st.markdown(f"""
    ## Welcome to the {settings.APP_NAME} Demonstrator
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk environments. It aims to convert 
    diverse data sources into life-saving, workflow-integrated decisions, even with 
    **minimal or intermittent internet connectivity.**
""")
st.markdown("#### Core Design Principles:")
core_principles_data_app_main_v2 = [
    ("📶 **Offline-First Operations**", "On-device Edge AI ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Insights aim to trigger clear, targeted responses relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces optimized for low-literacy, high-stress users, prioritizing immediate understanding."),
    ("🔗 **Resilience & Scalability**", "Modular design for scaling from personal devices to regional views with robust data sync.")
]
num_cols_principles_main_v2 = min(len(core_principles_data_app_main_v2), 2)
if num_cols_principles_main_v2 > 0:
    cols_principles_ui_main_v2 = st.columns(num_cols_principles_main_v2)
    for idx_principle_main_v2, (title_main_p_v2, desc_main_p_v2) in enumerate(core_principles_data_app_main_v2):
        with cols_principles_ui_main_v2[idx_principle_main_v2 % num_cols_principles_main_v2]:
            st.markdown(f"##### {title_main_p_v2}")
            st.markdown(f"<small>{html.escape(desc_main_p_v2)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)
st.markdown("---")

# --- Navigation Information ---
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

pages_base_dir_app_final = _project_root_dir / "pages" 

# Assuming filenames in 'pages/' directory are prefixed for order:
# e.g., 01_chw_dashboard.py, 02_clinic_dashboard.py, etc.
# The st.page_link paths should reflect these prefixed filenames.
role_navigation_config_final = [
    {"title": "🧑‍⚕️ CHW Operations Summary & Field Support View (Supervisor/Hub Level)", 
     "desc": "This view simulates how a CHW Supervisor or a Hub coordinator might access summarized data from CHW Personal Edge Devices (PEDs).<br><br><b>Focus (Tier 1-2):</b> Team performance monitoring, targeted support for CHWs, localized outbreak signal detection based on aggregated CHW reports.<br><b>Key Data Points:</b> CHW activity summaries (visits, tasks completed), patient alert escalations, critical supply needs for CHW kits, early epidemiological signals from specific zones.<br><b>Objective:</b> Enable supervisors to manage CHW teams effectively, provide timely support, identify emerging health issues quickly, and coordinate local responses. The CHW's primary tool is their offline-first native app on their PED, providing real-time alerts & task management.", 
     "page_filename": "01_chw_dashboard.py", "icon": "🧑‍⚕️"}, # Assumed prefixed filename
    {"title": "🏥 Clinic Operations & Environmental Safety View (Facility Node Level)", 
     "desc": "Simulates a dashboard for Clinic Managers at a Facility Node (Tier 2), providing insights into service efficiency, care quality, resource management, and environmental conditions.<br><br><b>Focus (Tier 2):</b> Optimizing clinic workflows, ensuring quality patient care, managing supplies and testing backlogs, monitoring clinic environment for safety and infection control.<br><b>Key Data Points:</b> Clinic performance KPIs (e.g., test TAT, patient throughput), supply stock forecasts, IoT sensor data summaries (CO2, PM2.5, occupancy), clinic-level epidemiological trends, flagged patient cases for review.<br><b>Objective:</b> Enhance operational efficiency, support clinical decision-making, maintain resource availability, and ensure a safe clinic environment.", 
     "page_filename": "02_clinic_dashboard.py", "icon": "🏥"}, # Assumed prefixed filename
    {"title": "🗺️ District Health Strategic Overview (DHO at Facility/Regional Node Level)", 
     "desc": "Presents a strategic dashboard for District Health Officers (DHOs), typically accessed at a Facility Node (Tier 2) or a Regional/Cloud Node (Tier 3).<br><br><b>Focus (Tier 2-3):</b> Population health insights, resource allocation across zones, monitoring environmental well-being, and planning targeted interventions.<br><b>Key Data Points:</b> District-wide health KPIs, interactive maps for zonal comparisons (risk, disease burden, resources), trend analyses, intervention planning tools based on aggregated data.<br><b>Objective:</b> Support evidence-based strategic planning, public health interventions, program monitoring, and policy development for the district.", 
     "page_filename": "03_district_dashboard.py", "icon": "🗺️"}, # Assumed prefixed filename
    {"title": "📊 Population Health Analytics Deep Dive (Epidemiologist/Analyst View - Tier 3)", 
     "desc": "A view designed for detailed epidemiological and health systems analysis, typically used by analysts or program managers at a Regional/Cloud Node (Tier 3) with access to more comprehensive, aggregated datasets.<br><br><b>Focus (Tier 3):</b> In-depth analysis of demographic patterns, SDOH impacts, clinical trends, health system performance, and equity across broader populations.<br><b>Key Data Points:</b> Stratified disease burden, AI risk distributions by various factors, aggregated test positivity trends, comorbidity analysis, referral pathway performance, health equity metrics.<br><b>Objective:</b> Provide robust analytical capabilities to understand population health dynamics, evaluate interventions, identify areas for research, and inform large-scale public health strategy.", 
     "page_filename": "04_population_dashboard.py", "icon": "📊"}, # Assumed prefixed filename
]

num_nav_cols_final_app = min(len(role_navigation_config_final), 2)
if num_nav_cols_final_app > 0:
    nav_cols_ui_final_app = st.columns(num_nav_cols_final_app)
    current_col_idx_nav_final = 0
    for nav_item_final_app in role_navigation_config_final:
        page_link_path_final_app = f"pages/{nav_item_final_app['page_filename']}" 
        physical_page_path_final_app = pages_base_dir_app_final / nav_item_final_app["page_filename"]
        
        if not physical_page_path_final_app.exists():
            logger.warning(f"Navigation page file for '{nav_item_final_app['title']}' not found: {physical_page_path_final_app}")
            continue

        with nav_cols_ui_final_app[current_col_idx_nav_final % num_nav_cols_final_app]:
            container_kwargs_final = {"border": True} if STREAMLIT_VERSION_GE_1_30 else {}
            with st.container(**container_kwargs_final):
                st.subheader(f"{nav_item_final_app['icon']} {html.escape(nav_item_final_app['title'])}")
                st.markdown(f"<small>{nav_item_final_app['desc']}</small>", unsafe_allow_html=True)
                link_label_final = f"Explore {nav_item_final_app['title'].split('(')[0].split('View')[0].strip()} View"
                if STREAMLIT_PAGE_LINK_AVAILABLE:
                    link_kwargs_final = {"use_container_width": True} if STREAMLIT_VERSION_GE_1_30 else {}
                    st.page_link(page_link_path_final_app, label=link_label_final, icon="➡️", **link_kwargs_final)
                else: 
                    st.markdown(f'<a href="{nav_item_final_app["page_filename"]}" target="_self" style="display:block;text-align:center;padding:0.5em;background-color:var(--sentinel-color-action-primary);color:white;border-radius:4px;text-decoration:none;">{link_label_final} ➡️</a>', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        current_col_idx_nav_final += 1
st.divider()

# --- Key Capabilities Section ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data_app_final_val = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, safety nudges on PEDs."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, guidance without continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "Raw data to clear, role-specific recommendations integrated into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram UIs, voice/tap commands, local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across devices/tiers."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design (personal to national), FHIR/HL7 considerations for integration.")
]
num_cap_cols_final_app = min(len(capabilities_data_app_final_val), 3)
if num_cap_cols_final_app > 0:
    cap_cols_ui_final_app = st.columns(num_cap_cols_final_app)
    current_col_idx_cap_final = 0
    for cap_title_item_final, cap_desc_item_final in capabilities_data_app_final_val:
        with cap_cols_ui_final_app[current_col_idx_cap_final % num_cap_cols_final_app]:
            st.markdown(f"##### {html.escape(cap_title_item_final)}")
            st.markdown(f"<small>{html.escape(cap_desc_item_final)}</small>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
        current_col_idx_cap_final += 1
st.divider()

# --- Sidebar Content ---
# Streamlit automatically generates sidebar links from files in the 'pages' directory,
# sorted alphanumerically by filename. To control order, prefix filenames:
# e.g., pages/01_chw_dashboard.py, pages/02_clinic_dashboard.py, ..., pages/05_glossary_page.py
# The app.py (home page) will implicitly be the first item.

st.sidebar.header(f"{settings.APP_NAME} v{settings.APP_VERSION}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info("Web app simulates higher-level dashboards. Frontline workers use dedicated PED apps.")
st.sidebar.divider()

# Example: If you want to *ensure* glossary is last and control its label precisely,
# you could use st.page_link here, but it would appear *after* auto-discovered pages
# unless you manage all navigation explicitly (e.g. using st.navigation, which is more complex).
# For default behavior, relying on filename prefixes like "05_glossary_page.py" is standard.
# This manual link is mostly for illustration if explicit addition is desired.
glossary_page_filename_sidebar = "05_glossary_page.py" # ASSUMING you rename the file
glossary_page_link_path_sidebar_final = f"pages/{glossary_page_filename_sidebar}"
glossary_physical_path_sidebar_final = pages_base_dir_app_final / glossary_page_filename_sidebar

if glossary_physical_path_sidebar_final.exists():
    # This will add a link, but its position relative to auto-discovered pages
    # depends on Streamlit's rendering order.
    # For guaranteed order, prefixing filenames in the pages/ directory is the primary method.
    if STREAMLIT_PAGE_LINK_AVAILABLE:
         pass # Streamlit will auto-discover prefixed files. Manual link not needed here if files are prefixed.
         # st.sidebar.page_link(glossary_page_link_path_sidebar_final, label="📜 System Glossary", icon="📚") # Example if needed
    else:
         st.sidebar.markdown(f'<a href="{glossary_page_filename_sidebar}" target="_self">📜 System Glossary</a>', unsafe_allow_html=True)
else:
    logger.warning(f"Glossary page file for sidebar link not found (expected: {glossary_physical_path_sidebar_final})")

st.sidebar.divider() # Add divider before org info
st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page loaded successfully.")
