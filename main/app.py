# sentinel_project_root/app.py
# Main Streamlit application file for the "Sentinel Health Co-Pilot" System Overview.

import streamlit as st
import sys
import os
import logging

# --- Robust Path Setup for Imports ---
# This ensures that 'config', 'data_processing', 'analytics', 'visualization'
# can be imported directly.
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
if _current_file_dir not in sys.path:
    sys.path.insert(0, _current_file_dir) # Add project root to sys.path

try:
    from config import settings # Import from the new settings module
    from visualization.ui_elements import render_kpi_card # Import new UI element renderer
    from visualization.plots import set_sentinel_plotly_theme # Import theme setter
except ImportError as e_import_app:
    # This is a critical failure if basic modules can't be imported.
    critical_error_msg = (
        f"CRITICAL IMPORT ERROR in app.py: {e_import_app}. Python Path: {sys.path}. "
        f"Ensure 'config/settings.py' and other core modules are correctly placed relative to 'app.py' "
        "and that all dependencies from requirements.txt are installed."
    )
    print(critical_error_msg, file=sys.stderr)
    # Attempt to show error in Streamlit if st is available, otherwise raise
    if 'st' in globals() and hasattr(st, 'error'):
        st.error(critical_error_msg)
        st.stop()
    else:
        raise ImportError(critical_error_msg) from e_import_app

# --- Global Logging Configuration ---
# Configure logging once for the entire application.
# Uses settings from config.settings.
logging.basicConfig(
    level=getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO),
    format=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)], # Log to stdout, Streamlit captures this
    force=True # Override any existing Streamlit logger configs if necessary
)
logger = logging.getLogger(__name__) # Logger for this main app file

# --- Apply Plotly Theme Globally ---
# This ensures all Plotly charts in the app use the custom theme.
try:
    set_sentinel_plotly_theme()
    logger.info("Sentinel Plotly theme applied globally.")
except Exception as e_theme:
    logger.error(f"Failed to apply Sentinel Plotly theme: {e_theme}", exc_info=True)
    st.warning("Warning: Custom chart theme could not be applied. Charts may use default styling.")


# --- Page Configuration (set_page_config must be the first Streamlit command) ---
page_icon_to_use = "🌍" # Default fallback icon
if os.path.exists(settings.APP_LOGO_SMALL_PATH):
    page_icon_to_use = settings.APP_LOGO_SMALL_PATH
else:
    logger.warning(f"Small app logo not found at '{settings.APP_LOGO_SMALL_PATH}'. Using fallback page icon.")

st.set_page_config(
    page_title=f"{settings.APP_NAME} - System Overview",
    page_icon=page_icon_to_use,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Help Request - {settings.APP_NAME}",
        'Report a bug': f"mailto:{settings.SUPPORT_CONTACT_INFO}?subject=Bug Report - {settings.APP_NAME} v{settings.APP_VERSION}",
        'About': f"""
### {settings.APP_NAME} (v{settings.APP_VERSION})
{settings.APP_FOOTER_TEXT}

An Edge-First Health Intelligence Co-Pilot designed for resource-limited environments.
This demonstrator showcases higher-level views. Frontline workers use dedicated native PED apps.
"""
    }
)

# --- CSS Loading ---
# Load custom CSS styles. This is usually done once.
# The style_web_reports.css should contain all necessary global styles.
# Specific component styles if needed can be included there or handled by Streamlit's theming.
@st.cache_resource # Cache the CSS loading to avoid re-reading file on every script rerun
def load_global_styles(css_file_path: str):
    if os.path.exists(css_file_path):
        try:
            with open(css_file_path, "r", encoding="utf-8") as f_css:
                st.markdown(f'<style>{f_css.read()}</style>', unsafe_allow_html=True)
            logger.info(f"Global CSS styles loaded from: {css_file_path}")
        except Exception as e_css:
            logger.error(f"Error reading global CSS file {css_file_path}: {e_css}", exc_info=True)
            st.error("Critical error: Application styles could not be loaded.")
    else:
        logger.warning(f"Global CSS file not found: {css_file_path}. Application will use default styles.")
        # Optionally, provide a less alarming message if CSS is considered optional enhancement:
        # st.caption("Note: Custom application styles not found. Display may use default theming.")

if settings.STYLE_CSS_PATH_WEB: # Check if path is configured
    load_global_styles(settings.STYLE_CSS_PATH_WEB)
else:
    logger.info("No global CSS path configured in settings. Skipping custom CSS load.")


# --- Main Application Header ---
header_cols = st.columns([0.12, 0.88]) # Adjust column ratios as needed
with header_cols[0]:
    logo_path_to_use = settings.APP_LOGO_LARGE_PATH
    # Fallback to small logo if large one isn't found
    if not os.path.exists(logo_path_to_use):
        logo_path_to_use = settings.APP_LOGO_SMALL_PATH
    
    if os.path.exists(logo_path_to_use):
        st.image(logo_path_to_use, width=90) # Adjust width as needed
    else:
        logger.warning("Large and Small app logos not found. Displaying fallback text/icon.")
        st.markdown("### 🌍", unsafe_allow_html=True) # Fallback icon if no logo
with header_cols[1]:
    st.title(settings.APP_NAME)
    st.subheader("Transforming Data into Lifesaving Action at the Edge")
st.divider()


# --- Enhanced Welcome & System Description ---
st.markdown(f"""
    ## Welcome to the Sentinel Health Co-Pilot
    
    Sentinel is an **edge-first health intelligence system** designed for **maximum clinical and 
    operational actionability** in resource-limited, high-risk LMIC environments. It converts 
    diverse data sources—wearables, IoT, contextual inputs—into life-saving, workflow-integrated 
    decisions, even with **minimal or no internet connectivity.**
""")

st.markdown("#### Core Principles Guiding Sentinel:")
# Using Streamlit columns for a cleaner layout of core principles
core_principles_cols = st.columns(2)
core_principles_data = [
    ("📶 **Offline-First Operations**", "On-device Edge AI on Personal Edge Devices (PEDs) ensures critical functionality without continuous connectivity."),
    ("🎯 **Action-Oriented Intelligence**", "Every insight aims to trigger a clear, targeted response relevant to frontline workflows."),
    ("🧑‍🤝‍🧑 **Human-Centered Design**", "Interfaces are optimized for low-literacy, high-stress users, prioritizing immediate understanding and ease of use on PEDs."),
    ("🔗 **Resilience & Scalability**", "Modular design allows scaling from individual PEDs to facility and regional views, with robust, flexible data synchronization mechanisms.")
]

for idx, (title, desc) in enumerate(core_principles_data):
    with core_principles_cols[idx % 2]: # Distribute items into columns
        st.markdown(f"##### {title}")
        st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True) # Vertical spacing


st.markdown("""
    ---
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

# --- Simulated Role-Specific Views Section (Navigation to Pages) ---
st.header("Explore Simulated Role-Specific Dashboards")
st.caption("These views demonstrate the information available at higher tiers (Facility/Regional Nodes). Frontline workers use dedicated native PED apps.")

# Navigation details for each dashboard page
# Page paths are relative to the `pages` directory. Streamlit handles this.
# Keys should be unique for Streamlit elements.
role_navigation_config = [
    {
        "title": "🧑‍⚕️ CHW Operations & Field Support (Supervisor)", 
        "description": (
            "Simulates how a CHW Supervisor monitors team performance, provides targeted support, "
            "and detects localized outbreak signals from aggregated CHW PED data.\n\n"
            "- **Focus (Tier 1-2):** Team performance, CHW support, local epi signals.\n"
            "- **Key Data:** CHW activity, patient alerts, supply needs, early symptom clusters.\n"
            "- **Objective:** Enable effective CHW team management and rapid local response."
        ),
        "page_path": "chw_dashboard", # File name in 'pages' dir without .py
        "key": "nav_chw_dashboard_main"
    },
    {
        "title": "🏥 Clinic Operations & Environmental Safety (Facility Node)", 
        "description": (
            "Dashboard for Clinic Managers providing insights into service efficiency, care quality, "
            "resource management, and clinic environmental safety (e.g., air quality, occupancy).\n\n"
            "- **Focus (Tier 2):** Clinic workflow optimization, patient care quality, supply management, facility safety.\n"
            "- **Key Data:** Clinic KPIs (TAT, throughput), supply forecasts, IoT sensor data, epi trends.\n"
            "- **Objective:** Enhance operational efficiency, support clinical decisions, ensure safe environment."
        ),
        "page_path": "clinic_dashboard",
        "key": "nav_clinic_dashboard_main"
    },
    {
        "title": "🗺️ District Health Strategic Overview (DHO)", 
        "description": (
            "Strategic dashboard for District Health Officers, aggregating data for population health insights, "
            "resource allocation, environmental well-being monitoring, and intervention planning.\n\n"
            "- **Focus (Tier 2-3):** Population health, resource allocation, environmental monitoring, strategic interventions.\n"
            "- **Key Data:** District KPIs, interactive zonal maps, trend analyses, intervention planning tools.\n"
            "- **Objective:** Support evidence-based strategic planning and public health program monitoring."
        ),
        "page_path": "district_dashboard",
        "key": "nav_district_dashboard_main"
    },
    {
        "title": "📊 Population Health Analytics Deep Dive (Analyst)", 
        "description": (
            "View for detailed epidemiological and health systems analysis, used by analysts or program managers "
            "with access to comprehensive, aggregated datasets.\n\n"
            "- **Focus (Tier 3):** In-depth analysis of demographics, SDOH impacts, clinical trends, health system performance.\n"
            "- **Key Data:** Stratified disease burden, AI risk distributions, comorbidity analysis, health equity metrics.\n"
            "- **Objective:** Provide robust analytics for public health strategy and research."
        ),
        "page_path": "population_dashboard",
        "key": "nav_population_analytics_main"
    },
]

# Display navigation options in a 2-column layout for better space utilization
nav_cols = st.columns(2)
for i, nav_item in enumerate(role_navigation_config):
    with nav_cols[i % 2]: # Alternate columns
        with st.container(border=True): # Use bordered container for each item
            st.subheader(nav_item["title"])
            st.markdown(f"<small>{nav_item['description']}</small>", unsafe_allow_html=True)
            
            button_label = f"Explore {nav_item['title'].split('(')[0].strip()}"
            # Streamlit's st.page_link is preferred for multipage app navigation
            st.page_link(f"pages/{nav_item['page_path']}.py", label=button_label, icon="➡️", use_container_width=True)
            # Old button logic with st.switch_page can be removed if st.page_link is used.
            # if st.button(button_label, key=nav_item["key"], type="primary", use_container_width=True):
            #     st.switch_page(f"pages/{nav_item['page_path']}.py") # Correct path for switch_page
            st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True) # Small bottom margin
st.divider()


# --- Key Capabilities Reimagined Section (3x2 Layout) ---
st.header(f"{settings.APP_NAME} - Key Capabilities Reimagined")
capabilities_data = [
    ("🛡️ Frontline Worker Safety & Support", "Real-time vitals/environmental monitoring, fatigue detection, and safety nudges on Personal Edge Devices (PEDs)."),
    ("🌍 Offline-First Edge AI", "On-device intelligence for alerts, prioritization, and guidance with zero reliance on continuous connectivity."),
    ("⚡ Actionable, Contextual Insights", "From raw data to clear, role-specific recommendations that integrate into field workflows."),
    ("🤝 Human-Centered & Accessible UX", "Pictogram-based UIs, voice/tap commands, and local language support for low-literacy, high-stress users on PEDs."),
    ("📡 Resilient Data Synchronization", "Flexible data sharing (Bluetooth, QR, SD card, SMS, opportunistic IP) across PEDs, Hubs, and Nodes."),
    ("🌱 Scalable & Interoperable Architecture", "Modular design from personal to national levels, with FHIR/HL7 considerations for system integration.")
]

# Use two rows of columns for a 3x2 layout
cap_row1 = st.columns(3)
cap_row2 = st.columns(3)
all_cap_cols = cap_row1 + cap_row2 # Combine column lists

for i, (cap_title, cap_desc) in enumerate(capabilities_data):
    if i < len(all_cap_cols): # Check to avoid index error if less than 6 capabilities
        with all_cap_cols[i]:
            with st.container(border=False): # No border for individual capability items here for cleaner look
                st.markdown(f"##### {cap_title}")
                st.markdown(f"<small>{cap_desc}</small>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)
st.divider()


# --- Link to the Glossary page ---
with st.expander("📜 **System Glossary & Terminology**", expanded=False):
    st.markdown(
        "Explore definitions for key terms, metrics, and system components specific to the "
        f"{settings.APP_NAME}. Understanding this terminology is crucial for interpreting "
        "the dashboards and system outputs effectively."
    )
    # Using st.page_link for navigation to the glossary page
    st.page_link("pages/glossary_page.py", label="Go to Glossary", icon="📚")
    # Old button logic:
    # if st.button("Go to Glossary", key="nav_glossary_from_home_main", type="secondary"):
    #     st.switch_page("pages/glossary_page.py") # Ensure filename matches

# --- Sidebar Content ---
st.sidebar.header(f"{settings.APP_NAME}")
st.sidebar.divider()
st.sidebar.markdown("#### About This Demonstrator:")
st.sidebar.info(
    "This web app simulates higher-level dashboards for Supervisors, Clinic Managers, DHOs, and Analysts. "
    "Frontline health worker interaction occurs on dedicated Personal Edge Devices (PEDs) with native applications."
)
st.sidebar.markdown("---")

# Glossary link in sidebar using st.page_link
st.sidebar.page_link("pages/glossary_page.py", label="📜 System Glossary", icon="📚")
st.sidebar.divider()

st.sidebar.markdown(f"**{settings.ORGANIZATION_NAME}**")
st.sidebar.markdown(f"Support: [{settings.SUPPORT_CONTACT_INFO}](mailto:{settings.SUPPORT_CONTACT_INFO})")
st.sidebar.divider()
st.sidebar.caption(settings.APP_FOOTER_TEXT)

logger.info(f"{settings.APP_NAME} (v{settings.APP_VERSION}) - System Overview page (app.py) loaded successfully.")
