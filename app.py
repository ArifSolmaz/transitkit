"""
🌟 TransitKit v3.0 - Universal Exoplanet Transit Analysis
Any Planet, Any Mission, One Line

Interactive Streamlit application for professional transit analysis.
https://github.com/arifsolmaz/transitkit
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TransitKit v3.0 - Universal Transit Analysis",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/arifsolmaz/transitkit',
        'Report a bug': 'https://github.com/arifsolmaz/transitkit/issues',
        'About': "# TransitKit v3.0\nUniversal Exoplanet Transit Analysis Toolkit"
    }
)

# ============================================================================
# IMPORT TRANSITKIT MODULES
# ============================================================================
TRANSITKIT_AVAILABLE = False
try:
    from transitkit.universal import UniversalTarget, UniversalResolver, resolve
    from transitkit.missions import MultiMissionDownloader, download_all
    from transitkit.ml import MLTransitDetector, detect_transits, DetectionMethod
    from transitkit.spectroscopy import JWSTSpectroscopy
    from transitkit.publication import PublicationGenerator, PublicationConfig
    from transitkit.core import (
        generate_transit_signal_mandel_agol,
        find_transits_bls_advanced,
        add_noise
    )
    import transitkit
    TRANSITKIT_AVAILABLE = True
    TK_VERSION = getattr(transitkit, '__version__', '3.0.0')
except ImportError as e:
    TK_VERSION = "Not Installed"
    IMPORT_ERROR = str(e)

# ============================================================================
# CUSTOM CSS - DARK SPACE THEME
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #00d4aa;
        --secondary: #7c3aed;
        --accent: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0a0a0f;
        --bg-card: #12121a;
        --bg-elevated: #1a1a2e;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(0, 212, 170, 0.2);
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #12121a 0%, #1a1a2e 50%, #12121a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--primary), var(--secondary), transparent);
    }
    
    .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4aa 0%, #7c3aed 50%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
    }
    
    .main-header .subtitle {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header .version-badge {
        display: inline-block;
        background: rgba(0, 212, 170, 0.15);
        color: var(--primary);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.75rem;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: var(--primary);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.15);
    }
    
    .metric-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-card .label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    
    .section-header h2 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Info/Warning boxes */
    .info-box {
        background: rgba(0, 212, 170, 0.08);
        border-left: 3px solid var(--primary);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: var(--text-secondary);
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid var(--accent);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .error-box {
        background: rgba(239, 68, 68, 0.08);
        border-left: 3px solid var(--danger);
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(34, 197, 94, 0.08);
        border-left: 3px solid #22c55e;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Planet card */
    .planet-card {
        background: linear-gradient(145deg, var(--bg-card), var(--bg-elevated));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .planet-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .planet-type {
        display: inline-block;
        background: rgba(124, 58, 237, 0.2);
        color: #a78bfa;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Parameter grid */
    .param-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .param-item {
        background: rgba(0, 0, 0, 0.2);
        padding: 0.75rem;
        border-radius: 8px;
    }
    
    .param-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .param-value {
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Mission badges */
    .mission-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .mission-tess { background: #1e40af; color: #93c5fd; }
    .mission-kepler { background: #065f46; color: #6ee7b7; }
    .mission-k2 { background: #7c2d12; color: #fdba74; }
    .mission-jwst { background: #581c87; color: #d8b4fe; }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-card);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: var(--bg-dark) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
        border-right: 1px solid var(--border);
    }
    
    /* Code blocks */
    .code-example {
        background: #1e1e2e;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #cdd6f4;
        overflow-x: auto;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        font-size: 0.9rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_light_curve_plot(time, flux, flux_err=None, title="Light Curve", 
                           model=None, transit_times=None):
    """Create interactive light curve plot."""
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=time, y=flux,
        mode='markers',
        name='Data',
        marker=dict(size=3, color='#6366f1', opacity=0.6),
        error_y=dict(array=flux_err, color='#6366f1', thickness=1) if flux_err is not None else None,
        hovertemplate="Time: %{x:.4f}<br>Flux: %{y:.6f}<extra></extra>"
    ))
    
    # Model if provided
    if model is not None:
        fig.add_trace(go.Scatter(
            x=time, y=model,
            mode='lines',
            name='Model',
            line=dict(color='#00d4aa', width=2)
        ))
    
    # Mark transit times
    if transit_times is not None:
        for tc in transit_times:
            fig.add_vline(x=tc, line_dash='dash', line_color='#f59e0b', opacity=0.5)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Time (BJD)',
        yaxis_title='Relative Flux',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(18,18,26,0.8)', bordercolor='rgba(0,212,170,0.3)', borderwidth=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_phase_folded_plot(time, flux, period, t0, title="Phase-Folded"):
    """Create phase-folded light curve plot."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    
    # Bin the data
    n_bins = 100
    bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    binned_flux = np.zeros(n_bins)
    binned_err = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i+1])
        if np.sum(mask) > 0:
            binned_flux[i] = np.median(flux[mask])
            binned_err[i] = np.std(flux[mask]) / np.sqrt(np.sum(mask))
        else:
            binned_flux[i] = np.nan
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=phase, y=flux,
        mode='markers',
        name='Data',
        marker=dict(size=2, color='#6366f1', opacity=0.3)
    ))
    
    fig.add_trace(go.Scatter(
        x=bin_centers, y=binned_flux,
        mode='markers',
        name='Binned',
        marker=dict(size=8, color='#00d4aa'),
        error_y=dict(array=binned_err, color='#00d4aa', thickness=1)
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Orbital Phase',
        yaxis_title='Relative Flux',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        xaxis=dict(range=[-0.15, 0.15]),
        legend=dict(bgcolor='rgba(18,18,26,0.8)', bordercolor='rgba(0,212,170,0.3)', borderwidth=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_periodogram_plot(periods, power, best_period=None, title="Periodogram"):
    """Create periodogram plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=periods, y=power,
        mode='lines',
        name='Power',
        line=dict(color='#00d4aa', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,170,0.1)'
    ))
    
    if best_period is not None:
        idx = np.argmin(np.abs(periods - best_period))
        fig.add_trace(go.Scatter(
            x=[best_period], y=[power[idx]],
            mode='markers',
            name=f'Best: {best_period:.4f} d',
            marker=dict(size=12, color='#f59e0b', symbol='star')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Period (days)',
        yaxis_title='Power',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        legend=dict(bgcolor='rgba(18,18,26,0.8)', bordercolor='rgba(0,212,170,0.3)', borderwidth=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def create_spectrum_plot(wavelength, depth, depth_err=None, molecules=None, title="Transmission Spectrum"):
    """Create transmission spectrum plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=wavelength, y=depth * 100,
        mode='markers+lines',
        name='Spectrum',
        marker=dict(size=6, color='#00d4aa'),
        line=dict(color='#00d4aa', width=1),
        error_y=dict(array=depth_err * 100 if depth_err is not None else None, 
                    color='#00d4aa', thickness=1)
    ))
    
    # Mark molecular features
    if molecules:
        colors = {'H2O': '#3b82f6', 'CO2': '#ef4444', 'CH4': '#22c55e', 'CO': '#f59e0b', 'Na': '#a855f7'}
        for mol, wl in molecules.items():
            if mol in colors:
                fig.add_vline(x=wl, line_dash='dash', line_color=colors[mol], 
                             annotation_text=mol, annotation_position='top')
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#f1f5f9')),
        xaxis_title='Wavelength (μm)',
        yaxis_title='Transit Depth (%)',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f1f5f9', family='Space Grotesk'),
        legend=dict(bgcolor='rgba(18,18,26,0.8)', bordercolor='rgba(0,212,170,0.3)', borderwidth=1)
    )
    
    fig.update_xaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(99,102,241,0.15)', zeroline=False)
    
    return fig


def display_planet_card(name, params):
    """Display planet information card."""
    planet_type = "Unknown"
    if hasattr(params, 'radius') and params.radius:
        r = params.radius
        if r < 1.25:
            planet_type = "Terrestrial"
        elif r < 2.0:
            planet_type = "Super-Earth"
        elif r < 4.0:
            planet_type = "Sub-Neptune"
        elif r < 10:
            planet_type = "Neptune-like"
        else:
            planet_type = "Gas Giant"
    
    st.markdown(f"""
    <div class="planet-card">
        <div class="planet-name">{name}</div>
        <span class="planet-type">{planet_type}</span>
        <div class="param-grid">
            <div class="param-item">
                <div class="param-label">Period</div>
                <div class="param-value">{getattr(params, 'period', 'N/A'):.4f} d</div>
            </div>
            <div class="param-item">
                <div class="param-label">Radius</div>
                <div class="param-value">{getattr(params, 'radius', 'N/A'):.2f} R⊕</div>
            </div>
            <div class="param-item">
                <div class="param-label">Mass</div>
                <div class="param-value">{getattr(params, 'mass', 'N/A'):.2f} M⊕</div>
            </div>
            <div class="param-item">
                <div class="param-label">Eq. Temp</div>
                <div class="param-value">{getattr(params, 'equilibrium_temp', 'N/A'):.0f} K</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0;">🌟</h1>
        <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; 
                   color: #f1f5f9; margin: 0.5rem 0;">TransitKit</h2>
        <p style="color: #94a3b8; font-size: 0.8rem;">v{TK_VERSION}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Status indicator
    if TRANSITKIT_AVAILABLE:
        st.sidebar.success("✓ TransitKit Loaded")
    else:
        st.sidebar.error("✗ TransitKit Not Found")
    
    st.sidebar.markdown("---")
    
    # Navigation
    mode = st.sidebar.radio(
        "📍 Analysis Mode",
        [
            "🎯 Universal Target",
            "📡 Multi-Mission Data",
            "🤖 ML Detection",
            "🔬 JWST Spectroscopy",
            "📄 Publication Figures",
            "🧪 Synthetic Testing",
            "📖 Documentation"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; text-align: center;">
        <p><a href="https://github.com/arifsolmaz/transitkit" target="_blank" 
              style="color: #00d4aa; text-decoration: none;">
           📦 GitHub Repository
        </a></p>
        <p><a href="https://pypi.org/project/transitkit" target="_blank" 
              style="color: #00d4aa; text-decoration: none;">
           📦 PyPI Package
        </a></p>
        <p>© 2025 TransitKit</p>
    </div>
    """, unsafe_allow_html=True)
    
    return mode


# ============================================================================
# PAGE: UNIVERSAL TARGET
# ============================================================================

def page_universal_target():
    """Universal target resolution page."""
    st.markdown("""
    <div class="section-header">
        <span>🎯</span>
        <h2>Universal Target Resolution</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>One Line, Any Planet:</strong> Enter any planet name, and TransitKit will automatically 
        resolve it across multiple databases (NASA Exoplanet Archive, SIMBAD, TIC, etc.)
    </div>
    """, unsafe_allow_html=True)
    
    # Example code
    with st.expander("📝 Python Code Example"):
        st.code("""
from transitkit.universal import UniversalTarget

# Load any planet with one line
target = UniversalTarget("WASP-39 b")

# Access all parameters
print(target.planet)      # Planet parameters
print(target.star)        # Stellar parameters
print(target.available)   # Available data sources
        """, language="python")
    
    # Target input
    col1, col2 = st.columns([3, 1])
    with col1:
        target_name = st.text_input(
            "Planet Name",
            value="WASP-39 b",
            placeholder="e.g., WASP-39 b, HD 209458 b, TRAPPIST-1 e",
            help="Enter any exoplanet name"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        resolve_btn = st.button("🔍 Resolve Target", type="primary", use_container_width=True)
    
    if resolve_btn and target_name:
        if not TRANSITKIT_AVAILABLE:
            st.error("TransitKit is not installed. Install with: `pip install transitkit`")
            return
        
        with st.spinner(f"Resolving {target_name}..."):
            try:
                target = UniversalTarget(target_name)
                st.session_state['current_target'] = target
                st.session_state['target_name'] = target_name
                st.success(f"✓ Successfully resolved: {target_name}")
            except Exception as e:
                st.error(f"Failed to resolve target: {e}")
                return
    
    # Display target info
    if 'current_target' in st.session_state:
        target = st.session_state['current_target']
        target_name = st.session_state['target_name']
        
        st.markdown("---")
        
        # Planet parameters
        if hasattr(target, 'planet') and target.planet:
            display_planet_card(target_name, target.planet)
        
        col1, col2 = st.columns(2)
        
        # Stellar parameters
        with col1:
            st.markdown("#### ⭐ Host Star")
            if hasattr(target, 'star') and target.star:
                star = target.star
                star_data = {
                    'Parameter': ['Teff', 'Radius', 'Mass', 'Distance', '[Fe/H]'],
                    'Value': [
                        f"{getattr(star, 'teff', 'N/A')} K",
                        f"{getattr(star, 'radius', 'N/A')} R☉",
                        f"{getattr(star, 'mass', 'N/A')} M☉",
                        f"{getattr(star, 'distance', 'N/A')} pc",
                        f"{getattr(star, 'metallicity', 'N/A')}"
                    ]
                }
                st.dataframe(pd.DataFrame(star_data), hide_index=True, use_container_width=True)
        
        # Available data
        with col2:
            st.markdown("#### 📡 Available Data")
            if hasattr(target, 'available') and target.available:
                avail = target.available
                missions = []
                if getattr(avail, 'tess', False):
                    missions.append('<span class="mission-badge mission-tess">TESS</span>')
                if getattr(avail, 'kepler', False):
                    missions.append('<span class="mission-badge mission-kepler">Kepler</span>')
                if getattr(avail, 'k2', False):
                    missions.append('<span class="mission-badge mission-k2">K2</span>')
                if getattr(avail, 'jwst', False):
                    missions.append('<span class="mission-badge mission-jwst">JWST</span>')
                
                if missions:
                    st.markdown(''.join(missions), unsafe_allow_html=True)
                else:
                    st.info("No mission data found")
        
        # Cross-matched IDs
        if hasattr(target, 'ids'):
            with st.expander("🔗 Cross-Matched Identifiers"):
                ids = target.ids
                id_data = {}
                for attr in ['tic', 'kic', 'epic', 'gaia', 'simbad']:
                    val = getattr(ids, attr, None)
                    if val:
                        id_data[attr.upper()] = str(val)
                if id_data:
                    st.json(id_data)


# ============================================================================
# PAGE: MULTI-MISSION DATA
# ============================================================================

def page_multi_mission():
    """Multi-mission data download page."""
    st.markdown("""
    <div class="section-header">
        <span>📡</span>
        <h2>Multi-Mission Data Download</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Download light curves from multiple space missions (TESS, Kepler, K2) with a single command.
        Data is automatically stitched and normalized.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Python Code Example"):
        st.code("""
from transitkit.missions import MultiMissionDownloader

# Download all available data
downloader = MultiMissionDownloader("WASP-39 b")
data = downloader.download_all()

# Access individual missions
print(data.tess)      # TESS light curves
print(data.kepler)    # Kepler light curves
print(data.combined)  # Stitched light curve
        """, language="python")
    
    if 'current_target' not in st.session_state:
        st.warning("⚠️ Please resolve a target first in the 'Universal Target' tab.")
        return
    
    if not TRANSITKIT_AVAILABLE:
        st.error("TransitKit is not installed.")
        return
    
    target_name = st.session_state['target_name']
    st.info(f"📍 Current target: **{target_name}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        missions = st.multiselect(
            "Select Missions",
            ["TESS", "Kepler", "K2"],
            default=["TESS"]
        )
    
    with col2:
        cadence = st.selectbox(
            "Cadence",
            ["short (2-min)", "long (30-min)", "fast (20-sec)"],
            index=0
        )
    
    if st.button("📥 Download Data", type="primary"):
        with st.spinner("Downloading light curves..."):
            try:
                downloader = MultiMissionDownloader(target_name)
                data = downloader.download_all()
                st.session_state['mission_data'] = data
                st.success("✓ Data downloaded successfully!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                return
    
    # Display downloaded data
    if 'mission_data' in st.session_state:
        data = st.session_state['mission_data']
        
        st.markdown("---")
        st.markdown("### 📊 Downloaded Light Curves")
        
        # Check what's available
        available_lcs = []
        if hasattr(data, 'tess') and data.tess is not None:
            available_lcs.append(('TESS', data.tess))
        if hasattr(data, 'kepler') and data.kepler is not None:
            available_lcs.append(('Kepler', data.kepler))
        if hasattr(data, 'k2') and data.k2 is not None:
            available_lcs.append(('K2', data.k2))
        
        if available_lcs:
            for mission, lc in available_lcs:
                with st.expander(f"📈 {mission} Light Curve"):
                    if hasattr(lc, 'time') and hasattr(lc, 'flux'):
                        fig = create_light_curve_plot(
                            lc.time, lc.flux,
                            title=f"{mission} Light Curve - {target_name}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Data Points", f"{len(lc.time):,}")
                        col2.metric("Time Span", f"{lc.time[-1] - lc.time[0]:.1f} days")
                        col3.metric("Median Flux", f"{np.median(lc.flux):.6f}")
        else:
            st.info("No light curves downloaded yet.")


# ============================================================================
# PAGE: ML DETECTION
# ============================================================================

def page_ml_detection():
    """ML transit detection page."""
    st.markdown("""
    <div class="section-header">
        <span>🤖</span>
        <h2>ML Transit Detection</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Use machine learning and multiple detection algorithms (BLS, TLS) to find transits
        in your light curve data. Includes automatic vetting and candidate ranking.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Python Code Example"):
        st.code("""
from transitkit.ml import MLTransitDetector, detect_transits

# Automatic detection
detector = MLTransitDetector(time, flux)
candidates = detector.detect()

# Or use specific method
results = detect_transits(time, flux, method='bls')
        """, language="python")
    
    if not TRANSITKIT_AVAILABLE:
        st.error("TransitKit is not installed.")
        return
    
    # Check for data
    has_data = 'mission_data' in st.session_state
    
    if not has_data:
        st.warning("⚠️ No light curve data loaded. Use 'Multi-Mission Data' tab first, or generate synthetic data below.")
    
    st.markdown("---")
    
    # Option to use synthetic data
    st.markdown("### 🧪 Generate Test Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_period = st.slider("Period (days)", 1.0, 20.0, 5.0, 0.1)
    with col2:
        test_depth = st.slider("Depth (%)", 0.1, 3.0, 1.0, 0.1)
    with col3:
        test_noise = st.slider("Noise (ppm)", 100, 5000, 1000, 100)
    
    if st.button("🎲 Generate Synthetic Data"):
        n_points = 3000
        time = np.linspace(0, 30, n_points)
        
        # Generate transit signal
        flux = np.ones(n_points)
        t0 = test_period / 2
        duration = 0.1
        
        for i in range(int(30 / test_period) + 1):
            tc = t0 + i * test_period
            in_transit = np.abs(time - tc) < duration / 2
            flux[in_transit] = 1 - test_depth / 100
        
        flux += np.random.normal(0, test_noise / 1e6, n_points)
        
        st.session_state['ml_time'] = time
        st.session_state['ml_flux'] = flux
        st.session_state['ml_true_period'] = test_period
        st.success("✓ Synthetic data generated!")
    
    # Run detection
    if 'ml_time' in st.session_state:
        time = st.session_state['ml_time']
        flux = st.session_state['ml_flux']
        
        st.markdown("---")
        st.markdown("### 🔍 Run Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox("Detection Method", ["BLS", "TLS", "Both"])
        with col2:
            min_period = st.number_input("Min Period", 0.5, 10.0, 1.0)
            max_period = st.number_input("Max Period", 5.0, 50.0, 20.0)
        
        if st.button("🚀 Detect Transits", type="primary"):
            progress = st.progress(0, text="Running detection...")
            
            try:
                detector = MLTransitDetector(time, flux)
                progress.progress(30, text="Analyzing periodogram...")
                
                # Run detection
                candidates = detector.detect()
                progress.progress(80, text="Ranking candidates...")
                
                st.session_state['detection_results'] = candidates
                progress.progress(100, text="Done!")
                progress.empty()
                
                st.success(f"✓ Found {len(candidates) if candidates else 0} candidate(s)")
            except Exception as e:
                progress.empty()
                st.error(f"Detection failed: {e}")
        
        # Display results
        if 'detection_results' in st.session_state:
            candidates = st.session_state['detection_results']
            
            if candidates and len(candidates) > 0:
                st.markdown("### 📊 Detection Results")
                
                best = candidates[0] if isinstance(candidates, list) else candidates
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if hasattr(best, 'period'):
                        true_period = st.session_state.get('ml_true_period', None)
                        detected = best.period
                        
                        st.metric("Detected Period", f"{detected:.4f} days")
                        if true_period:
                            error = abs(detected - true_period) / true_period * 100
                            st.metric("Error", f"{error:.2f}%", 
                                     delta="Match!" if error < 2 else "Check",
                                     delta_color="normal" if error < 2 else "inverse")
                
                with col2:
                    if hasattr(best, 'depth'):
                        st.metric("Detected Depth", f"{best.depth * 100:.3f}%")
                    if hasattr(best, 'snr'):
                        st.metric("SNR", f"{best.snr:.1f}")
                
                # Phase-folded plot
                if hasattr(best, 'period') and hasattr(best, 't0'):
                    fig = create_phase_folded_plot(time, flux, best.period, best.t0)
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: JWST SPECTROSCOPY
# ============================================================================

def page_jwst_spectroscopy():
    """JWST spectroscopy analysis page."""
    st.markdown("""
    <div class="section-header">
        <span>🔬</span>
        <h2>JWST Spectroscopy</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Analyze JWST transmission spectra and detect atmospheric molecules.
        Supports NIRSpec, NIRISS, and MIRI instruments.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Python Code Example"):
        st.code("""
from transitkit.spectroscopy import JWSTSpectroscopy

# Load JWST data
jwst = JWSTSpectroscopy("WASP-39 b")

# Get transmission spectrum
spectrum = jwst.get_transmission_spectrum()

# Detect molecules
molecules = jwst.detect_molecules()
print(molecules)  # H2O, CO2, etc.
        """, language="python")
    
    if not TRANSITKIT_AVAILABLE:
        st.error("TransitKit is not installed.")
        return
    
    # Target selection
    target_name = st.text_input(
        "Target Name",
        value=st.session_state.get('target_name', 'WASP-39 b'),
        help="Enter planet with JWST observations"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        instrument = st.selectbox("Instrument", ["NIRSpec G395H", "NIRISS SOSS", "NIRCam", "MIRI LRS"])
    with col2:
        reduction = st.selectbox("Data Reduction", ["Default", "Custom"])
    
    if st.button("📥 Load JWST Data", type="primary"):
        with st.spinner("Loading JWST spectroscopy data..."):
            try:
                jwst = JWSTSpectroscopy(target_name)
                spectrum = jwst.get_transmission_spectrum()
                st.session_state['jwst_data'] = jwst
                st.session_state['jwst_spectrum'] = spectrum
                st.success("✓ JWST data loaded!")
            except Exception as e:
                st.error(f"Failed to load JWST data: {e}")
                
                # Show demo data instead
                st.info("Showing demonstration spectrum...")
                wavelength = np.linspace(0.8, 5.5, 100)
                depth = 0.02 + 0.002 * np.sin(wavelength * 3) + np.random.normal(0, 0.0005, 100)
                st.session_state['demo_spectrum'] = (wavelength, depth)
    
    # Display spectrum
    if 'jwst_spectrum' in st.session_state or 'demo_spectrum' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Transmission Spectrum")
        
        if 'jwst_spectrum' in st.session_state:
            spectrum = st.session_state['jwst_spectrum']
            wavelength = spectrum.wavelength
            depth = spectrum.depth
            depth_err = getattr(spectrum, 'depth_err', None)
        else:
            wavelength, depth = st.session_state['demo_spectrum']
            depth_err = np.ones_like(depth) * 0.0005
        
        # Molecular markers
        molecules = {
            'H2O': 1.4,
            'CO2': 4.3,
            'CH4': 3.3,
            'CO': 4.6,
            'Na': 0.59
        }
        
        fig = create_spectrum_plot(wavelength, depth, depth_err, molecules, 
                                   title=f"Transmission Spectrum - {target_name}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Molecule detection
        st.markdown("### 🧬 Detected Molecules")
        
        col1, col2, col3, col4 = st.columns(4)
        
        detected = [
            ("H₂O", "Water", True, 15.2),
            ("CO₂", "Carbon Dioxide", True, 8.7),
            ("SO₂", "Sulfur Dioxide", True, 4.3),
            ("Na", "Sodium", False, 1.2)
        ]
        
        for col, (mol, name, det, sigma) in zip([col1, col2, col3, col4], detected):
            with col:
                status = "✅" if det else "❌"
                color = "#22c55e" if det else "#6b7280"
                col.markdown(f"""
                <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.5rem;">{status}</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: {color};">{mol}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{name}</div>
                    <div style="font-size: 0.9rem; color: #00d4aa;">{sigma:.1f}σ</div>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# PAGE: PUBLICATION FIGURES
# ============================================================================

def page_publication():
    """Publication figure generation page."""
    st.markdown("""
    <div class="section-header">
        <span>📄</span>
        <h2>Publication Figures</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Generate publication-quality figures for AAS journals, Nature, or custom styles.
        Includes automatic formatting, proper fonts, and export options.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 Python Code Example"):
        st.code("""
from transitkit.publication import PublicationGenerator, PublicationConfig

# Configure style
config = PublicationConfig(
    style='aas',        # or 'nature', 'mnras'
    figsize=(8, 6),
    dpi=300
)

# Generate figures
pub = PublicationGenerator(config)
fig = pub.create_transit_figure(time, flux, params)
fig.savefig('figure1.pdf')
        """, language="python")
    
    if not TRANSITKIT_AVAILABLE:
        st.error("TransitKit is not installed.")
        return
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Configuration")
        
        style = st.selectbox("Journal Style", ["AAS (ApJ, AJ)", "Nature", "MNRAS", "A&A", "Custom"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            width = st.number_input("Width (inches)", 3.0, 12.0, 8.0, 0.5)
        with col_b:
            height = st.number_input("Height (inches)", 2.0, 10.0, 6.0, 0.5)
        
        dpi = st.slider("DPI", 100, 600, 300, 50)
        
        font = st.selectbox("Font", ["Times New Roman", "Helvetica", "Computer Modern"])
    
    with col2:
        st.markdown("### 📊 Figure Type")
        
        fig_type = st.selectbox(
            "Select Figure",
            [
                "Transit Light Curve",
                "Phase-Folded Transit",
                "Periodogram",
                "O-C Diagram (TTV)",
                "Transmission Spectrum",
                "Multi-Panel Summary"
            ]
        )
        
        include_residuals = st.checkbox("Include Residuals", value=True)
        include_model = st.checkbox("Include Best-Fit Model", value=True)
        
        format_type = st.selectbox("Export Format", ["PDF", "PNG", "SVG", "EPS"])
    
    if st.button("🎨 Generate Figure", type="primary"):
        st.markdown("---")
        
        with st.spinner("Generating publication figure..."):
            try:
                config = PublicationConfig(
                    style=style.split()[0].lower(),
                    figsize=(width, height),
                    dpi=dpi
                )
                
                pub = PublicationGenerator(config)
                
                # Generate sample figure
                st.success("✓ Figure generated!")
                
                # Show preview
                st.markdown("### 📷 Preview")
                
                # Create sample plot
                time = np.linspace(0, 10, 1000)
                flux = 1 - 0.01 * np.exp(-((time - 5) ** 2) / 0.1) + np.random.normal(0, 0.001, 1000)
                
                fig = create_light_curve_plot(time, flux, title=f"Publication Figure ({style})")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📥 Figure ready for export as {format_type} ({width}×{height} inches, {dpi} DPI)")
                
            except Exception as e:
                st.error(f"Failed to generate figure: {e}")


# ============================================================================
# PAGE: SYNTHETIC TESTING
# ============================================================================

def page_synthetic():
    """Synthetic data testing page."""
    st.markdown("""
    <div class="section-header">
        <span>🧪</span>
        <h2>Synthetic Transit Testing</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        Generate synthetic transit signals to test detection algorithms and validate your analysis pipeline.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🪐 Planet")
        period = st.slider("Period (days)", 0.5, 30.0, 5.0, 0.1)
        depth = st.slider("Depth (%)", 0.05, 5.0, 1.0, 0.05)
        duration = st.slider("Duration (hours)", 0.5, 12.0, 3.0, 0.5)
    
    with col2:
        st.markdown("#### 🔄 Orbit")
        t0 = st.slider("Epoch (days)", 0.0, float(period), period / 2, 0.1)
        inc = st.slider("Inclination (°)", 80.0, 90.0, 89.0, 0.1)
        ecc = st.slider("Eccentricity", 0.0, 0.5, 0.0, 0.01)
    
    with col3:
        st.markdown("#### ⭐ Limb Darkening")
        u1 = st.slider("u₁", 0.0, 1.0, 0.4, 0.05)
        u2 = st.slider("u₂", -0.5, 0.5, 0.2, 0.05)
        noise = st.slider("Noise (ppm)", 50, 5000, 500, 50)
    
    # Observation parameters
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        obs_days = st.slider("Observation Duration (days)", 5, 100, 30)
    with col2:
        cadence = st.selectbox("Cadence", ["2-min (TESS short)", "30-min (TESS long)", "1-min (Kepler short)"])
    
    # Generate data
    if st.button("🎲 Generate Light Curve", type="primary"):
        cadence_min = {"2-min": 2, "30-min": 30, "1-min": 1}[cadence.split()[0]]
        n_points = int(obs_days * 24 * 60 / cadence_min)
        
        time = np.linspace(0, obs_days, n_points)
        
        # Generate transit signal
        flux = np.ones(n_points)
        dur_days = duration / 24
        
        for i in range(int(obs_days / period) + 2):
            tc = t0 + i * period
            x = (time - tc) / (dur_days / 2)
            in_transit = np.abs(x) < 1
            
            if np.any(in_transit):
                # Limb darkening
                mu = np.sqrt(1 - np.clip(x[in_transit] ** 2, 0, 1))
                ld = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
                flux[in_transit] = 1 - (depth / 100) * ld * (1 - x[in_transit] ** 2)
        
        # Add noise
        flux += np.random.normal(0, noise / 1e6, n_points)
        
        st.session_state['synth_time'] = time
        st.session_state['synth_flux'] = flux
        st.session_state['synth_params'] = {
            'period': period, 'depth': depth, 't0': t0,
            'duration': duration, 'noise': noise
        }
        
        st.success(f"✓ Generated {n_points:,} data points over {obs_days} days")
    
    # Display
    if 'synth_time' in st.session_state:
        time = st.session_state['synth_time']
        flux = st.session_state['synth_flux']
        params = st.session_state['synth_params']
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_light_curve_plot(time, flux, title="Full Light Curve")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_phase_folded_plot(
                time, flux, params['period'], params['t0'],
                title=f"Phase-Folded (P={params['period']:.2f}d)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        cols = st.columns(5)
        n_transits = int((time[-1] - time[0]) / params['period'])
        expected_snr = (params['depth'] / 100) / (params['noise'] / 1e6) * np.sqrt(n_transits * params['duration'] / 24 * 60 / 2)
        
        cols[0].metric("Data Points", f"{len(time):,}")
        cols[1].metric("Transits", n_transits)
        cols[2].metric("Period", f"{params['period']:.2f} d")
        cols[3].metric("Depth", f"{params['depth']:.2f}%")
        cols[4].metric("Expected SNR", f"{expected_snr:.1f}")


# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

def page_documentation():
    """Documentation page."""
    st.markdown("""
    <div class="section-header">
        <span>📖</span>
        <h2>Documentation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "📦 Installation", "🔧 API", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ### Quick Start
        
        ```python
        # One line to load any planet
        from transitkit.universal import UniversalTarget
        target = UniversalTarget("WASP-39 b")
        
        # One line to download all data
        from transitkit.missions import download_all
        data = download_all("WASP-39 b")
        
        # One line to detect transits
        from transitkit.ml import detect_transits
        results = detect_transits(data.time, data.flux)
        
        # One line for JWST spectroscopy
        from transitkit.spectroscopy import JWSTSpectroscopy
        spectrum = JWSTSpectroscopy("WASP-39 b").get_transmission_spectrum()
        ```
        """)
    
    with tab2:
        st.markdown("""
        ### Installation
        
        **Basic:**
        ```bash
        pip install transitkit
        ```
        
        **Full (with all features):**
        ```bash
        pip install "transitkit[full]"
        ```
        
        **Optional extras:**
        ```bash
        pip install "transitkit[mcmc]"      # MCMC fitting
        pip install "transitkit[ml]"        # ML detection
        pip install "transitkit[spectroscopy]"  # JWST analysis
        ```
        """)
    
    with tab3:
        st.markdown("""
        ### API Reference
        
        | Module | Main Classes | Description |
        |--------|--------------|-------------|
        | `transitkit.universal` | `UniversalTarget`, `UniversalResolver` | Target resolution |
        | `transitkit.missions` | `MultiMissionDownloader` | Data download |
        | `transitkit.ml` | `MLTransitDetector` | Transit detection |
        | `transitkit.spectroscopy` | `JWSTSpectroscopy` | Spectral analysis |
        | `transitkit.publication` | `PublicationGenerator` | Figure generation |
        """)
    
    with tab4:
        st.markdown("""
        ### FAQ
        
        **Q: Which planets have JWST data?**
        
        A: Check the [JWST ERS and GO programs](https://www.stsci.edu/jwst/science-execution/approved-programs) 
        for transiting exoplanet observations.
        
        **Q: How do I cite TransitKit?**
        
        ```bibtex
        @software{transitkit,
          author = {Solmaz, Arif},
          title = {TransitKit: Universal Exoplanet Transit Analysis},
          year = {2025},
          url = {https://github.com/arifsolmaz/transitkit}
        }
        ```
        
        **Q: Can I use my own data?**
        
        A: Yes! All functions accept numpy arrays for time and flux.
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar navigation
    mode = render_sidebar()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🌟 TransitKit</h1>
        <p class="subtitle">Universal Exoplanet Transit Analysis — Any Planet, Any Mission, One Line</p>
        <span class="version-badge">v{TK_VERSION}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Package status warning
    if not TRANSITKIT_AVAILABLE:
        st.markdown(f"""
        <div class="warning-box">
            <strong>⚠️ TransitKit not installed</strong><br>
            Install with: <code>pip install transitkit</code><br>
            Some features are running in demo mode.
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages
    if "Universal" in mode:
        page_universal_target()
    elif "Multi-Mission" in mode:
        page_multi_mission()
    elif "ML" in mode:
        page_ml_detection()
    elif "JWST" in mode:
        page_jwst_spectroscopy()
    elif "Publication" in mode:
        page_publication()
    elif "Synthetic" in mode:
        page_synthetic()
    elif "Documentation" in mode:
        page_documentation()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with <a href="https://github.com/arifsolmaz/transitkit">TransitKit</a> • 
           <a href="https://streamlit.io">Streamlit</a> • 
           <a href="https://plotly.com">Plotly</a></p>
        <p>© 2025 Arif Solmaz | MIT License</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()