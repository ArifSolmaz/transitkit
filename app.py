"""
TransitKit Interactive Demo App
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from transitkit.core import (
    generate_transit_signal_mandel_agol,
    find_transits_bls_advanced,
    find_transits_multiple_methods,
    add_noise,
)
from transitkit.analysis import measure_transit_timing_variations

st.set_page_config(page_title="TransitKit Demo", page_icon="🪐", layout="wide")

st.title("🪐 TransitKit")
st.markdown("**Professional Exoplanet Transit Detection Toolkit**")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a demo:", ["Home", "Synthetic Transit", "Multi-Method", "TTV Analysis", "Batch Analysis"])

# =============================================================================
# HOME PAGE
# =============================================================================
if page == "Home":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### What is TransitKit?")
        st.write("A professional Python toolkit for detecting and analyzing exoplanet transits.")
    with col2:
        st.markdown("### Features")
        st.write("BLS, GLS, PDM detection • MCMC fitting • TTV analysis • Publication plots")
    with col3:
        st.markdown("### Get Started")
        st.write("Use the sidebar to explore demos. Start with **Synthetic Transit**!")
    
    st.markdown("---")
    st.markdown("### Quick Demo")
    
    if st.button("Run Quick Detection Demo", type="primary"):
        st.write("Generating data...")
        time = np.linspace(0, 20, 1500)
        flux = generate_transit_signal_mandel_agol(time, period=4.0, t0=2.0, depth=0.01)
        flux = add_noise(flux, 0.001)
        
        st.write("Running BLS detection...")
        result = find_transits_bls_advanced(time, flux)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Period", f"{result['period']:.4f} d")
        col2.metric("Depth", f"{result['depth']*1e6:.0f} ppm")
        col3.metric("SNR", f"{result['snr']:.1f}")
        col4.metric("Rp/Rs", f"{np.sqrt(result['depth']):.4f}")
        
        phase = ((time - result['t0']) / result['period']) % 1
        phase[phase > 0.5] -= 1
        
        # Zoom on transit region only
        transit_mask = np.abs(phase) < 0.1
        phase_zoom = phase[transit_mask]
        flux_zoom = flux[transit_mask]
        sort_idx = np.argsort(phase_zoom)
        
        # Use matplotlib for better control
        st.subheader("Phase-Folded Light Curve (Zoomed on Transit)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(phase_zoom[sort_idx], flux_zoom[sort_idx], 'b.', ms=3, alpha=0.5)
        ax.set_xlabel('Orbital Phase')
        ax.set_ylabel('Normalized Flux')
        ax.set_xlim(-0.08, 0.08)
        ax.set_ylim(flux_zoom.min() - 0.002, flux_zoom.max() + 0.002)
        ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)
        
        st.success("Transit detected successfully!")

# =============================================================================
# SYNTHETIC TRANSIT PAGE
# =============================================================================
elif page == "Synthetic Transit":
    st.header("Synthetic Transit Generator")
    st.write("Create and detect synthetic exoplanet transits with custom parameters.")
    
    st.sidebar.markdown("### Transit Parameters")
    period = st.sidebar.slider("Orbital Period (days)", 1.0, 20.0, 5.0, 0.1)
    depth_pct = st.sidebar.slider("Transit Depth (%)", 0.1, 3.0, 1.0, 0.1)
    depth = depth_pct / 100
    duration_hr = st.sidebar.slider("Duration (hours)", 1.0, 8.0, 3.0, 0.5)
    duration = duration_hr / 24
    
    st.sidebar.markdown("### Data Parameters")
    baseline = st.sidebar.slider("Baseline (days)", 10, 100, 30, 5)
    n_points = st.sidebar.slider("Data Points", 500, 5000, 2000, 100)
    noise_ppm = st.sidebar.slider("Noise (ppm)", 100, 5000, 1000, 100)
    
    if st.button("Generate & Detect", type="primary"):
        with st.spinner("Processing..."):
            time = np.linspace(0, baseline, n_points)
            t0 = period / 2
            flux = generate_transit_signal_mandel_agol(time, period=period, t0=t0, depth=depth, duration=duration)
            flux_noisy = add_noise(flux, noise_level=noise_ppm/1e6)
            result = find_transits_bls_advanced(time, flux_noisy, min_period=0.5, max_period=min(baseline/2, 50))
            st.session_state['synth_data'] = {'time': time, 'flux': flux_noisy, 'result': result, 'true': {'period': period, 'depth': depth, 't0': t0}}
    
    if 'synth_data' in st.session_state:
        data = st.session_state['synth_data']
        result = data['result']
        true = data['true']
        time = data['time']
        flux = data['flux']
        
        st.markdown("### Detection Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Period (days)", f"{result['period']:.4f}", f"True: {true['period']:.2f}")
        col2.metric("Depth (ppm)", f"{result['depth']*1e6:.0f}", f"True: {true['depth']*1e6:.0f}")
        col3.metric("SNR", f"{result['snr']:.1f}")
        period_err = abs(result['period'] - true['period']) / true['period'] * 100
        col4.metric("Error", f"{period_err:.2f}%")
        
        tab1, tab2, tab3 = st.tabs(["Light Curve", "Periodogram", "Phase-Folded"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(time, flux, 'k.', ms=1, alpha=0.5)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Normalized Flux')
            ax.set_title('Synthetic Light Curve')
            st.pyplot(fig)
            plt.close(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(result['all_periods'], result['all_powers'], 'b-', lw=0.5)
            ax.axvline(result['period'], color='r', ls='--', lw=2, label=f"Detected: {result['period']:.3f}d")
            ax.axvline(true['period'], color='g', ls=':', lw=2, label=f"True: {true['period']:.3f}d")
            ax.set_xlabel('Period (days)')
            ax.set_ylabel('BLS Power')
            ax.legend()
            ax.set_xscale('log')
            st.pyplot(fig)
            plt.close(fig)
        
        with tab3:
            phase = ((time - result['t0']) / result['period']) % 1
            phase[phase > 0.5] -= 1
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(phase, flux, 'k.', ms=2, alpha=0.3, label='Data')
            n_bins = 50
            bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            binned = [np.nanmedian(flux[(phase >= bin_edges[i]) & (phase < bin_edges[i+1])]) for i in range(n_bins)]
            ax.plot(bin_centers, binned, 'ro-', ms=4, lw=1, label='Binned')
            ax.set_xlabel('Orbital Phase')
            ax.set_ylabel('Normalized Flux')
            ax.set_xlim(-0.15, 0.15)
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

# =============================================================================
# MULTI-METHOD PAGE
# =============================================================================
elif page == "Multi-Method":
    st.header("Multi-Method Detection")
    st.write("Compare BLS, GLS, and PDM detection methods.")
    
    st.sidebar.markdown("### Signal Parameters")
    period = st.sidebar.slider("True Period (days)", 1.0, 15.0, 4.5, 0.5)
    depth = st.sidebar.slider("Depth (%)", 0.3, 2.0, 1.0, 0.1) / 100
    noise = st.sidebar.slider("Noise (ppm)", 500, 3000, 1000, 100)
    
    if st.button("Run Multi-Method Analysis", type="primary"):
        with st.spinner("Running BLS, GLS, and PDM..."):
            time = np.linspace(0, 40, 2500)
            flux = generate_transit_signal_mandel_agol(time, period=period, t0=period/2, depth=depth)
            flux = add_noise(flux, noise/1e6)
            results = find_transits_multiple_methods(time, flux, min_period=1.0, max_period=20.0, methods=['bls', 'gls', 'pdm'])
            st.session_state['multi_data'] = {'results': results, 'true_period': period}
    
    if 'multi_data' in st.session_state:
        data = st.session_state['multi_data']
        results = data['results']
        true_p = data['true_period']
        
        st.markdown("### Method Comparison")
        df = pd.DataFrame({
            'Method': ['BLS', 'GLS', 'PDM', 'Consensus'],
            'Period (d)': [f"{results['bls']['period']:.4f}", f"{results['gls']['period']:.4f}", f"{results['pdm']['period']:.4f}", f"{results['consensus']['period']:.4f}"],
            'Error (%)': [f"{abs(results['bls']['period']-true_p)/true_p*100:.2f}", f"{abs(results['gls']['period']-true_p)/true_p*100:.2f}", f"{abs(results['pdm']['period']-true_p)/true_p*100:.2f}", f"{abs(results['consensus']['period']-true_p)/true_p*100:.2f}"]
        })
        st.table(df)
        st.info(f"True Period: {true_p:.4f} days")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(results['bls']['all_periods'], results['bls']['all_powers'], 'b-', lw=0.5)
        axes[0].axvline(true_p, color='g', ls='--', alpha=0.7, label='True')
        axes[0].axvline(results['bls']['period'], color='r', ls=':', label='Detected')
        axes[0].set_ylabel('BLS Power')
        axes[0].set_title('Box Least Squares (BLS)')
        axes[0].legend()
        
        gls_periods = 1 / results['gls']['frequencies']
        axes[1].plot(gls_periods, results['gls']['powers'], 'orange', lw=0.5)
        axes[1].axvline(true_p, color='g', ls='--', alpha=0.7)
        axes[1].axvline(results['gls']['period'], color='r', ls=':')
        axes[1].set_ylabel('GLS Power')
        axes[1].set_title('Generalized Lomb-Scargle (GLS)')
        axes[1].set_xlim(1, 20)
        
        axes[2].plot(results['pdm']['periods'], results['pdm']['thetas'], 'purple', lw=0.5)
        axes[2].axvline(true_p, color='g', ls='--', alpha=0.7)
        axes[2].axvline(results['pdm']['period'], color='r', ls=':')
        axes[2].set_ylabel('PDM Theta')
        axes[2].set_xlabel('Period (days)')
        axes[2].set_title('Phase Dispersion Minimization (PDM)')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================================================================
# TTV ANALYSIS PAGE
# =============================================================================
elif page == "TTV Analysis":
    st.header("Transit Timing Variations")
    st.write("Detect timing deviations that could indicate additional planets.")
    
    st.sidebar.markdown("### TTV Parameters")
    ttv_amp = st.sidebar.slider("TTV Amplitude (minutes)", 0, 60, 20, 5)
    ttv_period_epochs = st.sidebar.slider("TTV Super-Period (epochs)", 5, 30, 10, 1)
    
    if st.button("Generate TTV Data & Analyze", type="primary"):
        with st.spinner("Generating light curve with TTVs..."):
            PERIOD = 5.0
            DEPTH = 0.01
            TTV_AMP = ttv_amp / 24 / 60
            
            time = np.linspace(0, 80, 6000)
            flux = np.ones_like(time)
            n_transits = int(80 / PERIOD)
            
            true_ttvs = []
            epochs = []
            
            for n in range(n_transits):
                ttv = TTV_AMP * np.sin(2 * np.pi * n / ttv_period_epochs)
                t0_actual = PERIOD/2 + n * PERIOD + ttv
                true_ttvs.append(ttv * 24 * 60)
                epochs.append(n)
                transit = generate_transit_signal_mandel_agol(time, period=1000, t0=t0_actual, depth=DEPTH, duration=0.12)
                flux = flux * transit
            
            flux = add_noise(flux, 0.0008)
            ttv_result = measure_transit_timing_variations(time, flux, period=PERIOD, t0=PERIOD/2, duration=0.12)
            st.session_state['ttv_data'] = {'result': ttv_result, 'true': {'epochs': epochs, 'ttvs': true_ttvs}}
    
    if 'ttv_data' in st.session_state:
        data = st.session_state['ttv_data']
        result = data['result']
        true = data['true']
        
        st.markdown("### TTV Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("TTVs Detected", "Yes" if result['ttvs_detected'] else "No")
        col2.metric("RMS TTV", f"{result['rms_ttv']*24*60:.2f} min")
        col3.metric("Transits Measured", len(result['ttvs']))
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        measured_epochs = np.array(result['epochs'])
        measured_ttvs = np.array(result['ttvs']) * 24 * 60
        
        axes[0].plot(measured_epochs, measured_ttvs, 'bo', ms=6, label='Measured TTVs')
        axes[0].axhline(0, color='gray', ls='--')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('O-C (minutes)')
        axes[0].set_title('Transit Timing Variations (O-C Diagram)')
        axes[0].legend()
        
        axes[1].plot(true['epochs'], true['ttvs'], 'g-', lw=2, label='Injected TTVs')
        axes[1].plot(measured_epochs, measured_ttvs, 'ro', ms=4, label='Recovered')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('TTV (minutes)')
        axes[1].set_title('Injected vs Recovered TTVs')
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================================================================
# BATCH ANALYSIS PAGE
# =============================================================================
elif page == "Batch Analysis":
    st.header("Batch Analysis")
    st.write("Test detection efficiency across parameter space.")
    
    st.markdown("### Injection-Recovery Test")
    col1, col2 = st.columns(2)
    n_trials = col1.number_input("Trials per depth", 3, 20, 5)
    noise_level = col2.number_input("Noise (ppm)", 500, 3000, 1000, 100)
    
    if st.button("Run Injection-Recovery", type="primary"):
        depths = [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
        recovery_rates = []
        mean_snrs = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, depth in enumerate(depths):
            status.text(f"Testing depth {depth*1e6:.0f} ppm...")
            recovered = 0
            snrs = []
            
            for trial in range(n_trials):
                time = np.linspace(0, 30, 1500)
                flux = generate_transit_signal_mandel_agol(time, period=5.0, depth=depth)
                flux = add_noise(flux, noise_level/1e6)
                try:
                    result = find_transits_bls_advanced(time, flux)
                    if abs(result['period'] - 5.0) / 5.0 < 0.05:
                        recovered += 1
                        snrs.append(result['snr'])
                except:
                    pass
            
            recovery_rates.append(recovered / n_trials * 100)
            mean_snrs.append(np.mean(snrs) if snrs else 0)
            progress.progress((i + 1) / len(depths))
        
        status.text("Done!")
        st.session_state['inj_data'] = {'depths': depths, 'recovery': recovery_rates, 'snrs': mean_snrs}
    
    if 'inj_data' in st.session_state:
        data = st.session_state['inj_data']
        st.markdown("### Results")
        df = pd.DataFrame({'Depth (ppm)': [d*1e6 for d in data['depths']], 'Recovery (%)': data['recovery'], 'Mean SNR': [f"{s:.1f}" for s in data['snrs']]})
        st.table(df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot([d*1e6 for d in data['depths']], data['recovery'], 'bo-', ms=8, lw=2)
        ax.axhline(50, color='gray', ls='--', label='50% threshold')
        ax.set_xlabel('Transit Depth (ppm)')
        ax.set_ylabel('Recovery Rate (%)')
        ax.set_title('Detection Efficiency vs Transit Depth')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

st.markdown("---")
st.markdown("*TransitKit v2.0 | Created by Arif Solmaz*")
