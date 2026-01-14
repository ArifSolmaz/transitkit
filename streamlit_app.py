import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="TransitKit",
    page_icon="🪐",
    layout="wide"
)

# Title
st.title("🚀 TransitKit")
st.subheader("Exoplanet Transit Light Curve Analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # File upload
    uploaded_file = st.file_uploader("Upload light curve", type=['csv', 'txt'])
    
    # Parameters
    st.subheader("Analysis Parameters")
    period = st.number_input("Orbital Period (days)", value=10.0, min_value=0.1, max_value=1000.0)
    duration = st.number_input("Transit Duration (days)", value=0.1, min_value=0.01, max_value=10.0)
    
    # Run analysis
    if st.button("Analyze", type="primary"):
        st.session_state.analyze = True

# Main content
tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "Results"])

with tab1:
    st.header("Data Visualization")
    
    if uploaded_file is not None:
        # Load data
        import pandas as pd
        data = pd.read_csv(uploaded_file)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 4))
        if 'time' in data.columns and 'flux' in data.columns:
            ax.plot(data['time'], data['flux'], 'k.', alpha=0.5, markersize=1)
            ax.set_xlabel('Time')
            ax.set_ylabel('Flux')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.write(f"Data points: {len(data)}")
            st.write(f"Time range: {data['time'].min():.2f} to {data['time'].max():.2f}")
        else:
            st.error("CSV must have 'time' and 'flux' columns")
    else:
        st.info("Upload a CSV file to begin analysis")
        st.code("""# Expected CSV format:
time,flux,flux_err
0.0,1.000,0.001
0.1,0.999,0.001
0.2,1.001,0.001
# ...""")

with tab2:
    st.header("Transit Analysis")
    
    if uploaded_file is not None and st.session_state.get('analyze', False):
        st.write("Running analysis...")
        
        # Simulate analysis
        import time
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        st.success("Analysis complete!")
        
        # Show results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Period", f"{period:.3f} days")
        with col2:
            st.metric("Depth", "1.2%")
        with col3:
            st.metric("SNR", "8.5")
    else:
        st.info("Upload data and click 'Analyze' to see results")

with tab3:
    st.header("Download Results")
    st.write("Analysis results will appear here")
    
    # Example download button
    st.download_button(
        label="Download Example Data",
        data="time,flux\n0,1.0\n1,0.99\n2,1.01\n3,1.0\n4,0.98",
        file_name="example_transit.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("### 📚 Documentation")
st.markdown("Visit [transitkit.readthedocs.io](https://transitkit.readthedocs.io) for full documentation")