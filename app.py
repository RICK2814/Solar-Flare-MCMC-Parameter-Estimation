"""
Solar Flare MCMC Parameter Estimation
Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io

# Note: intentionally not calling `st.set_page_config` here to avoid
# StreamlitAPIException when the app is re-run; set this in deployment
# settings if needed.

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("☀️ Solar Flare MCMC Parameter Estimation")
st.markdown("**Bayesian inference using Metropolis-Hastings algorithm**")

# Flare model - simplified and numerically stable
def flare_model(t, A, tau, omega):
    """Solar flare intensity model: dampened oscillation"""
    try:
        # Avoid overflow: use dampened exponential decay instead of growth
        # Model: A * exp(-(t-tau)^2 / (2*tau^2)) * sin(omega * t)
        decay = np.exp(-((t - tau) ** 2) / (2 * tau ** 2))
        oscillation = np.sin(omega * t)
        result = A * decay * oscillation
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return np.full_like(t, 0.0, dtype=float)
        return result
    except Exception:
        return np.full_like(t, 0.0, dtype=float)

# Calculate log-likelihood
def calculate_log_likelihood(params, t_data, s_data):
    """Calculate log-likelihood for given parameters"""
    try:
        A, tau, omega = params
        if A <= 0 or omega <= 0:
            return -1e6
        y_model = flare_model(t_data, A, tau, omega)
        if np.any(np.isnan(y_model)) or np.any(np.isinf(y_model)):
            return -1e6
        # Relative error of 30% (more forgiving than 20%)
        sigma = 0.3 * np.maximum(np.abs(s_data), 0.1)
        residuals = s_data - y_model
        log_l = -0.5 * np.sum((residuals ** 2) / (sigma ** 2))
        if np.isnan(log_l) or np.isinf(log_l):
            return -1e6
        return log_l
    except Exception:
        return -1e6

# Prior bounds check
def is_valid_params(params):
    """Check if parameters are within prior bounds"""
    A, tau, omega = params
    return (0 < A < 2) and (1 < tau < 10) and (1 < omega < 20)

# MCMC step
def mcmc_step(current_params, current_log_l, t_data, s_data, step_sizes):
    """Perform one Metropolis-Hastings step"""
    A, tau, omega = current_params
    
    # Propose new parameters
    proposed = np.array([
        A + np.random.uniform(-step_sizes[0], step_sizes[0]),
        tau + np.random.uniform(-step_sizes[1], step_sizes[1]),
        omega + np.random.uniform(-step_sizes[2], step_sizes[2])
    ])
    
    # Check prior bounds
    if not is_valid_params(proposed):
        return current_params, current_log_l, False
    
    # Calculate proposed log-likelihood
    proposed_log_l = calculate_log_likelihood(proposed, t_data, s_data)
    
    # Acceptance criterion
    log_alpha = proposed_log_l - current_log_l
    accept = np.log(np.random.uniform()) < log_alpha
    
    if accept:
        return proposed, proposed_log_l, True
    else:
        return current_params, current_log_l, False

# Load data
@st.cache_data
def load_data():
    """Load flare data from CSV"""
    try:
        # Try to load from uploaded file or default location
        df = pd.read_csv('flare_data.csv', sep=r'\s+', header=None, names=['t', 's'])
        # Coerce to numeric and drop invalid rows
        df['t'] = pd.to_numeric(df['t'], errors='coerce')
        df['s'] = pd.to_numeric(df['s'], errors='coerce')
        df = df.dropna()
        if df.empty:
            raise ValueError('No numeric data found in flare_data.csv')
        return df['t'].values.astype(float), df['s'].values.astype(float)
    except:
        st.error("⚠️ Please upload flare_data.csv file")
        return None, None

# Initialize session state
if 'mcmc_state' not in st.session_state:
    st.session_state.mcmc_state = {
        'A': 0.5,
        'tau': 2.5,
        'omega': 5.0,
        'iteration': 0,
        'accepted': 0,
        'log_likelihood': -np.inf,
        'traces': {'A': [], 'tau': [], 'omega': [], 'log_l': []},
        'best_params': None,
        'best_log_l': -np.inf,
        'running': False,
        'error_count': 0
    }
# Initialize a small in-memory log for diagnostics
if 'mcmc_logs' not in st.session_state:
    st.session_state.mcmc_logs = []

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader("Upload flare_data.csv", type=['csv', 'txt'])
    if uploaded_file is not None:
        # Read raw bytes and decode (handle BOM)
        raw = uploaded_file.read()
        try:
            text = raw.decode('utf-8-sig')
        except Exception:
            try:
                text = raw.decode('latin-1')
            except Exception:
                st.error("⚠️ Unable to decode uploaded file; ensure it's UTF-8 or Latin-1 encoded.")
                t_data, s_data = None, None
                text = None

        if text is not None:
            lines = text.splitlines()
            
            # Skip leading comment/header lines (lines that don't start with a number)
            skip_rows = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not any(c in stripped[0] for c in '0123456789-+.'):
                    skip_rows += 1
                else:
                    break
            
            # Try to parse with pandas auto-detecting the delimiter
            parse_success = False
            parsed_df = None

            # Attempt 1: engine='python' with sep=None (auto-detect), skipping non-data rows
            try:
                parsed_df = pd.read_csv(io.StringIO(text), sep=None, engine='python', header=None, skiprows=skip_rows)
            except Exception:
                pass

            # Attempt 2: whitespace-separated
            if parsed_df is None or parsed_df.shape[1] == 1:
                try:
                    parsed_df = pd.read_csv(io.StringIO(text), delim_whitespace=True, header=None, skiprows=skip_rows)
                except Exception:
                    pass

            # Attempt 3: comma-separated
            if parsed_df is None or parsed_df.shape[1] == 1:
                try:
                    parsed_df = pd.read_csv(io.StringIO(text), sep=',', header=None, skiprows=skip_rows)
                except Exception:
                    pass

            # If parsed, try to find two numeric columns
            if parsed_df is not None and parsed_df.shape[1] >= 2:
                # Coerce every column to numeric and keep columns with numeric values
                numeric_cols = []
                for i, c in enumerate(parsed_df.columns):
                    converted = pd.to_numeric(parsed_df[c], errors='coerce')
                    non_na = converted.notna().sum()
                    numeric_cols.append((i, non_na, converted))

                # Sort by number of numeric values desc
                numeric_cols.sort(key=lambda x: x[1], reverse=True)
                if len(numeric_cols) >= 2 and numeric_cols[0][1] > 0 and numeric_cols[1][1] > 0:
                    df2 = pd.DataFrame({
                        't': numeric_cols[0][2],
                        's': numeric_cols[1][2]
                    })
                    df2 = df2.dropna()
                    if not df2.empty:
                        t_data = df2['t'].values.astype(float)
                        s_data = df2['s'].values.astype(float)
                        st.success(f"✅ Loaded {len(t_data)} data points (skipped {skip_rows} header rows)")
                        parse_success = True

            if parse_success:
                # Cache loaded data in session_state so reruns don't lose it
                st.session_state.loaded_data = (t_data, s_data)

            if not parse_success:
                with st.expander("Debug: File format"):
                    st.write(f"Skipped rows: {skip_rows}")
                    st.write(f"First 10 lines: {lines[:10]}")
                    if parsed_df is not None:
                        st.write(f"Parsed shape: {parsed_df.shape}")
                        st.write("Preview:")
                        st.dataframe(parsed_df.head(10))
                st.error("⚠️ Uploaded file contains no numeric data or couldn't be parsed automatically.")
                t_data, s_data = None, None
    else:
        # Prefer cached uploaded/loaded data so reruns don't reset the dataset
        if 'loaded_data' in st.session_state and st.session_state.loaded_data is not None:
            t_data, s_data = st.session_state.loaded_data
        else:
            t_data, s_data = load_data()
            if t_data is not None and s_data is not None:
                st.session_state.loaded_data = (t_data, s_data)
    
    st.divider()
    
    # MCMC parameters
    st.subheader("MCMC Settings")
    max_iterations = st.number_input("Max Iterations", 100, 50000, 10000, 1000)
    step_sizes = [
        st.slider("Step size (A)", 0.01, 0.2, 0.05, 0.01),
        st.slider("Step size (τ)", 0.05, 0.5, 0.1, 0.05),
        st.slider("Step size (ω)", 0.1, 2.0, 0.5, 0.1)
    ]
    batch_size = st.number_input("Batch Size (iterations per rerun)", 1, 5000, 100, 10)
    
    st.divider()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("▶️ Start" if not st.session_state.mcmc_state['running'] else "⏸️ Pause"):
            st.session_state.mcmc_state['running'] = not st.session_state.mcmc_state['running']
    
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.mcmc_state = {
                'A': 0.5,
                'tau': 2.5,
                'omega': 5.0,
                'iteration': 0,
                'accepted': 0,
                'log_likelihood': -np.inf,
                'traces': {'A': [], 'tau': [], 'omega': [], 'log_l': []},
                'best_params': None,
                'best_log_l': -np.inf,
                'running': False,
                'error_count': 0
            }
            # preserve loaded data across resets so the user doesn't need to re-upload
            st.session_state.mcmc_logs = []
            st.rerun()
    
    st.divider()
    
    # Current status
    st.subheader("📊 Status")
    state = st.session_state.mcmc_state
    st.metric("Iteration", f"{state['iteration']} / {max_iterations}")
    
    if state['iteration'] > 0:
        acceptance_rate = (state['accepted'] / state['iteration']) * 100
        st.metric("Acceptance Rate", f"{acceptance_rate:.1f}%")
    
    progress = min(state['iteration'] / max_iterations, 1.0)
    st.progress(progress)

# Main content
if t_data is not None and s_data is not None:
    
    # Run MCMC in batches if active (reduces reruns and stabilizes state)
    if st.session_state.mcmc_state['running'] and st.session_state.mcmc_state['iteration'] < max_iterations:
        state = st.session_state.mcmc_state

        # Initialize log-likelihood if not set
        if state['log_likelihood'] == -np.inf:
            state['log_likelihood'] = calculate_log_likelihood(
                np.array([state['A'], state['tau'], state['omega']]), t_data, s_data
            )

        # Ensure logs exist
        if 'mcmc_logs' not in st.session_state:
            st.session_state.mcmc_logs = []

        # Number of iterations to run in this chunk
        n_to_run = min(batch_size, max_iterations - state['iteration'])
        accepted_batch = 0

        for i in range(n_to_run):
            try:
                current_params = np.array([state['A'], state['tau'], state['omega']])
                new_params, new_log_l, accepted = mcmc_step(
                    current_params, state['log_likelihood'], t_data, s_data, step_sizes
                )

                # Validate results: if non-finite, treat as rejected proposal but don't count as exception
                if not (np.isfinite(new_log_l) and np.all(np.isfinite(new_params))):
                    # Log the non-finite proposal for diagnostics but do not increment error_count
                    st.session_state.mcmc_logs.append({'iteration': state['iteration'], 'note': 'non-finite proposal'})
                    # Record trace of current params (no change)
                    state['traces']['A'].append(state['A'])
                    state['traces']['tau'].append(state['tau'])
                    state['traces']['omega'].append(state['omega'])
                    state['traces']['log_l'].append(state['log_likelihood'])
                    state['iteration'] += 1
                    continue

                # Update state
                state['A'], state['tau'], state['omega'] = new_params
                state['log_likelihood'] = new_log_l
                state['iteration'] += 1
                state['accepted'] += int(accepted)
                accepted_batch += int(accepted)

                # Store traces
                state['traces']['A'].append(state['A'])
                state['traces']['tau'].append(state['tau'])
                state['traces']['omega'].append(state['omega'])
                state['traces']['log_l'].append(state['log_likelihood'])

                # Update best parameters (MAP)
                if new_log_l > state['best_log_l']:
                    state['best_params'] = new_params.copy()
                    state['best_log_l'] = new_log_l

            except Exception as e:
                # Count real exceptions; increase threshold so long runs aren't stopped prematurely
                state['error_count'] = state.get('error_count', 0) + 1
                st.session_state.mcmc_logs.append({'iteration': state['iteration'], 'error': str(e)})
                if state['error_count'] > 500:
                    st.warning("⚠️ Repeated exceptions encountered. Stopping MCMC.")
                    state['running'] = False
                    break

        # Append batch diagnostics
        # Trim logs to last 2000 entries and append batch diagnostics
        if len(st.session_state.mcmc_logs) > 2000:
            st.session_state.mcmc_logs = st.session_state.mcmc_logs[-2000:]

        st.session_state.mcmc_logs.append({
            'iteration_end': state['iteration'],
            'accepted_in_batch': int(accepted_batch),
            'error_count': int(state.get('error_count', 0)),
            'A': float(state['A']),
            'tau': float(state['tau']),
            'omega': float(state['omega'])
        })

        # Stop if done
        if state['iteration'] >= max_iterations:
            state['running'] = False
            st.success("✅ MCMC completed!")

        # Continue running in next chunk (rerun to update UI)
        if state['running']:
            time.sleep(0.001)
            st.rerun()
    
    # Display current parameters
    st.subheader("📈 Current Parameters")
    col1, col2, col3 = st.columns(3)
    
    state = st.session_state.mcmc_state
    with col1:
        st.metric("Amplitude (A)", f"{state['A']:.4f}")
    with col2:
        st.metric("Quench Time (τ)", f"{state['tau']:.4f}")
    with col3:
        st.metric("Angular Freq (ω)", f"{state['omega']:.4f}")
    
    # Display MAP estimates
    if state['best_params'] is not None:
        st.success("🎯 **Maximum A Posteriori (MAP) Estimates**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("A (MAP)", f"{state['best_params'][0]:.6f}")
        with col2:
            st.metric("τ (MAP)", f"{state['best_params'][1]:.6f}")
        with col3:
            st.metric("ω (MAP)", f"{state['best_params'][2]:.6f}")
        with col4:
            st.metric("Log-Likelihood", f"{state['best_log_l']:.2f}")
    
    # Tabs for visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Data & Fit", "📉 Trace Plots", "📈 Posteriors"])
    
    with tab1:
        st.subheader("Observed Data vs Model Fit")
        
        if state['best_params'] is not None:
            # Generate model predictions
            y_model = flare_model(t_data, *state['best_params'])
            
            # Create plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t_data, y=s_data,
                mode='lines',
                name='Observed Data',
                line=dict(color='#8884d8', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=t_data, y=y_model,
                mode='lines',
                name='Model Fit',
                line=dict(color='#82ca9d', width=2)
            ))
            
            fig.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Intensity",
                template="plotly_dark",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Start MCMC to see model fit")
    
    with tab2:
        st.subheader("Trace Plots")
        
        if len(state['traces']['A']) > 0:
            iterations = list(range(len(state['traces']['A'])))
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Amplitude (A)", "Quench Time (τ)", "Angular Frequency (ω)"),
                vertical_spacing=0.1
            )
            
            # A trace
            fig.add_trace(go.Scatter(
                x=iterations, y=state['traces']['A'],
                mode='lines',
                name='A',
                line=dict(color='#ff6b6b', width=1)
            ), row=1, col=1)
            
            # tau trace
            fig.add_trace(go.Scatter(
                x=iterations, y=state['traces']['tau'],
                mode='lines',
                name='τ',
                line=dict(color='#4ecdc4', width=1)
            ), row=2, col=1)
            
            # omega trace
            fig.add_trace(go.Scatter(
                x=iterations, y=state['traces']['omega'],
                mode='lines',
                name='ω',
                line=dict(color='#ffe66d', width=1)
            ), row=3, col=1)
            
            fig.update_xaxes(title_text="Iteration", row=3, col=1)
            fig.update_yaxes(title_text="Value", range=[0, 2], row=1, col=1)
            fig.update_yaxes(title_text="Value", range=[1, 10], row=2, col=1)
            fig.update_yaxes(title_text="Value", range=[1, 20], row=3, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                height=800,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Start MCMC to see trace plots")
    
    with tab3:
        st.subheader("Posterior Distributions")
        
        if len(state['traces']['A']) > 100:  # Need enough samples
            # Remove burn-in (first 20%)
            burn_in = int(len(state['traces']['A']) * 0.2)
            
            traces_burned = {
                'A': state['traces']['A'][burn_in:],
                'tau': state['traces']['tau'][burn_in:],
                'omega': state['traces']['omega'][burn_in:]
            }
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    "Posterior: Amplitude (A)",
                    "Posterior: Quench Time (τ)",
                    "Posterior: Angular Frequency (ω)"
                ),
                vertical_spacing=0.1
            )
            
            # A histogram
            fig.add_trace(go.Histogram(
                x=traces_burned['A'],
                name='A',
                marker_color='#ff6b6b',
                nbinsx=30
            ), row=1, col=1)
            
            # tau histogram
            fig.add_trace(go.Histogram(
                x=traces_burned['tau'],
                name='τ',
                marker_color='#4ecdc4',
                nbinsx=30
            ), row=2, col=1)
            
            # omega histogram
            fig.add_trace(go.Histogram(
                x=traces_burned['omega'],
                name='ω',
                marker_color='#ffe66d',
                nbinsx=30
            ), row=3, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                height=800,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display statistics
            st.subheader("📊 Posterior Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Amplitude (A)**")
                st.write(f"Mean: {np.mean(traces_burned['A']):.4f}")
                st.write(f"Std: {np.std(traces_burned['A']):.4f}")
            
            with col2:
                st.write("**Quench Time (τ)**")
                st.write(f"Mean: {np.mean(traces_burned['tau']):.4f}")
                st.write(f"Std: {np.std(traces_burned['tau']):.4f}")
            
            with col3:
                st.write("**Angular Frequency (ω)**")
                st.write(f"Mean: {np.mean(traces_burned['omega']):.4f}")
                st.write(f"Std: {np.std(traces_burned['omega']):.4f}")
        else:
            st.info("👆 Run more iterations to see posterior distributions")

    # Diagnostics: show recent MCMC logs
    with st.expander("🧾 MCMC Diagnostics (recent)"):
        logs = st.session_state.get('mcmc_logs', [])
        if logs:
            # Show last 20 entries
            for entry in logs[-20:]:
                st.write(entry)
        else:
            st.write("No diagnostics yet. Start MCMC to collect logs.")

else:
    st.warning("⚠️ Please upload flare_data.csv to begin")
    st.info("The file should be space or comma-separated with two columns: time (t) and intensity (s)")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Solar Flare MCMC Parameter Estimation | Built with Streamlit</p>
        <p>Competition: Solar Flare Pulse - January 2026</p>
    </div>
""", unsafe_allow_html=True)
"""
Solar Flare MCMC Parameter Estimation
Streamlit Web Application
"""

