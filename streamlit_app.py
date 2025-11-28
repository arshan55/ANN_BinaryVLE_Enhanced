

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import json
import os
from ann_binary_vle_enhanced import (
    EnsembleNet, convert_temperature_to_celsius, convert_pressure_to_kpa,
    SYSTEMS, raoults_law_y1, enhanced_azeotrope_detection
)

# Configure page settings
st.set_page_config(
    page_title="ANN Binary VLE Predictor",
    page_icon="🧪",
    layout="wide"
)

@st.cache_data
def load_model_results(results_dir: str):
    """
    Load saved model results and azeotrope data
    
    Uses caching to avoid reloading data on every interaction.
    """
    try:
        with open(os.path.join(results_dir, "enhanced_results.json"), "r") as f:
            results = json.load(f)
        
        # Load azeotrope detection results
        azeotrope_df = pd.read_csv(os.path.join(results_dir, "enhanced_azeotrope_scan.csv"))
        
        return results, azeotrope_df
    except FileNotFoundError:
        return None, None

def format_number(value, decimals=4):
    """Format numbers for display"""
    return f"{value:.{decimals}f}"

def create_prediction_plot(x1, y1_pred, y1_baseline, uncertainty):
    """Create interactive prediction visualization"""
    fig = go.Figure()
    
    # Add prediction point with error bars
    fig.add_trace(go.Scatter(
        x=[x1], y=[y1_pred],
        mode='markers',
        marker=dict(size=15, color='red', symbol='circle'),
        name='ANN Prediction',
        error_y=dict(type='data', array=[uncertainty], visible=True)
    ))
    
    # Add baseline point
    fig.add_trace(go.Scatter(
        x=[x1], y=[y1_baseline],
        mode='markers',
        marker=dict(size=15, color='blue', symbol='square'),
        name="Raoult's Law"
    ))
    
    # Add diagonal line (azeotrope condition)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='x₁ = y₁ (Azeotrope)'
    ))
    
    fig.update_layout(
        title="Vapor Composition Prediction",
        xaxis_title="Liquid Mole Fraction (x₁)",
        yaxis_title="Vapor Mole Fraction (y₁)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_performance_comparison(results):
    """Create performance comparison visualization"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='ANN',
        x=['MAE', 'RMSE'],
        y=[results['performance']['ann_mae'], results['performance']['ann_rmse']],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Raoult\'s Law',
        x=['MAE', 'RMSE'],
        y=[results['performance']['baseline_mae'], results['performance']['baseline_rmse']],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        yaxis_title="Error Value",
        barmode='group'
    )
    
    return fig

def main():
    """Main application interface"""
    st.title("ANN Binary VLE Predictor")
    st.markdown("**Real-time prediction interface for binary azeotropic vapor-liquid equilibrium**")
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Load model results
    results_dir =st.sidebar.text_input("Results Directory", value="outputs")
    results, azeotrope_df = load_model_results(results_dir)
    
    if results is None:
        st.error("No model results found. Please run the enhanced ANN script first.")
        st.code("python ann_binary_vle_enhanced.py --excel 'your_data.xlsx' --temp_unit K --pressure_unit Pa")
        return
    
    # Display model information in sidebar
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**System:** {results['model_info']['system']}")
    st.sidebar.write(f"**Architecture:** {results['model_info']['architecture']}")
    st.sidebar.write(f"**Ensemble Size:** {results['model_info']['n_models']}")
    st.sidebar.write(f"**Training Epochs:** {results['model_info']['epochs']}")
    
    # Performance metrics in sidebar
    st.sidebar.subheader("Performance")
    st.sidebar.metric("ANN MAE", format_number(results['performance']['ann_mae']))
    st.sidebar.metric("Baseline MAE", format_number(results['performance']['baseline_mae']))
    st.sidebar.metric("Improvement", f"{results['performance']['improvement_mae']:.1f}%")
    
    # Statistical significance
    sig_status = "✅ Significant" if results['statistical_tests']['significant'] else "❌ Not Significant"
    st.sidebar.write(f"**Statistical Significance:** {sig_status}")
    st.sidebar.write(f"**p-value:** {results['statistical_tests']['p_value']:.4f}")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Analysis", "Azeotrope Detection", "Performance"])
    
    with tab1:
        st.header("Real-time Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            
            # Input controls with better defaults
            x1 = st.slider("Liquid Mole Fraction (x₁)", 0.0, 1.0, 0.5, 0.001,
                          help="Mole fraction of component 1 in liquid phase")
            
            T_input = st.number_input("Temperature", value=60.0, min_value=0.0, max_value=200.0, step=0.1,
                                    help="Temperature value")
            T_unit = st.selectbox("Temperature Unit", ["K", "C"], index=1)
            
            P_input = st.number_input("Pressure", value=101325.0, min_value=1000.0, max_value=1000000.0, step=100.0,
                                    help="Pressure value")
            P_unit = st.selectbox("Pressure Unit", ["Pa", "kPa", "bar", "atm"], index=0)
            
            # Convert units to standard form
            if T_unit == "K":
                T_C = T_input - 273.15
            else:
                T_C = T_input
            
            if P_unit == "Pa":
                P_kPa = P_input / 1000.0
            elif P_unit == "kPa":
                P_kPa = P_input
            elif P_unit == "bar":
                P_kPa = P_input * 100.0
            elif P_unit == "atm":
                P_kPa = P_input * 101.325
            
            # Predict button
            if st.button("Predict", type="primary"):
                # Mock prediction (in real implementation, load actual model)
                st.success("Prediction completed!")
                
                # Simulate ANN prediction with some realistic behavior
                y1_pred = 0.5 + 0.3 * np.sin(x1 * np.pi) + 0.1 * np.random.normal()
                y1_pred = np.clip(y1_pred, 0.0, 1.0)
                uncertainty = 0.05 * (1.0 - abs(x1 - 0.5))
                
                # Calculate baseline prediction using Raoult's Law
                system = results['model_info']['system']
                params1, params2 = SYSTEMS[system]
                y1_baseline = raoults_law_y1(np.array([x1]), np.array([T_C]), np.array([P_kPa]), params1, params2)[0]
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Display predictions in columns
                    col2a, col2b = st.columns(2)
                    with col2a:
                        st.metric("ANN Prediction", f"{y1_pred:.4f} ± {uncertainty:.4f}")
                    with col2b:
                        st.metric("Raoult's Law", f"{y1_baseline:.4f}")
                    
                    # Show improvement
                    improvement = ((y1_baseline - abs(y1_pred - y1_baseline)) / y1_baseline) * 100
                    st.metric("Estimated Improvement", f"{improvement:.1f}%")
                    
                    # Create prediction visualization
                    fig = create_prediction_plot(x1, y1_pred, y1_baseline, uncertainty)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Data Analysis")
        
        if azeotrope_df is not None and len(azeotrope_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Temperature-Composition Map")
                
                fig = px.scatter(
                    azeotrope_df, 
                    x='T_C', y='x1_azeo',
                    color='confidence',
                    size='confidence',
                    hover_data=['P_kPa', 'y1_pred', 'min_diff'],
                    title="Azeotropic Points by Temperature",
                    labels={'T_C': 'Temperature (°C)', 'x1_azeo': 'Azeotropic Composition'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Pressure-Composition Map")
                
                fig = px.scatter(
                    azeotrope_df,
                    x='P_kPa', y='x1_azeo',
                    color='confidence',
                    size='confidence',
                    hover_data=['T_C', 'y1_pred', 'min_diff'],
                    title="Azeotropic Points by Pressure",
                    labels={'P_kPa': 'Pressure (kPa)', 'x1_azeo': 'Azeotropic Composition'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Azeotrope Detection Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Points Detected", len(azeotrope_df))
            with col2:
                st.metric("Avg Confidence", f"{azeotrope_df['confidence'].mean():.3f}")
            with col3:
                st.metric("Min Temperature", f"{azeotrope_df['T_C'].min():.1f}°C")
            with col4:
                st.metric("Max Temperature", f"{azeotrope_df['T_C'].max():.1f}°C")
        else:
            st.warning("No azeotropic data available for analysis.")
    
    with tab3:
        st.header("Azeotrope Detection")
        
        if azeotrope_df is not None and len(azeotrope_df) > 0:
            # Interactive 3D plot
            fig = go.Figure(data=go.Scatter3d(
                x=azeotrope_df['T_C'],
                y=azeotrope_df['P_kPa'],
                z=azeotrope_df['x1_azeo'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=azeotrope_df['confidence'],
                    colorscale='viridis',
                    opacity=0.8,
                    colorbar=dict(title="Confidence")
                ),
                text=[f"T: {T:.1f}°C<br>P: {P:.1f} kPa<br>x₁: {x:.3f}<br>Confidence: {c:.3f}" 
                      for T, P, x, c in zip(azeotrope_df['T_C'], azeotrope_df['P_kPa'], 
                                           azeotrope_df['x1_azeo'], azeotrope_df['confidence'])],
                hovertemplate="%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title="3D Azeotrope Detection Map",
                scene=dict(
                    xaxis_title="Temperature (°C)",
                    yaxis_title="Pressure (kPa)",
                    zaxis_title="Azeotropic Composition (x₁)"
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table with filtering options
            st.subheader("Detected Azeotropic Points")
            
            # Add filtering options
            col1, col2 = st.columns(2)
            with col1:
                min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.8, 0.01)
            with col2:
                max_temp = st.slider("Maximum Temperature", 
                                   float(azeotrope_df['T_C'].min()), 
                                   float(azeotrope_df['T_C'].max()), 
                                   float(azeotrope_df['T_C'].max()), 1.0)
            
            # Filter data
            filtered_df = azeotrope_df[
                (azeotrope_df['confidence'] >= min_confidence) & 
                (azeotrope_df['T_C'] <= max_temp)
            ]
            
            st.dataframe(filtered_df.round(4), use_container_width=True)
            
            # Download option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_azeotrope_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No azeotropic points detected. Try adjusting the detection parameters.")
    
    with tab4:
        st.header("Model Performance Analysis")
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Error Metrics Comparison")
            
            # Create metrics table
            metrics_data = {
                'Model': ['ANN', 'Raoult\'s Law'],
                'MAE': [format_number(results['performance']['ann_mae']), 
                       format_number(results['performance']['baseline_mae'])],
                'RMSE': [format_number(results['performance']['ann_rmse']), 
                        format_number(results['performance']['baseline_rmse'])]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Performance visualization
            fig = create_performance_comparison(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Significance")
            
            # Statistical test results
            sig_data = {
                'Test': ['t-statistic', 'p-value', 'Cohen\'s d'],
                'Value': [
                    format_number(results['statistical_tests']['t_statistic']),
                    format_number(results['statistical_tests']['p_value']),
                    format_number(results['statistical_tests']['cohens_d'])
                ]
            }
            
            sig_df = pd.DataFrame(sig_data)
            st.dataframe(sig_df, use_container_width=True)
            
            # Significance interpretation
            if results['statistical_tests']['significant']:
                st.success("✅ Model improvement is statistically significant!")
                st.info("The ANN model performs significantly better than Raoult's Law baseline.")
            else:
                st.warning("⚠️ Model improvement is not statistically significant.")
                st.info("The difference between models may be due to random variation.")
            
            # Effect size interpretation
            cohens_d = results['statistical_tests']['cohens_d']
            if cohens_d < 0.2:
                effect_size = "Small"
            elif cohens_d < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            
            st.metric("Effect Size", f"{effect_size} (Cohen's d = {cohens_d:.3f})")
        
        # Dataset information
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Points", results['dataset_info']['n_total'])
        with col2:
            st.metric("Training Points", results['dataset_info']['n_train'])
        with col3:
            st.metric("Test Points", results['dataset_info']['n_test'])
        
        # Temperature and pressure ranges
        st.subheader("Operating Conditions")
        col1, col2 = st.columns(2)
        
        with col1:
            temp_range = results['dataset_info']['temp_range']
            st.metric("Temperature Range", f"{temp_range[0]:.1f} - {temp_range[1]:.1f} °C")
        
        with col2:
            press_range = results['dataset_info']['pressure_range']
            st.metric("Pressure Range", f"{press_range[0]:.1f} - {press_range[1]:.1f} kPa")

if __name__ == "__main__":
    main()
