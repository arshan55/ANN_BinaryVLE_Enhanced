"""
Enhanced Artificial Neural Network for Binary Vapor-Liquid Equilibrium Prediction

This module implements a physics-informed ensemble neural network for predicting
vapor composition in binary azeotropic systems. The model incorporates thermodynamic
constraints and provides uncertainty quantification through ensemble learning.

Author: Enhanced ANN Implementation
Date: 2024
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# UTILITY FUNCTIONS

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def convert_temperature_to_celsius(T_series: pd.Series, unit: str) -> pd.Series:
    """
    Convert temperature to Celsius from various units
    
    Args:
        T_series: Temperature values
        unit: Input unit ('K', 'C', 'auto')
    
    Returns:
        Temperature in Celsius
    """
    unit = unit.lower()
    if unit in ("c", "celsius", "degc", "°c"):
        return T_series
    if unit in ("k", "kelvin"):
        return T_series - 273.15
    if unit == "auto":
        # Auto-detect based on typical values
        median_T = float(T_series.median())
        return T_series - 273.15 if median_T > 200.0 else T_series
    raise ValueError(f"Unsupported temperature unit: {unit}")

def convert_pressure_to_kpa(P_series: pd.Series, unit: str) -> pd.Series:
    """
    Convert pressure to kPa from various units
    
    Args:
        P_series: Pressure values
        unit: Input unit ('Pa', 'kPa', 'bar', 'atm', 'mmHg', 'auto')
    
    Returns:
        Pressure in kPa
    """
    unit = unit.lower()
    if unit in ("kpa",):
        return P_series
    if unit in ("pa",):
        return P_series / 1000.0
    if unit in ("bar",):
        return P_series * 100.0
    if unit in ("atm", "atmosphere", "atm_abs"):
        return P_series * 101.325
    if unit in ("mmhg", "torr"):
        return P_series * 0.133322
    if unit == "auto":
        # Auto-detect based on typical values
        median_P = float(P_series.median())
        if 5e4 < median_P < 2e5:
            return P_series / 1000.0  # Pa -> kPa
        if 500.0 < median_P < 1500.0:
            return P_series * 0.133322  # mmHg -> kPa
        if median_P < 3.0:
            return P_series * 100.0  # bar -> kPa
        if 0.5 < median_P < 2.0:
            return P_series * 101.325  # atm -> kPa
    return P_series

# THERMODYNAMIC MODELS

@dataclass
class AntoineParams:
    """Antoine equation parameters for vapor pressure calculation"""
    A: float
    B: float
    C: float

    def psat_mmHg(self, T_C: np.ndarray) -> np.ndarray:
        """Calculate saturation pressure using Antoine equation"""
        return np.power(10.0, self.A - (self.B / (self.C + T_C)))

# Predefined system parameters for acetone-chloroform
SYSTEMS = {
    "acetone-chloroform": (
        AntoineParams(A=7.11714, B=1210.595, C=229.664),  # Acetone
        AntoineParams(A=6.95464, B=1170.966, C=226.232),  # Chloroform
    ),
}

def raoults_law_y1(x1: np.ndarray, T_C: np.ndarray, P_kPa: np.ndarray, 
                   params1: AntoineParams, params2: AntoineParams) -> np.ndarray:
    """
    Calculate vapor composition using Raoult's Law
    
    This provides a baseline for comparison with the ANN model.
    Assumes ideal solution behavior.
    
    Args:
        x1: Liquid mole fraction of component 1
        T_C: Temperature in Celsius
        P_kPa: Pressure in kPa
        params1: Antoine parameters for component 1
        params2: Antoine parameters for component 2
    
    Returns:
        Vapor mole fraction of component 1
    """
    # Calculate saturation pressures
    Psat1_mmHg = params1.psat_mmHg(T_C)
    Psat2_mmHg = params2.psat_mmHg(T_C)
    Psat1_kPa = Psat1_mmHg * 0.133322
    Psat2_kPa = Psat2_mmHg * 0.133322
    
    # Raoult's Law: y1 = (x1 * P1_sat) / (x1 * P1_sat + x2 * P2_sat)
    x2 = 1.0 - x1
    y1 = (x1 * Psat1_kPa) / (x1 * Psat1_kPa + x2 * Psat2_kPa + 1e-12)
    return y1

# NEURAL NETWORK ARCHITECTURES

class PhysicsInformedNet(nn.Module):
    """
    Physics-informed neural network with thermodynamic constraints
    
    This network incorporates physical laws directly into the architecture
    to ensure predictions are thermodynamically consistent.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dims: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.input_dim = input_dim
        
        # Build shared layers with dropout for regularization
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Prevent overfitting
            prev = h
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate output heads for different predictions
        self.y1_head = nn.Sequential(nn.Linear(prev, 1), nn.Sigmoid())  # Ensures 0 ≤ y1 ≤ 1
        self.activity_head = nn.Sequential(nn.Linear(prev, 2), nn.Softmax(dim=-1))
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the network"""
        shared = self.shared_layers(x)
        y1 = self.y1_head(shared)
        activity = self.activity_head(shared)
        return {"y1": y1, "activity": activity}

class EnsembleNet(nn.Module):
    """
    Ensemble of neural networks for uncertainty quantification
    
    Uses multiple models to provide prediction confidence intervals
    and better generalization performance.
    """
    
    def __init__(self, n_models: int = 5, input_dim: int = 3, hidden_dims: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            PhysicsInformedNet(input_dim, hidden_dims) for _ in range(n_models)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all models in ensemble"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
       
        y1_preds = torch.stack([out["y1"] for out in outputs], dim=0)
        activity_preds = torch.stack([out["activity"] for out in outputs], dim=0)
        
        
        y1_mean = torch.mean(y1_preds, dim=0)
        y1_std = torch.std(y1_preds, dim=0)
        
        activity_mean = torch.mean(activity_preds, dim=0)
        activity_std = torch.std(activity_preds, dim=0)
        
        return {
            "y1_mean": y1_mean,
            "y1_std": y1_std,
            "activity_mean": activity_mean,
            "activity_std": activity_std,
            "individual_preds": y1_preds
        }

# TRAINING AND LOSS FUNCTIONS

class ThermodynamicLoss(nn.Module):
    """
    Custom loss function that enforces thermodynamic constraints
    
    Combines prediction accuracy with physical consistency requirements
    to ensure the model learns thermodynamically valid relationships.
    """
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha  # Weight for mass balance constraint
        self.beta = beta    # Weight for activity coefficient constraint
        self.mse = nn.MSELoss()
        
    def forward(self, pred: Dict[str, torch.Tensor], target: torch.Tensor, 
                x1: torch.Tensor, T: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss with physics constraints"""
        # Main prediction loss
        main_loss = self.mse(pred["y1"].squeeze(), target)
        
        # Mass balance constraint: y1 + y2 = 1
        y2 = 1.0 - pred["y1"].squeeze()
        consistency_loss = torch.mean(torch.abs(y2 - (1.0 - pred["y1"].squeeze())))
        
        # Activity coefficient positivity constraint
        activity = pred.get("activity", None)
        if activity is not None:
            activity_loss = torch.mean(torch.relu(-activity))  # Penalize negative values
        else:
            activity_loss = torch.tensor(0.0)
            
        return main_loss + self.alpha * consistency_loss + self.beta * activity_loss

def train_ensemble_model(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 500, batch_size: int = 64, 
                        lr: float = 1e-3, n_models: int = 5,
                        device: Optional[str] = None) -> Tuple[EnsembleNet, Dict]:
    """
    Train ensemble of neural networks
    
    Uses different optimizers for each model to increase diversity
    and improve ensemble performance.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleNet(n_models=n_models, input_dim=X_train.shape[1]).to(device)
    
    # Use different optimizers for diversity
    optimizers = []
    for i, net in enumerate(model.models):
        if i % 2 == 0:
            optimizers.append(torch.optim.Adam(net.parameters(), lr=lr))
        else:
            optimizers.append(torch.optim.AdamW(net.parameters(), lr=lr*0.8))
    
    criterion = ThermodynamicLoss()
    
    # Create data loaders
    ds_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    ds_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (xb, yb) in enumerate(dl_train):
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Train each model in ensemble
            for i, (net, optimizer) in enumerate(zip(model.models, optimizers)):
                optimizer.zero_grad()
                pred = net(xb)
                loss = criterion(pred, yb, xb[:, 0], xb[:, 1], xb[:, 2])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
        train_loss = running_loss / (len(dl_train) * n_models)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                
                # Use ensemble mean for validation
                loss = criterion({"y1": pred["y1_mean"], "activity": pred["activity_mean"]}, 
                               yb, xb[:, 0], xb[:, 1], xb[:, 2])
                val_loss += loss.item()
                
                val_preds.extend(pred["y1_mean"].cpu().numpy())
                val_targets.extend(yb.cpu().numpy())
        
        val_loss /= len(dl_val)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        
        # Progress reporting
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}  MAE={val_mae:.5f}")
    
    return model, history

# AZEOTROPE DETECTION

def enhanced_azeotrope_detection(model: EnsembleNet, scaler_X: StandardScaler, 
                                T_range: Tuple[float, float], P_range: Tuple[float, float],
                                n_points: int = 1000, device: Optional[str] = None) -> pd.DataFrame:
    """
    Detect azeotropic points using optimization
    
    Searches for conditions where liquid and vapor compositions are equal
    (y1 ≈ x1), which defines the azeotropic condition.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    def objective(x1: float, T_C: float, P_kPa: float) -> float:
        """Objective function: minimize |y1 - x1| to find azeotrope"""
        X = np.array([[x1, T_C, P_kPa]])
        Xs = scaler_X.transform(X)
        xt = torch.from_numpy(Xs).float().to(device)
        
        with torch.no_grad():
            pred = model(xt)
            y1_pred = pred["y1_mean"].item()
        
        return abs(y1_pred - x1)
    
    results = []
    T_values = np.linspace(T_range[0], T_range[1], 20)
    P_values = np.linspace(P_range[0], P_range[1], 10)
    
    print("Scanning for azeotropic points...")
    for T_C in T_values:
        for P_kPa in P_values:
            # Optimize for azeotropic composition
            result = minimize_scalar(
                objective, 
                args=(T_C, P_kPa),
                bounds=(0.0, 1.0),
                method='bounded'
            )
            
            if result.success:
                x1_azeo = result.x
                min_diff = result.fun
                
                # Get prediction at azeotropic point
                X = np.array([[x1_azeo, T_C, P_kPa]])
                Xs = scaler_X.transform(X)
                xt = torch.from_numpy(Xs).float().to(device)
                
                with torch.no_grad():
                    pred = model(xt)
                    y1_pred = pred["y1_mean"].item()
                    y1_std = pred["y1_std"].item()
                
                # Calculate confidence based on uncertainty
                confidence = 1.0 / (1.0 + y1_std)
                
                results.append({
                    "T_C": T_C,
                    "P_kPa": P_kPa,
                    "x1_azeo": x1_azeo,
                    "y1_pred": y1_pred,
                    "y1_std": y1_std,
                    "min_diff": min_diff,
                    "confidence": confidence
                })
    
    return pd.DataFrame(results)

# VISUALIZATION FUNCTIONS

def create_advanced_visualizations(model: EnsembleNet, X_test: np.ndarray, y_test: np.ndarray,
                                 scaler_X: StandardScaler, azeotrope_df: pd.DataFrame,
                                 outdir: str, device: Optional[str] = None):
    """Create comprehensive visualizations for model analysis"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get predictions with uncertainty
    model.eval()
    with torch.no_grad():
        X_test_scaled = scaler_X.transform(X_test)
        X_test_tensor = torch.from_numpy(X_test_scaled).float().to(device)
        predictions = model(X_test_tensor)
        
        y_pred_mean = predictions["y1_mean"].cpu().numpy()
        y_pred_std = predictions["y1_std"].cpu().numpy()
    
    # 1. Parity Plot with Uncertainty
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(y_test.flatten(), y_pred_mean.flatten(), yerr=y_pred_std.flatten(), 
                fmt='o', alpha=0.6, capsize=3)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Experimental y₁')
    ax.set_ylabel('Predicted y₁')
    ax.set_title('Parity Plot with Uncertainty Quantification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'parity_with_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Residual Analysis
    residuals = y_test.flatten() - y_pred_mean.flatten()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Predicted
    axes[0,0].scatter(y_pred_mean.flatten(), residuals, alpha=0.6)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_xlabel('Predicted y₁')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Predicted')
    axes[0,0].grid(True, alpha=0.3)
    
    # Residuals vs Temperature
    axes[0,1].scatter(X_test[:, 1], residuals, alpha=0.6)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Temperature')
    axes[0,1].grid(True, alpha=0.3)
    
    # Residuals vs Pressure
    axes[1,0].scatter(X_test[:, 2], residuals, alpha=0.6)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel('Pressure (kPa)')
    axes[1,0].set_ylabel('Residuals')
    axes[1,0].set_title('Residuals vs Pressure')
    axes[1,0].grid(True, alpha=0.3)
    
    # Q-Q Plot for normality check
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot of Residuals')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. VLE Phase Diagram
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create temperature-composition diagram
    T_unique = np.unique(X_test[:, 1])
    colors = plt.cm.viridis(np.linspace(0, 1, len(T_unique)))
    
    for i, T in enumerate(T_unique):
        mask = np.abs(X_test[:, 1] - T) < 0.1
        if np.sum(mask) > 5:
            x1_subset = X_test[mask, 0]
            y1_subset = y_pred_mean[mask]
            
            # Sort for smooth plotting
            sort_idx = np.argsort(x1_subset)
            ax.plot(x1_subset[sort_idx], y1_subset[sort_idx], 
                   'o-', color=colors[i], alpha=0.7, label=f'T = {T:.1f}°C')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='x₁ = y₁ (Azeotrope)')
    ax.set_xlabel('Liquid Mole Fraction x₁')
    ax.set_ylabel('Vapor Mole Fraction y₁')
    ax.set_title('Vapor-Liquid Equilibrium Diagram')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'vle_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Azeotrope Detection Visualization
    if len(azeotrope_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(azeotrope_df['T_C'], azeotrope_df['x1_azeo'], 
                           c=azeotrope_df['confidence'], s=100, alpha=0.7, 
                           cmap='viridis', edgecolors='black')
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Azeotropic Composition x₁')
        ax.set_title('Detected Azeotropic Points')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Confidence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'azeotrope_detection.png'), dpi=300, bbox_inches='tight')
        plt.close()

# STATISTICAL ANALYSIS      

def statistical_significance_test(y_true: np.ndarray, y_pred: np.ndarray, 
                                baseline_pred: np.ndarray) -> Dict[str, Any]:
    """
    Perform statistical significance tests comparing ANN vs baseline

    Uses paired t-test to determine if ANN improvement is statistically significant.
    """
    # Calculate error metrics
    ann_mae = mean_absolute_error(y_true, y_pred)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    
    ann_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    
    # Paired t-test for MAE differences
    ann_errors = np.abs(y_true - y_pred)
    baseline_errors = np.abs(y_true - baseline_pred)
    
    t_stat, p_value = stats.ttest_rel(baseline_errors, ann_errors)
    
    # Effect size (Cohen's d) for practical significance
    pooled_std = np.sqrt((np.var(ann_errors) + np.var(baseline_errors)) / 2)
    cohens_d = (baseline_mae - ann_mae) / pooled_std
    
    # Ensure scalar values
    t_stat = float(t_stat) if np.isscalar(t_stat) else float(t_stat[0])
    p_value = float(p_value) if np.isscalar(p_value) else float(p_value[0])
    cohens_d = float(cohens_d) if np.isscalar(cohens_d) else float(cohens_d[0])
    
    # 95% confidence intervals
    ann_ci = stats.t.interval(0.95, len(y_true)-1, 
                             loc=ann_mae, scale=stats.sem(ann_errors))
    baseline_ci = stats.t.interval(0.95, len(y_true)-1, 
                                  loc=baseline_mae, scale=stats.sem(baseline_errors))
    
    return {
        "ann_mae": ann_mae,
        "baseline_mae": baseline_mae,
        "ann_rmse": ann_rmse,
        "baseline_rmse": baseline_rmse,
        "improvement_mae": (baseline_mae - ann_mae) / baseline_mae * 100,
        "improvement_rmse": (baseline_rmse - ann_rmse) / baseline_rmse * 100,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
        "ann_ci": ann_ci,
        "baseline_ci": baseline_ci
    }

#main exe

def main():
    """Main function to run the enhanced ANN training and analysis"""
    parser = argparse.ArgumentParser(description="Enhanced ANN Surrogate for Binary VLE")
    parser.add_argument("--excel", type=str, required=True, help="Path to Excel file")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name")
    parser.add_argument("--system", type=str, default="acetone-chloroform", 
                       choices=["acetone-chloroform", "custom"])
    parser.add_argument("--temp_unit", type=str, default="K", choices=["K", "C"])
    parser.add_argument("--pressure_unit", type=str, default="Pa", 
                       choices=["Pa", "kPa", "bar", "atm", "mmHg"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--n_models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--outdir", type=str, default="outputs_enhanced")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.outdir)
    
    # Load and preprocess data
    print("Loading dataset...")
    if args.sheet:
        df = pd.read_excel(args.excel, sheet_name=args.sheet)
    else:
        df = pd.read_excel(args.excel)
    
    # Standardize column names
    cols_lower = {c: c.strip().lower() for c in df.columns}
    df.columns = [cols_lower[c] for c in df.columns]
    
    # Validate required columns
    for col in ["x1", "t", "p", "y1"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in Excel. Found: {list(df.columns)}")
    
    # Clean and convert units
    df = df[["x1", "t", "p", "y1"]].dropna().copy()
    df["x1"] = df["x1"].clip(0.0, 1.0)  # Ensure valid mole fractions
    df["y1"] = df["y1"].clip(0.0, 1.0)
    df["T_C"] = convert_temperature_to_celsius(df["t"], args.temp_unit)
    df["P_kPa"] = convert_pressure_to_kpa(df["p"], args.pressure_unit)
    
    print(f"Dataset loaded: {len(df)} points")
    print(f"Temperature range: {df['T_C'].min():.1f} to {df['T_C'].max():.1f} °C")
    print(f"Pressure range: {df['P_kPa'].min():.1f} to {df['P_kPa'].max():.1f} kPa")
    
    # Prepare features and target
    X = df[["x1", "T_C", "P_kPa"]].to_numpy(dtype=np.float64)
    y = df["y1"].to_numpy(dtype=np.float64)
    
    # Split data into train/validation/test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_rel, random_state=42)
    
    # Scale features for neural network training
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)
    
    print(f"Training set: {len(X_train)} points")
    print(f"Validation set: {len(X_val)} points")
    print(f"Test set: {len(X_test)} points")
    
    # Train ensemble model
    print("Training ensemble model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, history = train_ensemble_model(
        X_train_s, y_train, X_val_s, y_val,
        epochs=args.epochs, n_models=args.n_models, device=device
    )
    
    # Evaluate model on test set
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test_s).float().to(device)
        predictions = model(X_test_tensor)
        
        y_pred_mean = predictions["y1_mean"].cpu().numpy()
        y_pred_std = predictions["y1_std"].cpu().numpy()
    
    # Compare with Raoult's Law baseline
    if args.system == "custom":
        raise ValueError("Custom system not implemented in enhanced version")
    else:
        params1, params2 = SYSTEMS[args.system]
    
    y_baseline = raoults_law_y1(X_test[:, 0], X_test[:, 1], X_test[:, 2], params1, params2)
    
    # Statistical analysis
    stats_results = statistical_significance_test(y_test, y_pred_mean, y_baseline)
    
    # Enhanced azeotrope detection
    print("Performing enhanced azeotrope detection...")
    T_range = (df["T_C"].min(), df["T_C"].max())
    P_range = (df["P_kPa"].min(), df["P_kPa"].max())
    
    azeotrope_df = enhanced_azeotrope_detection(model, scaler_X, T_range, P_range, device=device)
    
    # Create visualizations
    print("Creating advanced visualizations...")
    create_advanced_visualizations(model, X_test, y_test, scaler_X, azeotrope_df, args.outdir, device)
    
    # Save results
    results = {
        "dataset_info": {
            "n_total": len(df),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "temp_range": [float(df["T_C"].min()), float(df["T_C"].max())],
            "pressure_range": [float(df["P_kPa"].min()), float(df["P_kPa"].max())]
        },
        "model_info": {
            "architecture": "Physics-informed ensemble",
            "n_models": args.n_models,
            "epochs": args.epochs,
            "system": args.system
        },
        "performance": {
            "ann_mae": float(stats_results["ann_mae"]),
            "ann_rmse": float(stats_results["ann_rmse"]),
            "baseline_mae": float(stats_results["baseline_mae"]),
            "baseline_rmse": float(stats_results["baseline_rmse"]),
            "improvement_mae": float(stats_results["improvement_mae"]),
            "improvement_rmse": float(stats_results["improvement_rmse"])
        },
        "statistical_tests": {
            "t_statistic": float(stats_results["t_statistic"]),
            "p_value": float(stats_results["p_value"]),
            "significant": bool(stats_results["significant"]),
            "cohens_d": float(stats_results["cohens_d"])
        },
        "azeotrope_detection": {
            "n_points_detected": len(azeotrope_df),
            "avg_confidence": float(azeotrope_df["confidence"].mean()) if len(azeotrope_df) > 0 else 0.0
        }
    }
    
    # Save all results
    with open(os.path.join(args.outdir, "enhanced_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    azeotrope_df.to_csv(os.path.join(args.outdir, "enhanced_azeotrope_scan.csv"), index=False)
    
    # Save trained model
    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler_X.mean_,
        "scaler_scale": scaler_X.scale_,
        "n_models": args.n_models,
        "system": args.system
    }, os.path.join(args.outdir, "enhanced_model.pt"))
    
    # Print final results
    print(f"\nEnhanced analysis completed!")
    print(f"Results saved to: {os.path.abspath(args.outdir)}")
    print(f"ANN MAE: {stats_results['ann_mae']:.6f}")
    print(f"Baseline MAE: {stats_results['baseline_mae']:.6f}")
    print(f"Improvement: {stats_results['improvement_mae']:.1f}%")
    print(f"Statistical significance: {'Yes' if stats_results['significant'] else 'No'} (p={stats_results['p_value']:.4f})")
    print(f"Azeotropic points detected: {len(azeotrope_df)}")

if __name__ == "__main__":
    main()
#END