"""
Model Comparison Script

Compares performance of all trained models (baselines, XGBoost, LSTM)
and generates a comprehensive comparison report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_results():
    """Load results from all models."""
    results = []
    
    # Load baseline results
    baseline_path = Path('models/baselines/baseline_comparison.csv')
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        results.append(baseline_df)
        logger.info(f"✅ Loaded baseline results: {len(baseline_df)} models")
    else:
        logger.warning(f"⚠️  Baseline results not found at {baseline_path}")
    
    # Load XGBoost results
    xgboost_path = Path('models/xgboost/xgboost_results.csv')
    if xgboost_path.exists():
        xgboost_df = pd.read_csv(xgboost_path)
        results.append(xgboost_df)
        logger.info(f"✅ Loaded XGBoost results")
    else:
        logger.warning(f"⚠️  XGBoost results not found at {xgboost_path}")
    
    # Combine all results
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        return combined_df
    else:
        logger.error("❌ No model results found")
        return None

def create_comparison_plot(df, output_path='reports/model_comparison.png'):
    """
    Create visualization comparing all models.
    
    Args:
        df (pd.DataFrame): DataFrame with model results
        output_path (str): Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison - Climate Temperature Prediction', fontsize=16, fontweight='bold')
    
    metrics = ['RMSE', 'MAE', 'MSE', 'R2']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        if metric in df.columns:
            # Sort by metric (ascending for errors, descending for R2)
            sorted_df = df.sort_values(by=metric, ascending=(metric != 'R2'))
            
            # Create bar plot
            bars = ax.barh(sorted_df['Model'], sorted_df[metric])
            
            # Color bars (green for best, red for worst)
            if metric == 'R2':
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
            else:
                colors = plt.cm.RdYlGn(np.linspace(0.9, 0.3, len(bars)))
            
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xlabel(metric, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (model, value) in enumerate(zip(sorted_df['Model'], sorted_df[metric])):
                ax.text(value, i, f'  {value:.4f}', va='center')
        else:
            ax.text(0.5, 0.5, f'{metric} not available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ Comparison plot saved to {output_path}")
    
    plt.close()

def generate_report(df, output_path='reports/model_comparison_report.txt'):
    """
    Generate a text report with model comparison.
    
    Args:
        df (pd.DataFrame): DataFrame with model results
        output_path (str): Path to save the report
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT - Climate Temperature Prediction\n")
        f.write("="*80 + "\n\n")
        
        # Sort by RMSE (best model first)
        if 'RMSE' in df.columns:
            df_sorted = df.sort_values(by='RMSE')
        else:
            df_sorted = df
        
        f.write("MODELS RANKED BY RMSE (Best to Worst):\n")
        f.write("-"*80 + "\n\n")
        
        for idx, row in df_sorted.iterrows():
            f.write(f"{idx + 1}. {row['Model']}\n")
            for col in df.columns:
                if col != 'Model':
                    f.write(f"   {col}: {row[col]:.6f}\n")
            f.write("\n")
        
        # Best model summary
        if len(df_sorted) > 0:
            best_model = df_sorted.iloc[0]
            f.write("="*80 + "\n")
            f.write("BEST MODEL SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {best_model['Model']}\n")
            
            if 'RMSE' in best_model:
                f.write(f"RMSE: {best_model['RMSE']:.6f}°C\n")
            if 'MAE' in best_model:
                f.write(f"MAE: {best_model['MAE']:.6f}°C\n")
            if 'R2' in best_model:
                f.write(f"R²: {best_model['R2']:.6f}\n")
            
            f.write("\n")
            f.write("Interpretation:\n")
            if 'RMSE' in best_model:
                f.write(f"- Average prediction error: ±{best_model['RMSE']:.4f}°C\n")
            if 'R2' in best_model:
                variance_explained = best_model['R2'] * 100
                f.write(f"- Variance explained: {variance_explained:.2f}%\n")
    
    logger.info(f"✅ Comparison report saved to {output_path}")

def main():
    """Main function to compare all models."""
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60 + "\n")
    
    # Load results
    df = load_results()
    
    if df is None or len(df) == 0:
        logger.error("❌ No results to compare")
        return
    
    # Display results
    logger.info("\nModel Results:")
    logger.info(df.to_string(index=False))
    
    # Create comparison plot
    logger.info("\nCreating comparison plot...")
    create_comparison_plot(df)
    
    # Generate text report
    logger.info("Generating comparison report...")
    generate_report(df)
    
    # Find and display best model
    if 'RMSE' in df.columns:
        best_model = df.loc[df['RMSE'].idxmin()]
        logger.info("\n" + "="*60)
        logger.info("🏆 BEST MODEL")
        logger.info("="*60)
        logger.info(f"Model: {best_model['Model']}")
        logger.info(f"RMSE: {best_model['RMSE']:.6f}°C")
        if 'MAE' in best_model:
            logger.info(f"MAE: {best_model['MAE']:.6f}°C")
        if 'R2' in best_model:
            logger.info(f"R²: {best_model['R2']:.6f}")
        logger.info("="*60 + "\n")
    
    logger.info("✅ Comparison complete!")

if __name__ == "__main__":
    main()
