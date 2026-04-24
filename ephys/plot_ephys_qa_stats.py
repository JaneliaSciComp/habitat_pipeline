#!/usr/bin/env python3
"""
Electrophysiology Quality Assessment Plotting Functions

This module provides comprehensive visualization tools for analyzing firing pattern
quality metrics from Kilosort data. It includes functions to plot metric distributions
and compare pass/fail results from quality filtering.

Usage:
    # Command line usage:
    python plot_ephys_qa_stats.py --animal_id 631 --session_id 20251216
    python plot_ephys_qa_stats.py --animal_id 631 --session_id 20251216 --save_plots --output_dir ./plots
    
    # Import and use functions directly:
    from plot_ephys_qa_stats import plot_firing_pattern_histograms, load_and_analyze_data
    ks_data, metrics, results = load_and_analyze_data("631", "20251216")
    fig = plot_firing_pattern_histograms(metrics, results)
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.kilosort_data_import import load_kilosort_data
from ingestion.data_paths import get_kilosort_path

def plot_firing_pattern_histograms(metrics, results=None, figsize=(15, 5)):
    """
    Plot histogram distributions for firing pattern quality metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with cluster metrics from calculate_firing_pattern_metrics()
    results : dict, optional
        Results from filter_cells_by_firing_patterns() to show pass/fail split
    figsize : tuple, default=(15, 5)
        Figure size (width, height)
    """
    # Extract metric values
    firing_rates = [m['firing_rate'] for m in metrics.values()]
    presence_ratios = [m['presence_ratio'] for m in metrics.values()]
    cv_isis = [m['cv_isi'] for m in metrics.values() if np.isfinite(m['cv_isi'])]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Firing Pattern Quality Metrics Distributions', fontsize=16, fontweight='bold')
    
    # Color scheme
    hist_color = 'skyblue'
    edge_color = 'navy'
    pass_color = 'lightgreen'
    fail_color = 'lightcoral'
    
    # 1. Firing Rate histogram
    ax = axes[0]
    n, bins, patches = ax.hist(firing_rates, bins=30, alpha=0.7, color=hist_color, 
                               edgecolor=edge_color, linewidth=0.5)
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Firing Rate Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_fr = np.mean(firing_rates)
    std_fr = np.std(firing_rates)
    ax.text(0.65, 0.95, f'Mean: {mean_fr:.2f} Hz\nStd: {std_fr:.2f} Hz\nN: {len(firing_rates)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Presence Ratio histogram  
    ax = axes[1]
    n, bins, patches = ax.hist(presence_ratios, bins=30, alpha=0.7, color=hist_color,
                               edgecolor=edge_color, linewidth=0.5)
    ax.set_xlabel('Presence Ratio')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Presence Ratio Distribution')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Add statistics text
    mean_pr = np.mean(presence_ratios)
    std_pr = np.std(presence_ratios)
    ax.text(0.05, 0.95, f'Mean: {mean_pr:.3f}\nStd: {std_pr:.3f}\nN: {len(presence_ratios)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. CV ISI histogram
    ax = axes[2]
    if len(cv_isis) > 0:
        # Remove extreme outliers for better visualization
        cv_isis_clipped = np.array(cv_isis)
        cv_isis_clipped = cv_isis_clipped[cv_isis_clipped < np.percentile(cv_isis_clipped, 95)]
        
        n, bins, patches = ax.hist(cv_isis_clipped, bins=30, alpha=0.7, color=hist_color,
                                   edgecolor=edge_color, linewidth=0.5)
        ax.set_xlabel('CV of ISI')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('CV ISI Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_cv = np.mean(cv_isis)
        std_cv = np.std(cv_isis)
        ax.text(0.65, 0.95, f'Mean: {mean_cv:.3f}\nStd: {std_cv:.3f}\nN: {len(cv_isis)}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No valid CV ISI data', transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('CV ISI Distribution (No Data)')
    
    plt.tight_layout()
    
    # If filter results provided, create a second figure showing pass/fail split
    if results is not None:
        plot_pass_fail_histograms(metrics, results, figsize=figsize)
    
    return fig

def plot_pass_fail_histograms(metrics, results, figsize=(15, 5)):
    """
    Plot histograms showing pass vs fail distributions for each metric.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with cluster metrics from calculate_firing_pattern_metrics()
    results : dict
        Results from filter_cells_by_firing_patterns()
    figsize : tuple, default=(15, 5)
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.Figure : The created figure object
    """
    passed_ids = set(results['passed_clusters'])
    failed_ids = set(results['failed_clusters'].keys())
    
    # Separate metrics by pass/fail status
    pass_metrics = {k: v for k, v in metrics.items() if k in passed_ids}
    fail_metrics = {k: v for k, v in metrics.items() if k in failed_ids}
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Quality Metrics: Pass vs Fail Distributions', fontsize=16, fontweight='bold')
    
    pass_color = 'green'
    fail_color = 'red'
    alpha = 0.6
    
    # 1. Firing Rate
    ax = axes[0]
    if pass_metrics:
        pass_fr = [m['firing_rate'] for m in pass_metrics.values()]
        ax.hist(pass_fr, bins=20, alpha=alpha, color=pass_color, label=f'Pass (n={len(pass_fr)})')
    if fail_metrics:
        fail_fr = [m['firing_rate'] for m in fail_metrics.values()]
        ax.hist(fail_fr, bins=20, alpha=alpha, color=fail_color, label=f'Fail (n={len(fail_fr)})')
    ax.set_xlabel('Firing Rate (Hz)')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Firing Rate: Pass vs Fail')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Presence Ratio
    ax = axes[1]
    if pass_metrics:
        pass_pr = [m['presence_ratio'] for m in pass_metrics.values()]
        ax.hist(pass_pr, bins=20, alpha=alpha, color=pass_color, label=f'Pass (n={len(pass_pr)})')
    if fail_metrics:
        fail_pr = [m['presence_ratio'] for m in fail_metrics.values()]
        ax.hist(fail_pr, bins=20, alpha=alpha, color=fail_color, label=f'Fail (n={len(fail_pr)})')
    ax.set_xlabel('Presence Ratio')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Presence Ratio: Pass vs Fail')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # 3. CV ISI
    ax = axes[2]
    if pass_metrics:
        pass_cv = [m['cv_isi'] for m in pass_metrics.values() if np.isfinite(m['cv_isi'])]
        if len(pass_cv) > 0:
            pass_cv_clipped = np.array(pass_cv)
            pass_cv_clipped = pass_cv_clipped[pass_cv_clipped < np.percentile(pass_cv_clipped, 95)]
            ax.hist(pass_cv_clipped, bins=20, alpha=alpha, color=pass_color, label=f'Pass (n={len(pass_cv)})')
    
    if fail_metrics:
        fail_cv = [m['cv_isi'] for m in fail_metrics.values() if np.isfinite(m['cv_isi'])]
        if len(fail_cv) > 0:
            fail_cv_clipped = np.array(fail_cv)
            fail_cv_clipped = fail_cv_clipped[fail_cv_clipped < np.percentile(fail_cv_clipped, 95)]
            ax.hist(fail_cv_clipped, bins=20, alpha=alpha, color=fail_color, label=f'Fail (n={len(fail_cv)})')
    
    ax.set_xlabel('CV of ISI')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('CV ISI: Pass vs Fail')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def test_threshold_combinations(ks_data):
    """
    Test different threshold combinations and print results.
    
    Parameters:
    -----------
    ks_data : KilosortData
        Loaded Kilosort data object
        
    Returns:
    --------
    dict : Results from moderate threshold settings
    """
    print("Testing different threshold combinations:")
    
    # Define threshold sets
    threshold_sets = [
        {"name": "Very Strict (Default)", "min_firing_rate": 0.5, "min_presence_ratio": 0.8, "max_cv_isi": 2.0},
        {"name": "Moderate", "min_firing_rate": 0.1, "min_presence_ratio": 0.5, "max_cv_isi": 5.0},
        {"name": "Lenient", "min_firing_rate": 0.05, "min_presence_ratio": 0.3, "max_cv_isi": 10.0},
    ]
    
    moderate_results = None
    
    for thresh in threshold_sets:
        name = thresh.pop("name")
        results_test = ks_data.filter_cells_by_firing_patterns(**thresh)
        summary = results_test['summary']
        print(f"\n{name}:")
        print(f"  Passed: {summary['passed_count']}/{summary['total_clusters']} ({summary['pass_rate']:.1%})")
        
        # Store moderate results for return
        if "Moderate" in name:
            moderate_results = results_test
            
        # Add name back for next iteration
        thresh["name"] = name
    
    return moderate_results


def load_and_analyze_data(animal_id, session_id, save_plots=False, output_dir=None):
    """
    Load Kilosort data and generate quality analysis plots.
    
    Parameters:
    -----------
    animal_id : str
        Animal identifier
    session_id : str  
        Session identifier
    save_plots : bool, default=False
        Whether to save plots to files
    output_dir : str, optional
        Directory to save plots (default: current directory)
        
    Returns:
    --------
    tuple : (ks_data, metrics, results)
        - ks_data: Loaded KilosortData object
        - metrics: Calculated firing pattern metrics
        - results: Filtering results with moderate thresholds
    """
    print(f"Loading data for animal {animal_id}, session {session_id}")
    
    # Load data
    try:
        kilosort_path = get_kilosort_path(animal_id, session_id)[0]
        ks_data = load_kilosort_data(kilosort_path)
        print(f"Successfully loaded {len(ks_data.ks_ids)} clusters")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    # Calculate metrics
    print("Calculating firing pattern metrics...")
    metrics = ks_data.calculate_firing_pattern_metrics()
    
    # Test different thresholds
    moderate_results = test_threshold_combinations(ks_data)
    
    print("\n" + "="*60)
    print("MODERATE THRESHOLDS ANALYSIS:")
    ks_data.print_firing_pattern_summary(moderate_results)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Main distribution plots
    fig1 = plot_firing_pattern_histograms(metrics, moderate_results)
    
    if save_plots:
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Save plots
        filename1 = f"firing_metrics_distributions_{animal_id}_{session_id}.png"
        fig1.savefig(output_dir / filename1, dpi=300, bbox_inches='tight')
        print(f"Saved main plot: {filename1}")
        
        # The pass/fail plot is created automatically by plot_firing_pattern_histograms
        # We need to get the current figure to save it
        fig2 = plt.gcf()  # Get current figure (pass/fail plot)
        filename2 = f"firing_metrics_pass_fail_{animal_id}_{session_id}.png"
        fig2.savefig(output_dir / filename2, dpi=300, bbox_inches='tight')
        print(f"Saved pass/fail plot: {filename2}")
    
    plt.show()
    
    return ks_data, metrics, moderate_results


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Generate firing pattern quality analysis plots')
    parser.add_argument('--animal_id', type=str, required=True, help='Animal identifier')
    parser.add_argument('--session_id', type=str, required=True, help='Session identifier')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    parser.add_argument('--output_dir', type=str, help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Load data and generate plots
    ks_data, metrics, results = load_and_analyze_data(
        args.animal_id, 
        args.session_id, 
        save_plots=args.save_plots,
        output_dir=args.output_dir
    )
    
    if ks_data is not None:
        print(f"\nAnalysis complete!")
        print(f"Dataset: {ks_data}")
        print(f"Quality cells with moderate thresholds: {len(results['passed_clusters'])}/{results['summary']['total_clusters']}")
    else:
        print("Analysis failed - could not load data")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())