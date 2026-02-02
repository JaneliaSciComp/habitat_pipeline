"""Quality control specific plots."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class QualityPlotter:
    """Create quality control visualization plots."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize quality plotter.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory for saving plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_quality_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        title: str = "Quality Metrics"
    ) -> plt.Figure:
        """
        Plot quality metrics overview.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of quality metrics
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title)
        
        # SNR plot
        if 'snr' in metrics:
            ax = axes[0, 0]
            ax.bar(range(len(metrics['snr'])), metrics['snr'])
            ax.set_xlabel('Channel')
            ax.set_ylabel('SNR (dB)')
            ax.set_title('Signal-to-Noise Ratio')
            ax.axhline(y=3, color='r', linestyle='--', label='Threshold')
            ax.legend()
        
        # Noise level plot
        if 'noise_level' in metrics:
            ax = axes[0, 1]
            ax.bar(range(len(metrics['noise_level'])), metrics['noise_level'])
            ax.set_xlabel('Channel')
            ax.set_ylabel('Noise Level')
            ax.set_title('Channel Noise Levels')
        
        # Drift plot
        if 'drift' in metrics:
            ax = axes[1, 0]
            ax.bar(range(len(metrics['drift'])), metrics['drift'])
            ax.set_xlabel('Channel')
            ax.set_ylabel('Drift')
            ax.set_title('Baseline Drift')
        
        # Correlation matrix
        if 'correlation_matrix' in metrics:
            ax = axes[1, 1]
            im = ax.imshow(metrics['correlation_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')
            ax.set_title('Channel Correlation Matrix')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'quality_metrics.png', dpi=300)
            logger.info(f"Saved quality metrics plot")
        
        return fig
    
    def plot_assessment_summary(
        self,
        assessment: Dict[str, any],
        title: str = "Quality Assessment Summary"
    ) -> plt.Figure:
        """
        Plot quality assessment summary.
        
        Parameters
        ----------
        assessment : dict
            Assessment results
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect pass/fail status
        checks = ['SNR', 'Noise', 'Drift']
        status = [
            'PASS' if assessment.get('snr_passed', False) else 'FAIL',
            'PASS' if assessment.get('noise_passed', False) else 'FAIL',
            'PASS' if assessment.get('drift_passed', False) else 'FAIL',
        ]
        colors = ['green' if s == 'PASS' else 'red' for s in status]
        
        # Bar plot
        ax.barh(checks, [1] * len(checks), color=colors, alpha=0.7)
        
        # Add text
        for i, (check, stat) in enumerate(zip(checks, status)):
            ax.text(0.5, i, stat, ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('')
        ax.set_title(title)
        ax.set_xticks([])
        
        # Overall status
        overall = 'PASS' if assessment.get('overall_passed', False) else 'FAIL'
        overall_color = 'green' if overall == 'PASS' else 'red'
        fig.text(0.5, 0.02, f'Overall: {overall}', ha='center', fontsize=16,
                fontweight='bold', color=overall_color)
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'quality_assessment.png', dpi=300)
            logger.info(f"Saved quality assessment plot")
        
        return fig
