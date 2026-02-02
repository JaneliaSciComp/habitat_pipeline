"""Basic plotting utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Plotter:
    """
    Create publication-quality plots for electrophysiology data.
    
    Provides methods for plotting raw data, spikes, and analysis results.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize plotter.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory for saving plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_traces(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channels: Optional[List[int]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        title: str = "Electrophysiology Traces"
    ) -> plt.Figure:
        """
        Plot raw electrophysiology traces.
        
        Parameters
        ----------
        data : np.ndarray
            Data array (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
        channels : list of int, optional
            Specific channels to plot
        time_range : tuple, optional
            Time range to plot (start, end) in seconds
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select channels
        if channels is None:
            channels = list(range(min(16, data.shape[0])))  # Plot max 16 channels
        
        # Select time range
        time = np.arange(data.shape[1]) / sampling_rate
        if time_range:
            mask = (time >= time_range[0]) & (time <= time_range[1])
            time = time[mask]
            plot_data = data[:, mask]
        else:
            plot_data = data
        
        # Plot traces with offset
        offset = np.max(np.abs(plot_data)) * 2
        for i, ch in enumerate(channels):
            ax.plot(time, plot_data[ch] + i * offset, linewidth=0.5, label=f'Ch {ch}')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        ax.set_title(title)
        ax.set_yticks(np.arange(len(channels)) * offset)
        ax.set_yticklabels([f'Ch {ch}' for ch in channels])
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'traces.png', dpi=300)
            logger.info(f"Saved traces plot to {self.output_dir / 'traces.png'}")
        
        return fig
    
    def plot_raster(
        self,
        spike_times: Dict[int, np.ndarray],
        sampling_rate: float,
        time_range: Optional[Tuple[float, float]] = None,
        title: str = "Spike Raster Plot"
    ) -> plt.Figure:
        """
        Plot spike raster.
        
        Parameters
        ----------
        spike_times : dict
            Dictionary mapping channel/unit to spike times (in samples)
        sampling_rate : float
            Sampling rate in Hz
        time_range : tuple, optional
            Time range to plot (start, end) in seconds
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (unit_id, times) in enumerate(spike_times.items()):
            # Convert to seconds
            times_sec = times / sampling_rate
            
            # Filter by time range
            if time_range:
                mask = (times_sec >= time_range[0]) & (times_sec <= time_range[1])
                times_sec = times_sec[mask]
            
            # Plot raster
            ax.scatter(times_sec, np.ones_like(times_sec) * i, marker='|', s=50, c='black')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Unit ID')
        ax.set_title(title)
        ax.set_yticks(range(len(spike_times)))
        ax.set_yticklabels(list(spike_times.keys()))
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'raster.png', dpi=300)
            logger.info(f"Saved raster plot to {self.output_dir / 'raster.png'}")
        
        return fig
    
    def plot_waveforms(
        self,
        waveforms: np.ndarray,
        sampling_rate: float,
        title: str = "Spike Waveforms"
    ) -> plt.Figure:
        """
        Plot spike waveforms.
        
        Parameters
        ----------
        waveforms : np.ndarray
            Waveforms array (n_spikes, n_samples)
        sampling_rate : float
            Sampling rate in Hz
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Time axis
        time_ms = np.arange(waveforms.shape[1]) / sampling_rate * 1000
        
        # Plot individual waveforms
        for waveform in waveforms:
            ax.plot(time_ms, waveform, 'k', alpha=0.1, linewidth=0.5)
        
        # Plot mean waveform
        mean_waveform = np.mean(waveforms, axis=0)
        ax.plot(time_ms, mean_waveform, 'r', linewidth=2, label='Mean')
        
        # Plot std
        std_waveform = np.std(waveforms, axis=0)
        ax.fill_between(
            time_ms,
            mean_waveform - std_waveform,
            mean_waveform + std_waveform,
            alpha=0.3,
            color='r',
            label='±1 SD'
        )
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'waveforms.png', dpi=300)
            logger.info(f"Saved waveforms plot to {self.output_dir / 'waveforms.png'}")
        
        return fig
    
    def plot_psd(
        self,
        data: np.ndarray,
        sampling_rate: float,
        channels: Optional[List[int]] = None,
        title: str = "Power Spectral Density"
    ) -> plt.Figure:
        """
        Plot power spectral density.
        
        Parameters
        ----------
        data : np.ndarray
            Data array (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
        channels : list of int, optional
            Specific channels to plot
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        from scipy.signal import welch
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select channels
        if channels is None:
            channels = list(range(min(8, data.shape[0])))
        
        for ch in channels:
            freqs, psd = welch(data[ch], fs=sampling_rate, nperseg=1024)
            ax.semilogy(freqs, psd, alpha=0.7, label=f'Ch {ch}')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if self.output_dir:
            fig.savefig(self.output_dir / 'psd.png', dpi=300)
            logger.info(f"Saved PSD plot to {self.output_dir / 'psd.png'}")
        
        return fig
