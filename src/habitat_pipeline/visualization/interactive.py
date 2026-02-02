"""Interactive visualization utilities."""

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

logger = logging.getLogger(__name__)


class InteractivePlotter:
    """
    Create interactive plots for data exploration.
    
    Allows user interaction with plots for zooming, panning, and parameter adjustment.
    """
    
    def __init__(self):
        """Initialize interactive plotter."""
        self.fig = None
        self.axes = []
    
    def plot_traces_interactive(
        self,
        data: np.ndarray,
        sampling_rate: float
    ) -> plt.Figure:
        """
        Create interactive trace plot with time and channel navigation.
        
        Parameters
        ----------
        data : np.ndarray
            Data array (n_channels, n_samples)
        sampling_rate : float
            Sampling rate in Hz
            
        Returns
        -------
        plt.Figure
            Matplotlib figure with interactive controls
        """
        # Create figure and axes
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main plot
        ax_main = plt.subplot2grid((6, 1), (0, 0), rowspan=5)
        
        # Sliders
        ax_time = plt.subplot2grid((6, 1), (5, 0))
        
        # Initial parameters
        window_duration = 1.0  # seconds
        n_channels_display = min(16, data.shape[0])
        
        # Time axis
        time = np.arange(data.shape[1]) / sampling_rate
        
        # Initial plot
        start_time = 0
        end_time = window_duration
        
        def update_plot(start_time, end_time, start_channel=0):
            """Update the plot with new time window and channels."""
            ax_main.clear()
            
            # Select time window
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            
            time_window = time[start_sample:end_sample]
            data_window = data[start_channel:start_channel + n_channels_display, start_sample:end_sample]
            
            # Plot with offset
            offset = np.max(np.abs(data_window)) * 2
            for i in range(data_window.shape[0]):
                ax_main.plot(
                    time_window,
                    data_window[i] + i * offset,
                    linewidth=0.5,
                    label=f'Ch {start_channel + i}'
                )
            
            ax_main.set_xlabel('Time (s)')
            ax_main.set_ylabel('Channel')
            ax_main.set_title(f'Interactive Trace View ({start_time:.2f}-{end_time:.2f}s)')
            ax_main.set_yticks(np.arange(data_window.shape[0]) * offset)
            ax_main.set_yticklabels([f'Ch {start_channel + i}' for i in range(data_window.shape[0])])
            
            self.fig.canvas.draw_idle()
        
        # Time slider
        slider_time = Slider(
            ax_time,
            'Time',
            0,
            time[-1] - window_duration,
            valinit=start_time,
            valstep=window_duration / 10
        )
        
        def update_time(val):
            """Update plot when slider changes."""
            start = slider_time.val
            end = start + window_duration
            update_plot(start, end)
        
        slider_time.on_changed(update_time)
        
        # Initial plot
        update_plot(start_time, end_time)
        
        plt.tight_layout()
        
        logger.info("Created interactive trace plot")
        
        return self.fig
    
    def show(self):
        """Show interactive plot."""
        plt.show()
