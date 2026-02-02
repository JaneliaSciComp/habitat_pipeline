"""Quality assessment and validation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Assess data quality and provide pass/fail decisions.
    
    Uses computed quality metrics to determine if data meets quality standards.
    """
    
    def __init__(
        self,
        snr_threshold: float = 3.0,
        noise_threshold: float = 100.0,
        drift_threshold: float = 50.0,
        correlation_threshold: float = 0.95
    ):
        """
        Initialize quality assessor.
        
        Parameters
        ----------
        snr_threshold : float
            Minimum SNR in dB
        noise_threshold : float
            Maximum noise level
        drift_threshold : float
            Maximum drift
        correlation_threshold : float
            Maximum correlation for detecting bridged channels
        """
        self.snr_threshold = snr_threshold
        self.noise_threshold = noise_threshold
        self.drift_threshold = drift_threshold
        self.correlation_threshold = correlation_threshold
        
        self.results = {}
    
    def assess_snr(self, snr: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Assess SNR quality.
        
        Parameters
        ----------
        snr : np.ndarray
            SNR values for each channel
            
        Returns
        -------
        passed : bool
            Whether assessment passed
        bad_channels : list of int
            Channels failing SNR threshold
        """
        bad_channels = np.where(snr < self.snr_threshold)[0].tolist()
        passed = len(bad_channels) == 0
        
        logger.info(f"SNR assessment: {len(bad_channels)} channels below threshold")
        
        self.results['snr_passed'] = passed
        self.results['snr_bad_channels'] = bad_channels
        
        return passed, bad_channels
    
    def assess_noise(self, noise_level: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Assess noise levels.
        
        Parameters
        ----------
        noise_level : np.ndarray
            Noise level for each channel
            
        Returns
        -------
        passed : bool
            Whether assessment passed
        bad_channels : list of int
            Channels exceeding noise threshold
        """
        bad_channels = np.where(noise_level > self.noise_threshold)[0].tolist()
        passed = len(bad_channels) == 0
        
        logger.info(f"Noise assessment: {len(bad_channels)} channels above threshold")
        
        self.results['noise_passed'] = passed
        self.results['noise_bad_channels'] = bad_channels
        
        return passed, bad_channels
    
    def assess_drift(self, drift: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Assess baseline drift.
        
        Parameters
        ----------
        drift : np.ndarray
            Drift metric for each channel
            
        Returns
        -------
        passed : bool
            Whether assessment passed
        bad_channels : list of int
            Channels exceeding drift threshold
        """
        bad_channels = np.where(drift > self.drift_threshold)[0].tolist()
        passed = len(bad_channels) == 0
        
        logger.info(f"Drift assessment: {len(bad_channels)} channels above threshold")
        
        self.results['drift_passed'] = passed
        self.results['drift_bad_channels'] = bad_channels
        
        return passed, bad_channels
    
    def detect_bridged_channels(
        self,
        correlation_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Detect bridged channels based on high correlation.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Channel correlation matrix
            
        Returns
        -------
        list of tuple
            Pairs of potentially bridged channels
        """
        bridged_pairs = []
        
        # Find pairs with correlation above threshold (excluding diagonal)
        n_channels = correlation_matrix.shape[0]
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                if correlation_matrix[i, j] > self.correlation_threshold:
                    bridged_pairs.append((i, j))
        
        logger.info(f"Detected {len(bridged_pairs)} potentially bridged channel pairs")
        
        self.results['bridged_pairs'] = bridged_pairs
        
        return bridged_pairs
    
    def assess_all(self, metrics: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Perform all quality assessments.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of computed quality metrics
            
        Returns
        -------
        dict
            Assessment results
        """
        logger.info("Performing comprehensive quality assessment")
        
        if 'snr' in metrics:
            self.assess_snr(metrics['snr'])
        
        if 'noise_level' in metrics:
            self.assess_noise(metrics['noise_level'])
        
        if 'drift' in metrics:
            self.assess_drift(metrics['drift'])
        
        if 'correlation_matrix' in metrics:
            self.detect_bridged_channels(metrics['correlation_matrix'])
        
        # Overall assessment
        all_passed = (
            self.results.get('snr_passed', True) and
            self.results.get('noise_passed', True) and
            self.results.get('drift_passed', True) and
            len(self.results.get('bridged_pairs', [])) == 0
        )
        
        self.results['overall_passed'] = all_passed
        
        return self.results
    
    def get_results(self) -> Dict[str, any]:
        """Get assessment results."""
        return self.results
