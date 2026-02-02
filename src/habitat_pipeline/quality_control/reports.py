"""Quality report generation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import json

logger = logging.getLogger(__name__)


class QualityReport:
    """
    Generate quality control reports.
    
    Creates human-readable and machine-readable reports of quality assessments.
    """
    
    def __init__(self):
        """Initialize quality report generator."""
        self.report_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'assessment': {},
            'summary': {}
        }
    
    def add_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Add quality metrics to report.
        
        Parameters
        ----------
        metrics : dict
            Quality metrics dictionary
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'tolist'):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        self.report_data['metrics'] = serializable_metrics
    
    def add_assessment(self, assessment: Dict[str, Any]) -> None:
        """
        Add quality assessment results to report.
        
        Parameters
        ----------
        assessment : dict
            Assessment results dictionary
        """
        self.report_data['assessment'] = assessment
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of quality assessment.
        
        Returns
        -------
        dict
            Summary dictionary
        """
        assessment = self.report_data.get('assessment', {})
        
        summary = {
            'overall_quality': 'PASS' if assessment.get('overall_passed', False) else 'FAIL',
            'snr_status': 'PASS' if assessment.get('snr_passed', False) else 'FAIL',
            'noise_status': 'PASS' if assessment.get('noise_passed', False) else 'FAIL',
            'drift_status': 'PASS' if assessment.get('drift_passed', False) else 'FAIL',
            'num_snr_bad_channels': len(assessment.get('snr_bad_channels', [])),
            'num_noise_bad_channels': len(assessment.get('noise_bad_channels', [])),
            'num_drift_bad_channels': len(assessment.get('drift_bad_channels', [])),
            'num_bridged_pairs': len(assessment.get('bridged_pairs', [])),
        }
        
        self.report_data['summary'] = summary
        return summary
    
    def save_json(self, output_path: str) -> None:
        """
        Save report as JSON file.
        
        Parameters
        ----------
        output_path : str
            Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        logger.info(f"Saved quality report to {output_path}")
    
    def save_text(self, output_path: str) -> None:
        """
        Save report as human-readable text file.
        
        Parameters
        ----------
        output_path : str
            Path to output text file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("QUALITY CONTROL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {self.report_data['timestamp']}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = self.report_data.get('summary', {})
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Assessment details
            f.write("ASSESSMENT DETAILS\n")
            f.write("-" * 80 + "\n")
            assessment = self.report_data.get('assessment', {})
            
            if 'snr_bad_channels' in assessment and assessment['snr_bad_channels']:
                f.write(f"SNR bad channels: {assessment['snr_bad_channels']}\n")
            
            if 'noise_bad_channels' in assessment and assessment['noise_bad_channels']:
                f.write(f"Noise bad channels: {assessment['noise_bad_channels']}\n")
            
            if 'drift_bad_channels' in assessment and assessment['drift_bad_channels']:
                f.write(f"Drift bad channels: {assessment['drift_bad_channels']}\n")
            
            if 'bridged_pairs' in assessment and assessment['bridged_pairs']:
                f.write(f"Bridged channel pairs: {assessment['bridged_pairs']}\n")
            
            f.write("\n")
        
        logger.info(f"Saved quality report to {output_path}")
    
    def get_report_data(self) -> Dict[str, Any]:
        """Get report data dictionary."""
        return self.report_data
