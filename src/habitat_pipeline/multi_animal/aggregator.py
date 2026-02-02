"""Result aggregation across animals."""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Aggregate analysis results across multiple animals."""
    
    def __init__(self):
        """Initialize result aggregator."""
        self.results = {}
    
    def add_result(self, animal_id: str, result: Any) -> None:
        """
        Add result for an animal.
        
        Parameters
        ----------
        animal_id : str
            Animal identifier
        result : Any
            Analysis result
        """
        self.results[animal_id] = result
        logger.info(f"Added result for animal: {animal_id}")
    
    def aggregate_metrics(
        self,
        metric_name: str,
        aggregation: str = 'mean'
    ) -> float:
        """
        Aggregate a specific metric across animals.
        
        Parameters
        ----------
        metric_name : str
            Name of metric to aggregate
        aggregation : str
            Aggregation method ('mean', 'median', 'sum', 'min', 'max')
            
        Returns
        -------
        float
            Aggregated metric value
        """
        values = []
        
        for animal_id, result in self.results.items():
            if isinstance(result, dict) and metric_name in result:
                values.append(result[metric_name])
            elif hasattr(result, metric_name):
                values.append(getattr(result, metric_name))
        
        if not values:
            logger.warning(f"No values found for metric: {metric_name}")
            return np.nan
        
        values = np.array(values)
        
        if aggregation == 'mean':
            return np.mean(values)
        elif aggregation == 'median':
            return np.median(values)
        elif aggregation == 'sum':
            return np.sum(values)
        elif aggregation == 'min':
            return np.min(values)
        elif aggregation == 'max':
            return np.max(values)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def compute_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all metrics.
        
        Returns
        -------
        dict
            Dictionary of metric names to statistics
        """
        logger.info("Computing summary statistics")
        
        # Find all metric names
        metric_names = set()
        for result in self.results.values():
            if isinstance(result, dict):
                metric_names.update(result.keys())
        
        # Compute statistics for each metric
        summary = {}
        for metric_name in metric_names:
            summary[metric_name] = {
                'mean': self.aggregate_metrics(metric_name, 'mean'),
                'median': self.aggregate_metrics(metric_name, 'median'),
                'min': self.aggregate_metrics(metric_name, 'min'),
                'max': self.aggregate_metrics(metric_name, 'max'),
            }
        
        return summary
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results."""
        return self.results
