"""Parallel processing utilities."""

import logging
from typing import Any, Callable, List

from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Execute processing tasks in parallel."""
    
    def __init__(self, n_jobs: int = -1, backend: str = 'loky'):
        """
        Initialize parallel processor.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        backend : str
            Joblib backend ('loky', 'multiprocessing', or 'threading')
        """
        self.n_jobs = n_jobs
        self.backend = backend
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Map function over items in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply to each item
        items : list
            List of items to process
        **kwargs
            Additional arguments for function
            
        Returns
        -------
        list
            Results for each item
        """
        logger.info(f"Processing {len(items)} items in parallel with {self.n_jobs} jobs")
        
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(func)(item, **kwargs) for item in items
        )
        
        return results
    
    def starmap(
        self,
        func: Callable,
        items: List[tuple]
    ) -> List[Any]:
        """
        Map function over items with multiple arguments.
        
        Parameters
        ----------
        func : callable
            Function to apply
        items : list of tuple
            List of argument tuples
            
        Returns
        -------
        list
            Results for each item
        """
        logger.info(f"Processing {len(items)} items in parallel")
        
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(func)(*args) for args in items
        )
        
        return results
