"""Coordinator for multi-animal experiments."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class MultiAnimalCoordinator:
    """
    Coordinate processing and analysis across multiple animals.
    
    Manages data loading, synchronization, and parallel processing for
    multi-animal recording sessions.
    """
    
    def __init__(self, n_jobs: int = -1):
        """
        Initialize multi-animal coordinator.
        
        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        """
        self.n_jobs = n_jobs
        self.animals = {}
        self.sync_info = {}
    
    def register_animal(
        self,
        animal_id: str,
        data_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an animal in the multi-animal session.
        
        Parameters
        ----------
        animal_id : str
            Unique identifier for the animal
        data_path : str
            Path to animal's data
        metadata : dict, optional
            Additional metadata for the animal
        """
        self.animals[animal_id] = {
            'data_path': Path(data_path),
            'metadata': metadata or {},
            'status': 'registered'
        }
        
        logger.info(f"Registered animal: {animal_id}")
    
    def synchronize_animals(
        self,
        sync_signals: Optional[Dict[str, np.ndarray]] = None,
        reference_animal: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Synchronize timestamps across all animals.
        
        Parameters
        ----------
        sync_signals : dict, optional
            Dictionary of animal_id to sync signals
        reference_animal : str, optional
            ID of reference animal for synchronization
            
        Returns
        -------
        dict
            Synchronization transforms for each animal
        """
        if reference_animal is None:
            reference_animal = list(self.animals.keys())[0]
        
        logger.info(f"Synchronizing animals to reference: {reference_animal}")
        
        from habitat_pipeline.synchronization import TimestampAligner
        
        aligner = TimestampAligner(reference_stream=reference_animal)
        
        # Build streams dictionary (placeholder - would need actual timestamps)
        streams = {}
        for animal_id in self.animals.keys():
            # In real implementation, load actual timestamps
            streams[animal_id] = np.arange(1000)  # Placeholder
        
        # Align streams
        if sync_signals:
            aligned_streams = aligner.align_streams(streams, sync_signals)
        else:
            aligned_streams = aligner.align_streams(streams)
        
        # Store synchronization info
        self.sync_info = {
            'reference': reference_animal,
            'transforms': aligner.transforms,
            'aligned_streams': aligned_streams
        }
        
        return aligner.transforms
    
    def process_animal(
        self,
        animal_id: str,
        processing_func: callable,
        **kwargs
    ) -> Any:
        """
        Process data for a single animal.
        
        Parameters
        ----------
        animal_id : str
            Animal identifier
        processing_func : callable
            Function to apply to animal data
        **kwargs
            Additional arguments for processing function
            
        Returns
        -------
        Any
            Processing results
        """
        logger.info(f"Processing animal: {animal_id}")
        
        animal_info = self.animals[animal_id]
        
        try:
            result = processing_func(
                animal_id=animal_id,
                data_path=animal_info['data_path'],
                metadata=animal_info['metadata'],
                **kwargs
            )
            
            animal_info['status'] = 'processed'
            return result
            
        except Exception as e:
            logger.error(f"Error processing animal {animal_id}: {e}")
            animal_info['status'] = 'error'
            raise
    
    def process_all_animals(
        self,
        processing_func: callable,
        parallel: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process data for all registered animals.
        
        Parameters
        ----------
        processing_func : callable
            Function to apply to each animal's data
        parallel : bool
            Whether to process in parallel
        **kwargs
            Additional arguments for processing function
            
        Returns
        -------
        dict
            Dictionary mapping animal_id to processing results
        """
        logger.info(f"Processing {len(self.animals)} animals")
        
        if parallel and len(self.animals) > 1:
            # Parallel processing
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self.process_animal)(animal_id, processing_func, **kwargs)
                for animal_id in self.animals.keys()
            )
            
            results_dict = dict(zip(self.animals.keys(), results))
        else:
            # Sequential processing
            results_dict = {}
            for animal_id in self.animals.keys():
                results_dict[animal_id] = self.process_animal(
                    animal_id, processing_func, **kwargs
                )
        
        return results_dict
    
    def get_animal_status(self) -> Dict[str, str]:
        """Get processing status for all animals."""
        return {
            animal_id: info['status']
            for animal_id, info in self.animals.items()
        }
    
    def get_sync_info(self) -> Dict[str, Any]:
        """Get synchronization information."""
        return self.sync_info
