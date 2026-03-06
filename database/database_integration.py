"""
Database Integration Helpers

Functions to integrate the database with existing pipeline components.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .database import HabitatDatabase, ExperimentSession, DataFile
from ingestion.kilosort_data_import import KilosortData
from video.tracking_import import load_tracking_data, parse_tracking


class PipelineIntegration:
    """Integration between database and analysis pipeline"""
    
    def __init__(self, db_path: Optional[str] = None, verbose: bool = True):
        """Initialize with database connection"""
        self.db = HabitatDatabase(db_path, verbose=verbose)
    
    def register_kilosort_data(self, kilosort_path: Path, animal_id: str, session_id: str) -> bool:
        """Register Kilosort data in database"""
        try:
            # Ensure animal and session exist
            self.db.add_animal(animal_id)
            self.db.add_session(session_id, animal_id, session_id)  # Use session_id as date
            
            # Add ephys data file
            self.db.add_data_file(session_id, 'ephys', kilosort_path)
            
            # Load and validate data
            ks_data = KilosortData(kilosort_path)
            
            # Update session with metadata
            with self.db.get_db_session() as session:
                db_session = session.query(ExperimentSession).filter(
                    ExperimentSession.session_id == session_id,
                    ExperimentSession.animal_id == animal_id
                ).first()
                
                if db_session:
                    db_session.duration_minutes = ks_data.metadata.get('duration', 0) / 60
                    db_session.notes = f"Clusters: {ks_data.metadata.get('n_clusters', 0)}, Spikes: {ks_data.metadata.get('n_spikes', 0)}"
                    session.commit()
            
            return True
            
        except Exception as e:
            print(f"Error registering Kilosort data: {e}")
            return False
    
    def register_tracking_data(self, tracking_path: Path, session_id: str) -> bool:
        """Register tracking data in database"""
        try:
            # Add tracking data file
            self.db.add_data_file(session_id, 'tracking', tracking_path)
            
            # Load and validate data
            tracking_df = load_tracking_data(tracking_path)
            animals = parse_tracking(tracking_df)
            
            # Update processing notes
            with self.db.get_db_session() as session:
                data_file = session.query(DataFile).filter(
                    DataFile.session_id == session_id,
                    DataFile.data_type == 'tracking',
                    DataFile.file_path == str(tracking_path)
                ).first()
                
                if data_file:
                    data_file.is_processed = True
                    data_file.processing_notes = f"Animals tracked: {list(animals.keys())}, Frames: {len(tracking_df)}"
                    session.commit()
            
            return True
            
        except Exception as e:
            print(f"Error registering tracking data: {e}")
            return False
    
    def get_available_sessions(self, animal_id: str = None, data_types: List[str] = None) -> pd.DataFrame:
        """Get sessions with specified data types available"""
        availability = self.db.check_data_availability(animal_id=animal_id)
        
        if data_types and not availability.empty:
            # Filter sessions that have all required data types
            required_cols = [col for col in data_types if col in availability.columns]
            if required_cols:
                mask = availability[required_cols].sum(axis=1) == len(required_cols)
                availability = availability[mask]
        
        return availability
    
    def load_session_data(self, animal_id: str, session_id: str) -> Dict:
        """Load all available data for a session"""
        session_data = {}
        
        # Get data files for session
        data_files = self.db.get_session_data(session_id)
        
        # Load ephys data if available
        if 'ephys' in data_files:
            ephys_file = data_files['ephys'][0]  # Take first file
            try:
                session_data['ephys'] = KilosortData(ephys_file.file_path)
            except Exception as e:
                print(f"Error loading ephys data: {e}")
                session_data['ephys'] = None
        
        # Load tracking data if available
        if 'tracking' in data_files:
            tracking_file = data_files['tracking'][0]  # Take first file
            try:
                tracking_df = load_tracking_data(tracking_file.file_path)
                session_data['tracking'] = {
                    'raw_df': tracking_df,
                    'animals': parse_tracking(tracking_df)
                }
            except Exception as e:
                print(f"Error loading tracking data: {e}")
                session_data['tracking'] = None
        
        return session_data
    
    def create_analysis_summary(self, animal_id: str = None) -> pd.DataFrame:
        """Create comprehensive analysis summary"""
        sessions_df = self.get_available_sessions(animal_id)
        
        if sessions_df.empty:
            return pd.DataFrame()
        
        # Add analysis readiness flags
        sessions_df['has_ephys'] = sessions_df.get('ephys', 0) > 0
        sessions_df['has_tracking'] = sessions_df.get('tracking', 0) > 0
        sessions_df['ready_for_place_analysis'] = sessions_df['has_ephys'] & sessions_df['has_tracking']
        
        return sessions_df
    
    def batch_process_directory(self, base_path: Path) -> Dict:
        """Process entire directory structure and register all data"""
        results = {
            'processed_sessions': [],
            'errors': [],
            'summary': {}
        }
        
        # First scan to populate database
        scan_results = self.db.scan_data_directory(base_path, auto_add=True)
        
        # Then process each session
        availability = self.db.check_data_availability()
        
        for _, row in availability.iterrows():
            animal_id = row['animal_id']
            session_id = row['session_id']
            
            try:
                # Check data file existence without loading
                has_ephys = self._check_ephys_files_exist(session_id)
                has_tracking = self._check_tracking_files_exist(session_id)
                
                results['processed_sessions'].append({
                    'animal_id': animal_id,
                    'session_id': session_id,
                    'has_ephys': has_ephys,
                    'has_tracking': has_tracking
                })
                
            except Exception as e:
                results['errors'].append(f"Error processing {animal_id}/{session_id}: {e}")
        
        # Create summary
        processed_df = pd.DataFrame(results['processed_sessions'])
        if not processed_df.empty:
            results['summary'] = {
                'total_sessions': len(processed_df),
                'sessions_with_ephys': processed_df['has_ephys'].sum(),
                'sessions_with_tracking': processed_df['has_tracking'].sum(),
                'complete_sessions': (processed_df['has_ephys'] & processed_df['has_tracking']).sum()
            }
        
        return results
    
    def _check_ephys_files_exist(self, session_id: str) -> bool:
        """Check if ephys files exist on disk without loading KilosortData"""
        try:
            data_files = self.db.get_session_data(session_id)
            
            if 'ephys' not in data_files:
                return False
            
            for ephys_file in data_files['ephys']:
                ephys_path = Path(ephys_file.file_path)
                
                # Check if the directory exists
                if not ephys_path.exists():
                    continue
                
                # Check for essential kilosort files
                spike_times_file = ephys_path / "spike_times.npy"
                spike_clusters_file = ephys_path / "spike_clusters.npy"
                
                if spike_times_file.exists() and spike_clusters_file.exists():
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _check_tracking_files_exist(self, session_id: str) -> bool:
        """Check if tracking files exist on disk"""
        try:
            data_files = self.db.get_session_data(session_id)
            
            if 'tracking' not in data_files:
                return False
            
            for tracking_file in data_files['tracking']:
                tracking_path = Path(tracking_file.file_path)
                
                if tracking_path.exists():
                    return True
            
            return False
            
        except Exception as e:
            return False


def quick_setup(data_directory: Path, db_path: Optional[str] = None, verbose: bool = True) -> PipelineIntegration:
    """Quick setup: scan directory and create database"""
    integration = PipelineIntegration(db_path, verbose=verbose)
    
    print(f"Scanning and setting up database from: {data_directory}")
    results = integration.batch_process_directory(data_directory)
    
    print(f"Setup complete:")
    print(f"  Sessions processed: {len(results['processed_sessions'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results['summary']:
        summary = results['summary']
        print(f"  Total sessions: {summary['total_sessions']}")
        print(f"  With ephys: {summary['sessions_with_ephys']}")
        print(f"  With tracking: {summary['sessions_with_tracking']}")
        print(f"  Complete sessions: {summary['complete_sessions']}")
    
    return integration


# Convenience functions for notebook use
def get_database(db_path: Optional[str] = None) -> HabitatDatabase:
    """Get database instance"""
    return HabitatDatabase(db_path)


def get_session_list(db_path: Optional[str] = None, animal_id: Optional[str] = None) -> pd.DataFrame:
    """Get list of available sessions"""
    db = HabitatDatabase(db_path)
    return db.check_data_availability(animal_id=animal_id)


def load_session(animal_id: str, session_id: str, db_path: Optional[str] = None) -> Dict:
    """Convenience function to load session data"""
    integration = PipelineIntegration(db_path)
    return integration.load_session_data(animal_id, session_id)