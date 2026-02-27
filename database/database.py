"""
Database Module for Habitat Pipeline

Manages information about animals, sessions, and available data types
using SQLite with SQLAlchemy ORM for easy creation, maintenance, and access.
"""

import os
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()


class Animal(Base):
    """Animal model - stores information about each animal subject"""
    __tablename__ = 'animals'
    
    id = Column(Integer, primary_key=True)
    animal_id = Column(String(50), unique=True, nullable=False, index=True)
    species = Column(String(50))
    strain = Column(String(50))
    sex = Column(String(10))
    birth_date = Column(DateTime)
    weight_g = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("ExperimentSession", back_populates="animal")
    
    def __repr__(self):
        return f"<Animal(id={self.animal_id}, species={self.species}, sessions={len(self.sessions)})>"


class ExperimentSession(Base):
    """Experiment session model - stores information about recording sessions"""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), nullable=False, index=True)
    animal_id = Column(String(50), ForeignKey('animals.animal_id'), nullable=False)
    session_date = Column(DateTime, nullable=False)
    experiment_type = Column(String(100))  # e.g., "open_field", "linear_track"
    duration_minutes = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    animal = relationship("Animal", back_populates="sessions")
    data_files = relationship("DataFile", back_populates="session")
    
    def __repr__(self):
        return f"<Session(id={self.session_id}, animal={self.animal_id}, date={self.session_date})>"


class DataFile(Base):
    """Data file model - tracks available data for each session"""
    __tablename__ = 'data_files'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey('sessions.session_id'), nullable=False)
    data_type = Column(String(50), nullable=False, index=True)  # 'ephys', 'video', 'annotation'
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    checksum = Column(String(64))  # For data integrity
    is_processed = Column(Boolean, default=False)
    processing_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("ExperimentSession", back_populates="data_files")
    
    def __repr__(self):
        return f"<DataFile(session={self.session_id}, type={self.data_type}, path={Path(self.file_path).name})>"


class HabitatDatabase:
    """Main database interface for the habitat pipeline"""
    
    def __init__(self, db_path: Union[str, Path] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. If None, creates in current directory.
        """
        if db_path is None:
            db_path = Path.cwd() / "habitat_pipeline.db"
        else:
            db_path = Path(db_path)
            
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database initialized at: {db_path}")
    
    def get_db_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    # Animal operations
    def add_animal(self, animal_id: str, species: str = None, strain: str = None,
                   sex: str = None, birth_date: datetime = None, weight_g: float = None,
                   notes: str = None) -> Animal:
        """Add new animal to database"""
        with self.get_db_session() as session:
            # Check if animal already exists
            existing = session.query(Animal).filter(Animal.animal_id == animal_id).first()
            if existing:
                logger.warning(f"Animal {animal_id} already exists. Use update_animal() to modify.")
                return existing
            
            animal = Animal(
                animal_id=animal_id,
                species=species,
                strain=strain,
                sex=sex,
                birth_date=birth_date,
                weight_g=weight_g,
                notes=notes
            )
            session.add(animal)
            session.commit()
            session.refresh(animal)
            logger.info(f"Added animal: {animal_id}")
            return animal
    
    def get_animal(self, animal_id: str) -> Optional[Animal]:
        """Get animal by ID"""
        with self.get_db_session() as session:
            return session.query(Animal).filter(Animal.animal_id == animal_id).first()
    
    def get_all_animals(self) -> List[Animal]:
        """Get all animals"""
        with self.get_db_session() as session:
            return session.query(Animal).all()
    
    # Session operations
    def add_session(self, session_id: str, animal_id: str, session_date: Union[datetime, date, str],
                    experiment_type: str = None, duration_minutes: float = None,
                    notes: str = None) -> ExperimentSession:
        """Add new experiment session"""
        with self.get_db_session() as session:
            # Parse date if string
            if isinstance(session_date, str):
                try:
                    session_date = datetime.strptime(session_date, "%Y%m%d")
                except ValueError:
                    session_date = datetime.strptime(session_date, "%Y-%m-%d")
            elif isinstance(session_date, date):
                session_date = datetime.combine(session_date, datetime.min.time())
            
            # Check if animal exists
            animal = session.query(Animal).filter(Animal.animal_id == animal_id).first()
            if not animal:
                logger.warning(f"Animal {animal_id} not found. Creating new animal entry.")
                animal = Animal(animal_id=animal_id)
                session.add(animal)
            
            # Check if session already exists
            existing = session.query(ExperimentSession).filter(
                ExperimentSession.session_id == session_id,
                ExperimentSession.animal_id == animal_id
            ).first()
            if existing:
                logger.warning(f"Session {session_id} for animal {animal_id} already exists.")
                return existing
            
            exp_session = ExperimentSession(
                session_id=session_id,
                animal_id=animal_id,
                session_date=session_date,
                experiment_type=experiment_type,
                duration_minutes=duration_minutes,
                notes=notes
            )
            session.add(exp_session)
            session.commit()
            session.refresh(exp_session)
            logger.info(f"Added session: {session_id} for animal {animal_id}")
            return exp_session
    
    def get_session(self, session_id: str, animal_id: str = None) -> Optional[ExperimentSession]:
        """Get session by ID"""
        with self.get_db_session() as session:
            query = session.query(ExperimentSession).filter(ExperimentSession.session_id == session_id)
            if animal_id:
                query = query.filter(ExperimentSession.animal_id == animal_id)
            return query.first()
    
    def get_animal_sessions(self, animal_id: str) -> List[ExperimentSession]:
        """Get all sessions for an animal"""
        with self.get_db_session() as session:
            return session.query(ExperimentSession).filter(
                ExperimentSession.animal_id == animal_id
            ).order_by(ExperimentSession.session_date).all()
    
    # Data file operations
    def add_data_file(self, session_id: str, data_type: str, file_path: Union[str, Path],
                     is_processed: bool = False, processing_notes: str = None) -> DataFile:
        """Add data file to database"""
        file_path = Path(file_path)
        
        with self.get_db_session() as session:
            # Get file info
            file_size = file_path.stat().st_size if file_path.exists() else None
            
            # Check if file already exists
            existing = session.query(DataFile).filter(
                DataFile.session_id == session_id,
                DataFile.data_type == data_type,
                DataFile.file_path == str(file_path)
            ).first()
            
            if existing:
                logger.warning(f"Data file already exists: {file_path}")
                return existing
            
            data_file = DataFile(
                session_id=session_id,
                data_type=data_type,
                file_path=str(file_path),
                file_size_bytes=file_size,
                is_processed=is_processed,
                processing_notes=processing_notes
            )
            session.add(data_file)
            session.commit()
            session.refresh(data_file)
            logger.info(f"Added data file: {data_type} for session {session_id}")
            return data_file
    
    def get_session_data(self, session_id: str) -> Dict[str, List[DataFile]]:
        """Get all data files for a session, grouped by type"""
        with self.get_db_session() as session:
            files = session.query(DataFile).filter(DataFile.session_id == session_id).all()
            
            data_by_type = {}
            for file in files:
                if file.data_type not in data_by_type:
                    data_by_type[file.data_type] = []
                data_by_type[file.data_type].append(file)
            
            return data_by_type
    
    def check_data_availability(self, animal_id: str = None, session_id: str = None) -> pd.DataFrame:
        """Create availability matrix showing what data exists for each session"""
        with self.get_db_session() as session:
            query = """
            SELECT 
                s.animal_id,
                s.session_id,
                s.session_date,
                s.experiment_type,
                df.data_type,
                COUNT(df.id) as file_count,
                SUM(CASE WHEN df.is_processed = 1 THEN 1 ELSE 0 END) as processed_count
            FROM sessions s
            LEFT JOIN data_files df ON s.session_id = df.session_id
            """
            
            params = {}
            conditions = []
            
            if animal_id:
                conditions.append("s.animal_id = :animal_id")
                params['animal_id'] = animal_id
            
            if session_id:
                conditions.append("s.session_id = :session_id")
                params['session_id'] = session_id
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " GROUP BY s.animal_id, s.session_id, df.data_type ORDER BY s.session_date"
            
            df = pd.read_sql(query, self.engine, params=params)
            
            # Pivot to create availability matrix
            if not df.empty:
                pivot_df = df.pivot_table(
                    index=['animal_id', 'session_id', 'session_date', 'experiment_type'],
                    columns='data_type',
                    values='file_count',
                    fill_value=0
                ).reset_index()
                
                return pivot_df
            else:
                return pd.DataFrame()
    
    # Utility methods
    def scan_data_directory(self, base_path: Union[str, Path], auto_add: bool = True) -> Dict:
        """Scan directory structure and optionally auto-populate database"""
        base_path = Path(base_path)
        scan_results = {
            'animals_found': [],
            'sessions_found': [],
            'data_files_found': [],
            'errors': []
        }
        
        try:
            # Look for session directories
            for session_dir in base_path.iterdir():
                if not session_dir.is_dir():
                    continue
                
                # Extract session ID (remove _merged.kilosort suffix if present)
                session_id = session_dir.name.replace('_merged.kilosort', '')
                
                # Look for animal directories within session
                for animal_dir in session_dir.iterdir():
                    if not animal_dir.is_dir():
                        continue
                    
                    animal_id = animal_dir.name
                    
                    # Add to results
                    if animal_id not in scan_results['animals_found']:
                        scan_results['animals_found'].append(animal_id)
                    scan_results['sessions_found'].append((animal_id, session_id))
                    
                    if auto_add:
                        # Add animal if not exists
                        self.add_animal(animal_id)
                        
                        try:
                            # Try to parse date from session_id
                            session_date = datetime.strptime(session_id, "%Y%m%d")
                            self.add_session(session_id, animal_id, session_date)
                        except ValueError:
                            # If date parsing fails, use current date
                            self.add_session(session_id, animal_id, datetime.now())
                    
                    # Look for data files in the animal directory
                    self._scan_session_directory(animal_dir, session_id, scan_results, auto_add)
        
        except Exception as e:
            scan_results['errors'].append(f"Error scanning directory: {e}")
            logger.error(f"Error scanning directory: {e}")
        
        return scan_results
    
    def _scan_session_directory(self, session_dir: Path, session_id: str, 
                               scan_results: Dict, auto_add: bool):
        """Helper method to scan individual session directory"""
        try:
            # Look for kilosort data
            kilosort_dirs = list(session_dir.glob("**/kilosort*"))
            for ks_dir in kilosort_dirs:
                if (ks_dir / "spike_times.npy").exists():
                    scan_results['data_files_found'].append((session_id, 'ephys', str(ks_dir)))
                    if auto_add:
                        self.add_data_file(session_id, 'ephys', ks_dir)
            
            # Look for video files
            video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
            for ext in video_extensions:
                video_files = list(session_dir.glob(f"**/*{ext}"))
                for video_file in video_files:
                    scan_results['data_files_found'].append((session_id, 'video', str(video_file)))
                    if auto_add:
                        self.add_data_file(session_id, 'video', video_file)
            
            # Look for tracking data
            tracking_extensions = ['.csv', '.tsv', '.dlc']
            for ext in tracking_extensions:
                tracking_files = list(session_dir.glob(f"**/*track*{ext}"))
                for tracking_file in tracking_files:
                    scan_results['data_files_found'].append((session_id, 'tracking', str(tracking_file)))
                    if auto_add:
                        self.add_data_file(session_id, 'tracking', tracking_file)
        
        except Exception as e:
            scan_results['errors'].append(f"Error scanning session {session_id}: {e}")
    
    def export_summary(self, output_path: Union[str, Path] = None) -> pd.DataFrame:
        """Export database summary to CSV"""
        availability_df = self.check_data_availability()
        
        if output_path:
            availability_df.to_csv(output_path, index=False)
            logger.info(f"Database summary exported to: {output_path}")
        
        return availability_df
    
    def __repr__(self):
        with self.get_db_session() as session:
            n_animals = session.query(Animal).count()
            n_sessions = session.query(ExperimentSession).count()
            n_files = session.query(DataFile).count()
        
        return f"<HabitatDatabase(animals={n_animals}, sessions={n_sessions}, files={n_files})>"


# Convenience function
def create_database(db_path: Union[str, Path] = None) -> HabitatDatabase:
    """Create and return a new HabitatDatabase instance"""
    return HabitatDatabase(db_path)


if __name__ == "__main__":
    # Example usage
    print("Creating habitat database...")
    
    # Create database
    db = create_database("example_habitat.db")
    
    # Add some example data
    animal = db.add_animal("613", species="mouse", strain="C57BL/6", sex="M")
    session = db.add_session("20251210", "613", "2025-12-10", experiment_type="open_field")
    
    # Check availability
    availability = db.check_data_availability()
    print("\nData availability:")
    print(availability)
    
    print(f"\nDatabase info: {db}")