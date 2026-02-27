# Habitat Pipeline Database System

A comprehensive database solution for managing animals, experimental sessions, and data files in the habitat pipeline. Built with SQLite and SQLAlchemy for easy creation, maintenance, and access.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Database Schema](#database-schema)
- [Python API](#python-api)
- [Command Line Interface](#command-line-interface)
- [Integration Examples](#integration-examples)
- [Common Workflows](#common-workflows)
- [Advanced Usage](#advanced-usage)

## 🎯 Overview

The database system tracks three main entities:

- **Animals**: Subject information (ID, species, strain, sex, etc.)
- **Sessions**: Experimental recordings (date, type, duration, notes)
- **Data Files**: Available data types (ephys, video, tracking, annotations)

### Key Features

- **Easy Setup**: Single SQLite file, no server required
- **Auto-Discovery**: Scans directory structure to populate database
- **Data Validation**: Validates ephys and tracking data during registration
- **Analysis Readiness**: Identifies complete datasets for analysis
- **CLI Tools**: Command-line interface for database management
- **Pipeline Integration**: Seamlessly works with existing analysis modules

## 🚀 Installation

### Prerequisites

```bash
pip install sqlalchemy pandas numpy
```

### Files Required

- `database.py` - Core database models and operations
- `database_cli.py` - Command line interface
- `database_integration.py` - Pipeline integration helpers

## ⚡ Quick Start

### 1. Create and Setup Database

```python
from database.database_integration import quick_setup
from pathlib import Path

# One-command setup: scan directory and create database
integration = quick_setup(Path("c:/path/to/your/data"))
```

### 2. Basic Database Operations

```python
from database.database import HabitatDatabase

# Create database
db = HabitatDatabase("my_habitat.db")

# Add animal
animal = db.add_animal("613", species="mouse", strain="C57BL/6", sex="M")

# Add session
session = db.add_session("20251210", "613", "2025-12-10", 
                        experiment_type="open_field")

# Check data availability
availability = db.check_data_availability()
print(availability)
```

### 3. Load Session Data

```python
from database.database_integration import load_session

# Load all data for a session
data = load_session("613", "20251210")

# Access ephys data (KilosortData object)
ephys = data['ephys']
print(f"Clusters: {ephys.metadata['n_clusters']}")

# Access tracking data (your existing format)
animals = data['tracking']['animals']
print(f"Tracked animals: {list(animals.keys())}")
```

## 🗃️ Database Schema

### Animals Table
```sql
CREATE TABLE animals (
    id INTEGER PRIMARY KEY,
    animal_id VARCHAR(50) UNIQUE NOT NULL,
    species VARCHAR(50),
    strain VARCHAR(50), 
    sex VARCHAR(10),
    birth_date DATETIME,
    weight_g FLOAT,
    notes TEXT,
    created_at DATETIME,
    updated_at DATETIME
);
```

### Sessions Table
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    animal_id VARCHAR(50) REFERENCES animals(animal_id),
    session_date DATETIME NOT NULL,
    experiment_type VARCHAR(100),
    duration_minutes FLOAT,
    notes TEXT,
    created_at DATETIME,
    updated_at DATETIME
);
```

### Data Files Table
```sql
CREATE TABLE data_files (
    id INTEGER PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES sessions(session_id),
    data_type VARCHAR(50) NOT NULL,  -- 'ephys', 'video', 'tracking'
    file_path VARCHAR(500) NOT NULL,
    file_size_bytes INTEGER,
    checksum VARCHAR(64),
    is_processed BOOLEAN DEFAULT FALSE,
    processing_notes TEXT,
    created_at DATETIME,
    updated_at DATETIME
);
```

## 🐍 Python API

### Core Database Class

```python
from database.database import HabitatDatabase

db = HabitatDatabase("habitat.db")
```

#### Animal Operations

```python
# Add animal
animal = db.add_animal(
    animal_id="613",
    species="mouse", 
    strain="C57BL/6",
    sex="M",
    birth_date=datetime(2025, 1, 15),
    weight_g=25.5,
    notes="Control group"
)

# Get animal
animal = db.get_animal("613")

# Get all animals
animals = db.get_all_animals()
```

#### Session Operations

```python
# Add session
session = db.add_session(
    session_id="20251210",
    animal_id="613", 
    session_date="2025-12-10",
    experiment_type="open_field",
    duration_minutes=60,
    notes="Baseline recording"
)

# Get sessions for animal
sessions = db.get_animal_sessions("613")
```

#### Data File Operations

```python
# Add data file
data_file = db.add_data_file(
    session_id="20251210",
    data_type="ephys",
    file_path="/path/to/kilosort/output",
    is_processed=True
)

# Get session data files
data_files = db.get_session_data("20251210")
# Returns: {'ephys': [DataFile], 'tracking': [DataFile], ...}
```

#### Analysis Helpers

```python
# Check data availability
availability = db.check_data_availability(animal_id="613")

# Scan directory for data
results = db.scan_data_directory("/data/directory", auto_add=True)

# Export summary
summary = db.export_summary("database_summary.csv")
```

### Pipeline Integration Class

```python
from database.database_integration import PipelineIntegration

integration = PipelineIntegration("habitat.db")
```

#### Data Registration

```python
# Register Kilosort data
success = integration.register_kilosort_data(
    kilosort_path=Path("/path/to/kilosort"),
    animal_id="613",
    session_id="20251210"
)

# Register tracking data  
success = integration.register_tracking_data(
    tracking_path=Path("/path/to/tracking.csv"),
    session_id="20251210"
)
```

#### Analysis Queries

```python
# Get sessions with specific data types
sessions = integration.get_available_sessions(
    animal_id="613",
    data_types=['ephys', 'tracking']
)

# Create analysis readiness summary
summary = integration.create_analysis_summary()

# Batch process directory
results = integration.batch_process_directory(Path("/data"))
```

## 💻 Command Line Interface

### Initialize Database

```bash
python -m database.database_cli init --db-path habitat.db
```

### Add Animals and Sessions

```bash
# Add animal
python -m database.database_cli add-animal 613 \
    --species mouse \
    --strain C57BL/6 \
    --sex M \
    --birth-date 2025-01-15 \
    --weight 25.5

# Add session
python -m database.database_cli add-session 20251210 613 2025-12-10 \
    --experiment-type open_field \
    --duration 60 \
    --notes "Baseline recording"
```

### Directory Scanning

```bash
# Scan directory and auto-populate database
python -m database.database_cli scan /path/to/data --auto-add

# Scan without adding (dry run)
python -m database.database_cli scan /path/to/data
```

### Database Status and Export

```bash
# Show database status
python -m database.database_cli status

# Filter by animal
python -m database.database_cli status --animal-id 613

# Export summary
python -m database.database_cli export --output summary.csv
```

## 🔬 Integration Examples

### Notebook Analysis Workflow

```python
# In Jupyter notebook
from database.database_integration import load_session, get_session_list
import pandas as pd

# 1. List available sessions with both ephys and tracking
sessions = get_session_list()
complete_sessions = sessions[
    (sessions['ephys'] > 0) & (sessions['tracking'] > 0)
]
print("Complete sessions:")
print(complete_sessions[['animal_id', 'session_id', 'session_date']])

# 2. Load specific session
animal_id, session_id = "613", "20251210"
data = load_session(animal_id, session_id)

# 3. Access loaded data
ephys_data = data['ephys']  # KilosortData object
tracking_data = data['tracking']['animals']  # Dict of animal DataFrames

# 4. Perform analysis
from video.plot_trajectory import plot_proximity_network
fig = plot_proximity_network(tracking_data)

# 5. Analyze place fields (if implemented)
place_fields = ephys_data.get_place_fields(tracking_data['613'])
```

### Automated Pipeline Processing

```python
from database.database_integration import PipelineIntegration
from pathlib import Path

integration = PipelineIntegration("pipeline.db")

# Process all data in directory
base_path = Path("/data/experiments")
results = integration.batch_process_directory(base_path)

# Generate processing report
print(f"Processed {results['summary']['total_sessions']} sessions")
print(f"Complete datasets: {results['summary']['complete_sessions']}")

# Find sessions ready for analysis
ready_sessions = integration.get_available_sessions(
    data_types=['ephys', 'tracking']
)

# Process each ready session
for _, session in ready_sessions.iterrows():
    animal_id = session['animal_id']
    session_id = session['session_id']
    
    # Load and analyze
    data = integration.load_session_data(animal_id, session_id)
    
    # Your analysis here...
    print(f"Analyzing {animal_id}/{session_id}")
```

## 🔄 Common Workflows

### 1. New Experiment Setup

```python
from database.database import HabitatDatabase

db = HabitatDatabase("experiment.db")

# Add animals
for animal_id in ["613", "614", "615"]:
    db.add_animal(animal_id, species="mouse", strain="C57BL/6")

# Add sessions for each animal
for animal_id in ["613", "614", "615"]:
    for day in ["20251201", "20251202", "20251203"]:
        db.add_session(day, animal_id, day, experiment_type="novel_object")
```

### 2. Data Quality Assessment

```python
from database.database_integration import PipelineIntegration

integration = PipelineIntegration("experiment.db")

# Get analysis summary
summary = integration.create_analysis_summary()

# Check data completeness
complete_mask = summary['ready_for_place_analysis']
incomplete_sessions = summary[~complete_mask]

print("Sessions missing data:")
print(incomplete_sessions[['animal_id', 'session_id', 'has_ephys', 'has_tracking']])

# Export for review
incomplete_sessions.to_csv("incomplete_sessions.csv", index=False)
```

### 3. Batch Analysis Pipeline

```python
from database.database_integration import PipelineIntegration
import pandas as pd

integration = PipelineIntegration("experiment.db")

# Get all complete sessions
complete_sessions = integration.get_available_sessions(
    data_types=['ephys', 'tracking']
)

results = []

for _, session in complete_sessions.iterrows():
    try:
        # Load session data
        data = integration.load_session_data(
            session['animal_id'], 
            session['session_id']
        )
        
        # Extract metrics
        ephys = data['ephys']
        tracking = data['tracking']['animals']
        
        # Calculate metrics
        n_clusters = len(ephys.ks_ids)
        n_spikes = len(ephys.spike_times)
        n_animals = len(tracking)
        
        results.append({
            'animal_id': session['animal_id'],
            'session_id': session['session_id'],
            'n_clusters': n_clusters,
            'n_spikes': n_spikes,
            'n_tracked_animals': n_animals
        })
        
    except Exception as e:
        print(f"Error processing {session['animal_id']}/{session['session_id']}: {e}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("analysis_results.csv", index=False)
```

## 🔧 Advanced Usage

### Custom Database Queries

```python
from database.database import HabitatDatabase
from database.database import Animal, ExperimentSession, DataFile

db = HabitatDatabase("habitat.db")

# Direct SQLAlchemy queries
with db.get_session() as session:
    # Find animals with multiple sessions
    animals_with_multiple_sessions = session.query(Animal).join(ExperimentSession).group_by(Animal.id).having(func.count(ExperimentSession.id) > 1).all()
    
    # Find sessions with incomplete data
    incomplete_sessions = session.query(ExperimentSession).filter(
        ~ExperimentSession.data_files.any(DataFile.data_type == 'ephys') |
        ~ExperimentSession.data_files.any(DataFile.data_type == 'tracking')
    ).all()
```

### Database Maintenance

```python
# Backup database
import shutil
shutil.copy("habitat.db", "habitat_backup.db")

# Validate file paths
db = HabitatDatabase("habitat.db")
with db.get_session() as session:
    data_files = session.query(DataFile).all()
    for df in data_files:
        if not Path(df.file_path).exists():
            print(f"Missing file: {df.file_path}")

# Clean up orphaned records
with db.get_session() as session:
    orphaned_files = session.query(DataFile).filter(
        ~DataFile.session_id.in_(
            session.query(ExperimentSession.session_id)
        )
    ).all()
    
    for orphan in orphaned_files:
        session.delete(orphan)
    session.commit()
```

### Performance Optimization

```python
# For large datasets, use pandas for bulk operations
import pandas as pd

# Export to DataFrame for analysis
db = HabitatDatabase("habitat.db")
availability_df = db.check_data_availability()

# Use pandas operations instead of individual queries
high_activity_sessions = availability_df[
    (availability_df['ephys'] > 0) & 
    (availability_df['tracking'] > 0)
].groupby('animal_id').agg({
    'session_id': 'count',
    'session_date': ['min', 'max']
})
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and file paths are correct
2. **Database Locked**: Close all connections before modifying database structure  
3. **File Path Issues**: Use absolute paths or ensure working directory is correct
4. **Data Loading Failures**: Check file formats and validate data integrity

### Debug Mode

```python
# Enable SQLAlchemy logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Create database with echo=True
from sqlalchemy import create_engine
engine = create_engine('sqlite:///debug.db', echo=True)
```

---

## 📝 License

This database system is part of the Habitat Pipeline project.

## 🤝 Contributing

To extend the database system:

1. Add new models to `database.py`
2. Update integration helpers in `database_integration.py` 
3. Add CLI commands to `database_cli.py`
4. Update this documentation

For questions or issues, please create an issue in the project repository.