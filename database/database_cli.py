"""
Database CLI Tools for Habitat Pipeline

Command-line interface for managing the habitat database.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

from .database import HabitatDatabase


def init_database(args):
    """Initialize a new database"""
    db_path = args.db_path or "habitat_pipeline.db"
    db = HabitatDatabase(db_path)
    print(f"Database initialized at: {db_path}")
    print(f"Database info: {db}")


def add_animal(args):
    """Add an animal to the database"""
    db = HabitatDatabase(args.db_path)
    
    birth_date = None
    if args.birth_date:
        birth_date = datetime.strptime(args.birth_date, "%Y-%m-%d")
    
    animal = db.add_animal(
        animal_id=args.animal_id,
        species=args.species,
        strain=args.strain,
        sex=args.sex,
        birth_date=birth_date,
        weight_g=args.weight,
        notes=args.notes
    )
    print(f"Added animal: {animal}")


def add_session(args):
    """Add a session to the database"""
    db = HabitatDatabase(args.db_path)
    
    session = db.add_session(
        session_id=args.session_id,
        animal_id=args.animal_id,
        session_date=args.date,
        experiment_type=args.experiment_type,
        duration_minutes=args.duration,
        notes=args.notes
    )
    print(f"Added session: {session}")


def scan_directory(args):
    """Scan directory and populate database"""
    db = HabitatDatabase(args.db_path, verbose=args.verbose)
    
    print(f"Scanning directory: {args.directory}")
    results = db.scan_data_directory(args.directory, auto_add=args.auto_add, verbose=args.verbose)
    
    print(f"\\nScan results:")
    print(f"  Animals found: {len(results['animals_found'])}")
    print(f"  Sessions found: {len(results['sessions_found'])}")
    print(f"  Data files found: {len(results['data_files_found'])}")
    
    if results['errors']:
        print(f"  Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"    - {error}")
    
    # Show summary when not verbose
    if not args.verbose:
        print(f"\\nSummary (verbose mode disabled):")
        print(f"  Total animals processed: {len(results['animals_found'])}")
        print(f"  Total sessions processed: {len(results['sessions_found'])}")
        print(f"  Total data files found: {len(results['data_files_found'])}")
        if results['errors']:
            print(f"  Total errors: {len(results['errors'])}")
        else:
            print(f"  No errors encountered")


def show_status(args):
    """Show database status and availability"""
    db = HabitatDatabase(args.db_path)
    
    print(f"Database info: {db}")
    print()
    
    # Show availability matrix
    availability = db.check_data_availability(args.animal_id, args.session_id)
    
    if not availability.empty:
        print("Data Availability:")
        print(availability.to_string(index=False))
    else:
        print("No data found.")


def export_summary(args):
    """Export database summary"""
    db = HabitatDatabase(args.db_path)
    
    output_path = args.output or "habitat_database_summary.csv"
    summary = db.export_summary(output_path)
    
    print(f"Summary exported to: {output_path}")
    print(f"Records exported: {len(summary)}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Habitat Pipeline Database Management")
    parser.add_argument("--db-path", default="habitat_pipeline.db", help="Path to database file")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output (default: True)")
    parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new database")
    init_parser.set_defaults(func=init_database)
    
    # Add animal command
    animal_parser = subparsers.add_parser("add-animal", help="Add animal to database")
    animal_parser.add_argument("animal_id", help="Animal ID")
    animal_parser.add_argument("--species", help="Species")
    animal_parser.add_argument("--strain", help="Strain")
    animal_parser.add_argument("--sex", help="Sex")
    animal_parser.add_argument("--birth-date", help="Birth date (YYYY-MM-DD)")
    animal_parser.add_argument("--weight", type=float, help="Weight in grams")
    animal_parser.add_argument("--notes", help="Notes")
    animal_parser.set_defaults(func=add_animal)
    
    # Add session command
    session_parser = subparsers.add_parser("add-session", help="Add session to database")
    session_parser.add_argument("session_id", help="Session ID")
    session_parser.add_argument("animal_id", help="Animal ID")
    session_parser.add_argument("date", help="Session date (YYYY-MM-DD or YYYYMMDD)")
    session_parser.add_argument("--experiment-type", help="Experiment type")
    session_parser.add_argument("--duration", type=float, help="Duration in minutes")
    session_parser.add_argument("--notes", help="Notes")
    session_parser.set_defaults(func=add_session)
    
    # Scan directory command
    scan_parser = subparsers.add_parser("scan", help="Scan directory and populate database")
    scan_parser.add_argument("directory", help="Directory to scan")
    scan_parser.add_argument("--auto-add", action="store_true", help="Automatically add found data")
    scan_parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output (default: True)")
    scan_parser.add_argument("--quiet", dest="verbose", action="store_false", help="Disable verbose output")
    scan_parser.set_defaults(func=scan_directory)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show database status")
    status_parser.add_argument("--animal-id", help="Filter by animal ID")
    status_parser.add_argument("--session-id", help="Filter by session ID")
    status_parser.set_defaults(func=show_status)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export database summary")
    export_parser.add_argument("--output", help="Output file path")
    export_parser.set_defaults(func=export_summary)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set db_path on args if not set
    if not hasattr(args, 'db_path') or not args.db_path:
        args.db_path = "habitat_pipeline.db"
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()