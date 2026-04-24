#!/usr/bin/env python3
"""
Habitat Pipeline Workflow

This script provides a complete workflow for processing electrophysiology and
video tracking data in the habitat pipeline. It integrates Kilosort data loading,
video tracking analysis, and synchronization between ephys and behavioral data.

Usage:
    python workflow.py --animal_id 613 --session_id 20251210
    python workflow.py --animal_id rat613 --session_id 20251210_110059 --output_dir results/
    python workflow.py -a 613 -s 20251210 --skip_plots --verbose

Author: Mikhail Proskurin
Date: 2026
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
import traceback

# Import our custom modules
import ingestion.data_paths as data_path
try:
    import ingestion.kilosort_data_import as kilosort_data
except ImportError:
    print("Warning: kilosort_data_import module not available")
    kilosort_data = None

import video.tracking_import as tracking_import
import video.plot_trajectory as path_viz

# Try to import ephys sync if available
try:
    import ingestion.ephys_sync as ephys_sync
    EPHYS_SYNC_AVAILABLE = True
except ImportError:
    print("Warning: ephys_sync module not available")
    EPHYS_SYNC_AVAILABLE = False

# Import plotting libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the workflow.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Complete habitat pipeline workflow for ephys and tracking data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with partial IDs
    python workflow.py --animal_id 613 --session_id 20251210
    
    # Full IDs with custom output directory
    python workflow.py --animal_id rat613 --session_id 20251210_110059 --output_dir results/
    
    # Skip plotting (faster processing)
    python workflow.py -a 613 -s 20251210 --skip_plots
    
    # Verbose output for debugging
    python workflow.py -a 613 -s 20251210 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument('-a', '--animal_id', type=str, required=True,
                      help='Animal identifier (full or partial, e.g., "613" or "rat613")')
    parser.add_argument('-s', '--session_id', type=str, required=True,
                      help='Session identifier (full or partial, e.g., "20251210" or "20251210_110059")')
    
    # Optional arguments
    parser.add_argument('-o', '--output_dir', type=str, default='output',
                      help='Output directory for results (default: output/)')
    parser.add_argument('-c', '--config_path', type=str, default=None,
                      help='Path to configuration file (default: config/default_paths.json)')
    
    # Processing options
    parser.add_argument('--skip_plots', action='store_true',
                      help='Skip generating visualization plots (faster processing)')
    parser.add_argument('--skip_ephys', action='store_true',
                      help='Skip electrophysiology data processing')
    parser.add_argument('--skip_tracking', action='store_true',
                      help='Skip video tracking data processing')
    parser.add_argument('--skip_sync', action='store_true',
                      help='Skip ephys-video synchronization')
    
    # Output options
    parser.add_argument('--save_plots', action='store_true',
                      help='Save plots to files instead of just displaying')
    parser.add_argument('--plot_format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                      help='Format for saved plots (default: png)')
    parser.add_argument('--plot_dpi', type=int, default=300,
                      help='DPI for saved plots (default: 300)')
    
    # Verbose output
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output for debugging')
    
    return parser.parse_args()


def setup_output_directory(output_dir: str, animal_id: str, session_id: str) -> Path:
    """
    Create and setup output directory structure.
    
    Args:
        output_dir: Base output directory
        animal_id: Animal identifier
        session_id: Session identifier
        
    Returns:
        Path to the created session-specific output directory
    """
    # Create session-specific directory
    session_dir = Path(output_dir) / f"{animal_id}_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (session_dir / "plots").mkdir(exist_ok=True)
    (session_dir / "data").mkdir(exist_ok=True)
    
    return session_dir


def process_kilosort_data(animal_id: str, session_id: str, config_path: Optional[str] = None,
                         verbose: bool = False) -> tuple:
    """
    Process Kilosort electrophysiology data.
    
    Args:
        animal_id: Animal identifier
        session_id: Session identifier
        config_path: Optional path to config file
        verbose: Enable verbose output
        
    Returns:
        Tuple of (kilosort_path, kilosort_data_object)
    """
    print("=" * 60)
    print("PROCESSING KILOSORT DATA")
    print("=" * 60)
    
    try:
        # Get Kilosort path using our path management function
        print(f"Looking for Kilosort data for animal '{animal_id}', session '{session_id}'...")
        kilosort_path = data_path.get_kilosort_path(animal_id, session_id, config_path)
        
        # Verify the path exists and has required files
        is_valid, message = data_path.verify_kilosort_path(
            kilosort_path, check_files=True, return_message=True
        )
        
        if not is_valid:
            raise FileNotFoundError(f"Kilosort validation failed: {message}")
        
        print(f"✓ Found Kilosort data at: {kilosort_path}")
        
        # Load Kilosort data if the module is available
        kilosort_data_obj = None
        if kilosort_data is not None:
            try:
                print("Loading Kilosort data...")
                kilosort_data_obj = kilosort_data.load_kilosort_data(kilosort_path)
                print(f"✓ Loaded Kilosort data: {kilosort_data_obj}")
                
                if verbose:
                    print("Kilosort Data Details:")
                    print(f"  Animal ID: {kilosort_data_obj.animal_id}")
                    print(f"  Session ID: {kilosort_data_obj.session_id}")
                    print(f"  Number of clusters: {len(kilosort_data_obj.ks_ids)}")
                    print(f"  Number of spikes: {len(kilosort_data_obj.spike_times)}")
                    if hasattr(kilosort_data_obj, 'metadata'):
                        print(f"  Metadata: {kilosort_data_obj.metadata}")
                        
            except Exception as e:
                print(f"Warning: Could not load Kilosort data object: {e}")
                if verbose:
                    traceback.print_exc()
        
        return kilosort_path, kilosort_data_obj
        
    except Exception as e:
        print(f"ERROR: Failed to process Kilosort data: {e}")
        if verbose:
            traceback.print_exc()
        return None, None


def process_tracking_data(session_id: str, verbose: bool = False) -> tuple:
    """
    Process video tracking data.
    
    Args:
        session_id: Session identifier
        verbose: Enable verbose output
        
    Returns:
        Tuple of (tracking_path, tracking_df, animals_dict, timestamps)
    """
    print("\n" + "=" * 60)
    print("PROCESSING VIDEO TRACKING DATA")
    print("=" * 60)
    
    try:
        # This would need to be implemented based on your tracking file discovery method
        # For now, I'll use a placeholder approach
        print(f"Looking for tracking data for session '{session_id}'...")
        
        # Try to find tracking files (you'll need to implement get_tracking_files_by_date)
        # tracking_files = data_paths.get_tracking_files_by_date(session_id)
        # For now, we'll create a placeholder
        tracking_path = None  # Placeholder
        
        if tracking_path is None:
            print("Warning: No tracking file discovery method implemented")
            print("You'll need to manually specify tracking file paths")
            return None, None, None, None
        
        print(f"✓ Found tracking file: {tracking_path}")
        
        # Load tracking data
        print("Loading tracking data...")
        tracking_df = tracking_import.load_tracking_data(tracking_path)
        print(f"✓ Loaded tracking data: {tracking_df.shape}")
        
        if verbose:
            print("Tracking DataFrame columns:", list(tracking_df.columns))
            print("Tracking DataFrame shape:", tracking_df.shape)
            print("First few rows:")
            print(tracking_df.head())
        
        # Parse tracking data by animal
        print("Parsing tracking data by animal...")
        animals = tracking_import.parse_tracking(tracking_df)
        print(f"✓ Found {len(animals)} animals: {list(animals.keys())}")
        
        if verbose:
            for animal_name, animal_df in animals.items():
                stats = path_viz.calculate_path_statistics(animal_df)
                print(f"\n{animal_name} statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value:.2f}")
        
        # Load timestamps
        print("Loading timestamps...")
        try:
            timestamps = tracking_import.load_timestamps(tracking_path)
            print(f"✓ Loaded timestamps: {timestamps.shape}")
        except Exception as e:
            print(f"Warning: Could not load timestamps: {e}")
            timestamps = None
        
        return tracking_path, tracking_df, animals, timestamps
        
    except Exception as e:
        print(f"ERROR: Failed to process tracking data: {e}")
        if verbose:
            traceback.print_exc()
        return None, None, None, None


def generate_visualizations(animals_dict: Dict, output_dir: Path, animal_id: str,
                          save_plots: bool = False, plot_format: str = 'png', 
                          plot_dpi: int = 300, verbose: bool = False) -> None:
    """
    Generate visualization plots for tracking data.
    
    Args:
        animals_dict: Dictionary of animal tracking data
        output_dir: Output directory for plots
        animal_id: Target animal ID
        save_plots: Whether to save plots to files
        plot_format: Format for saved plots
        plot_dpi: DPI for saved plots
        verbose: Enable verbose output
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    try:
        if not animals_dict:
            print("No tracking data available for visualization")
            return
        
        plots_dir = output_dir / "plots"
        
        # Find the target animal (handle partial matching)
        target_animal = None
        for animal_name in animals_dict.keys():
            if animal_id in animal_name:
                target_animal = animal_name
                break
        
        if target_animal is None:
            print(f"Warning: Target animal '{animal_id}' not found in tracking data")
            print(f"Available animals: {list(animals_dict.keys())}")
            target_animal = list(animals_dict.keys())[0]  # Use first animal as fallback
            print(f"Using '{target_animal}' instead")
        
        # 1. Individual animal path
        print(f"Creating individual path plot for {target_animal}...")
        fig1 = path_viz.plot_animal_path(animals_dict[target_animal], target_animal)
        
        if save_plots:
            output_file = plots_dir / f"{target_animal}_path.{plot_format}"
            path_viz.save_visualization(fig1, output_file, dpi=plot_dpi, format=plot_format)
        
        # 2. Multiple animals comparison (if more than one animal)
        if len(animals_dict) > 1:
            print("Creating multi-animal comparison plot...")
            fig2 = path_viz.plot_multiple_paths(animals_dict)
            
            if save_plots:
                output_file = plots_dir / f"all_animals_comparison.{plot_format}"
                path_viz.save_visualization(fig2, output_file, dpi=plot_dpi, format=plot_format)
        
        # 3. Position heatmap for target animal
        print(f"Creating position heatmap for {target_animal}...")
        fig3 = path_viz.plot_path_heatmap(animals_dict[target_animal], target_animal)
        
        if save_plots:
            output_file = plots_dir / f"{target_animal}_heatmap.{plot_format}"
            path_viz.save_visualization(fig3, output_file, dpi=plot_dpi, format=plot_format)
        
        print("✓ Visualizations created successfully")
        
        if not save_plots:
            print("Plots are displayed but not saved. Use --save_plots to save them.")
        
    except Exception as e:
        print(f"ERROR: Failed to generate visualizations: {e}")
        if verbose:
            traceback.print_exc()


def process_synchronization(animal_id: str, session_id: str, timestamps: Optional[Any] = None,
                          verbose: bool = False) -> None:
    """
    Process ephys-video synchronization.
    
    Args:
        animal_id: Animal identifier
        session_id: Session identifier
        timestamps: Video timestamps
        verbose: Enable verbose output
    """
    print("\n" + "=" * 60)
    print("PROCESSING EPHYS-VIDEO SYNCHRONIZATION")
    print("=" * 60)
    
    if not EPHYS_SYNC_AVAILABLE:
        print("Ephys sync module not available - skipping synchronization")
        return
    
    try:
        print(f"Loading ephys sync data for animal '{animal_id}', session '{session_id}'...")
        
        # Load ephys sync data (channel 1 as example)
        TSESync, TSBSync, system_time_at_creation = ephys_sync.load_ephys_sync(
            animal_id, session_id, 1
        )
        print("✓ Loaded ephys sync data")
        
        if verbose:
            print(f"TSESync shape: {TSESync.shape if TSESync is not None else 'None'}")
            print(f"TSBSync shape: {TSBSync.shape if TSBSync is not None else 'None'}")
            print(f"System time at creation: {system_time_at_creation}")
        
        # Find sync mapping
        print("Finding sync mapping between ephys and video...")
        mapping = ephys_sync.find_sync_mapping(TSBSync, TSESync, system_time_at_creation)
        print("✓ Sync mapping completed")
        
        if verbose and mapping is not None:
            print("Sync mapping details:")
            print(f"  Type: {type(mapping)}")
            if hasattr(mapping, 'shape'):
                print(f"  Shape: {mapping.shape}")
        
        # Generate sync plots if available
        # Uncomment if plot_sync_results is available
        # print("Generating sync visualization...")
        # fig = ephys_sync.plot_sync_results(mapping, TSESync=TSESync, TSBSync=TSBSync)
        # print("✓ Sync visualization created")
        
        return mapping
        
    except Exception as e:
        print(f"ERROR: Failed to process synchronization: {e}")
        if verbose:
            traceback.print_exc()
        return None


def main():
    """
    Main workflow function that orchestrates the entire pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("HABITAT PIPELINE WORKFLOW")
    print("=" * 60)
    print(f"Animal ID: {args.animal_id}")
    print(f"Session ID: {args.session_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Config Path: {args.config_path or 'default'}")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir, args.animal_id, args.session_id)
    print(f"✓ Output directory created: {output_dir}")
    
    # Initialize results dictionary
    results = {
        'animal_id': args.animal_id,
        'session_id': args.session_id,
        'output_dir': str(output_dir),
        'success': True,
        'errors': []
    }
    
    try:
        # 1. Process Kilosort data
        if not args.skip_ephys:
            kilosort_path, kilosort_data_obj = process_kilosort_data(
                args.animal_id, args.session_id, args.config_path, args.verbose
            )
            results['kilosort_path'] = str(kilosort_path) if kilosort_path else None
            results['kilosort_data'] = kilosort_data_obj is not None
        else:
            print("Skipping ephys data processing (--skip_ephys)")
            kilosort_path, kilosort_data_obj = None, None
        
        # 2. Process tracking data  
        if not args.skip_tracking:
            tracking_path, tracking_df, animals, timestamps = process_tracking_data(
                args.session_id, args.verbose
            )
            results['tracking_path'] = str(tracking_path) if tracking_path else None
            results['animals_found'] = list(animals.keys()) if animals else []
        else:
            print("Skipping tracking data processing (--skip_tracking)")
            tracking_path, tracking_df, animals, timestamps = None, None, None, None
        
        # 3. Generate visualizations
        if not args.skip_plots and animals:
            generate_visualizations(
                animals, output_dir, args.animal_id,
                args.save_plots, args.plot_format, args.plot_dpi, args.verbose
            )
            results['plots_generated'] = True
        else:
            if args.skip_plots:
                print("Skipping plot generation (--skip_plots)")
            else:
                print("No tracking data available for plotting")
            results['plots_generated'] = False
        
        # 4. Process synchronization
        if not args.skip_sync and not args.skip_ephys and not args.skip_tracking:
            sync_mapping = process_synchronization(
                args.animal_id, args.session_id, timestamps, args.verbose
            )
            results['synchronization'] = sync_mapping is not None
        else:
            print("Skipping synchronization (data missing or explicitly skipped)")
            results['synchronization'] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("WORKFLOW SUMMARY")
        print("=" * 60)
        print(f"✓ Animal ID: {results['animal_id']}")
        print(f"✓ Session ID: {results['session_id']}")
        print(f"✓ Output Directory: {results['output_dir']}")
        print(f"✓ Kilosort Data: {'✓' if results.get('kilosort_data') else '✗'}")
        print(f"✓ Tracking Data: {'✓' if results['animals_found'] else '✗'}")
        print(f"✓ Animals Found: {', '.join(results['animals_found']) if results['animals_found'] else 'None'}")
        print(f"✓ Plots Generated: {'✓' if results['plots_generated'] else '✗'}")
        print(f"✓ Synchronization: {'✓' if results['synchronization'] else '✗'}")
        
        if results['errors']:
            print("\nErrors encountered:")
            for error in results['errors']:
                print(f"  • {error}")
        else:
            print("\n✓ Workflow completed successfully!")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        if args.verbose:
            traceback.print_exc()
        results['success'] = False
        results['errors'].append(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
