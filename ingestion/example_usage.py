"""
Example usage of the KilosortData class for habitat_pipeline.

This script demonstrates how to use the KilosortData class to:
1. Load Kilosort 4 output data
2. Filter clusters by various criteria
3. Extract spike times for behavioral analysis
4. Bin spike times for continuous features
5. Align spikes to behavioral events
"""

from kilosort_data import KilosortData, load_kilosort_session, load_multiple_sessions
import numpy as np
from pathlib import Path

def example_usage():
    """
    Example demonstrating how to use KilosortData for multi-animal analysis.
    """
    
    print("=== KilosortData Example Usage ===\n")
    
    # Example 1: Load a single session
    print("1. Loading a single Kilosort session:")
    try:
        # Replace with actual path to your Kilosort output
        example_path = Path("./example_kilosort_output")  
        
        if not example_path.exists():
            print("   Creating example directory structure...")
            # Create example structure for demonstration
            example_path.mkdir(exist_ok=True)
            
            # Create dummy files for demonstration
            np.save(example_path / "spike_times.npy", np.random.randint(0, 300000, 1000))
            np.save(example_path / "spike_clusters.npy", np.random.randint(0, 10, 1000))
            np.save(example_path / "amplitudes.npy", np.random.random(1000))
            
            print("   Example files created.")
        
        ks_data = load_kilosort_session(
            data_path=example_path,
            animal_id="rat001", 
            session_id="session001"
        )
        
        print(f"   Loaded: {ks_data}")
        
    except Exception as e:
        print(f"   Demo mode: {e}")
        return
    
    # Example 2: Filter clusters by criteria
    print("\n2. Filtering clusters by criteria:")
    try:
        # Get all clusters
        all_clusters = ks_data.get_clusters()
        print(f"   Total clusters: {len(all_clusters)}")
        
        # Get clusters with specific firing rate range
        good_clusters = ks_data.get_clusters(
            min_firing_rate=1.0,
            max_firing_rate=50.0,
            cluster_group='good'
        )
        print(f"   Good clusters (1-50 Hz): {len(good_clusters)}")
        
        # Get clusters from specific channels (e.g., hippocampus)
        if 'channel' in ks_data.cluster_info.columns:
            hipp_channels = list(range(100, 200))  # Example channel range
            hipp_clusters = ks_data.get_clusters(channels=hipp_channels)
            print(f"   Clusters in channels 100-200: {len(hipp_clusters)}")
        
    except Exception as e:
        print(f"   Error in filtering: {e}")
    
    # Example 3: Extract spike times for analysis
    print("\n3. Extracting spike times:")
    try:
        if good_clusters:
            # Get spike times for first good cluster
            cluster_id = good_clusters[0]
            spike_times = ks_data.get_spike_times(cluster_id)
            print(f"   Cluster {cluster_id}: {len(spike_times)} spikes")
            
            # Get spike times for multiple clusters
            multi_spikes = ks_data.get_spike_times(good_clusters[:3])
            print(f"   Loaded spike times for {len(multi_spikes)} clusters")
            
    except Exception as e:
        print(f"   Error extracting spikes: {e}")
    
    # Example 4: Bin spike times for continuous behavioral features
    print("\n4. Binning spike times for behavioral analysis:")
    try:
        if good_clusters:
            # Bin spikes at 40 Hz (25ms bins) for position analysis
            binned_spikes = ks_data.bin_spike_times(
                good_clusters[0],
                bin_size=0.025,  # 25ms bins
                start_time=0,
                end_time=60      # First minute
            )
            print(f"   Binned spikes shape: {binned_spikes.shape}")
            print(f"   Mean firing rate: {binned_spikes.mean():.2f} spikes/bin")
            
            # Bin multiple clusters
            multi_binned = ks_data.bin_spike_times(
                good_clusters[:3],
                bin_size=0.025
            )
            print(f"   Binned {len(multi_binned)} clusters")
            
    except Exception as e:
        print(f"   Error in binning: {e}")
    
    # Example 5: Event-aligned spike analysis
    print("\n5. Event-aligned spike analysis:")
    try:
        if good_clusters:
            # Simulate behavioral events (e.g., social interactions)
            event_times = np.array([10.5, 25.3, 45.1, 58.7])  # Example event times
            
            aligned_spikes = ks_data.get_event_aligned_spikes(
                good_clusters[:2],
                event_times=event_times,
                window_pre=1.0,   # 1 second before
                window_post=2.0   # 2 seconds after
            )
            
            print(f"   Aligned spikes for {len(aligned_spikes)} clusters")
            for cluster_id, events in aligned_spikes.items():
                print(f"   Cluster {cluster_id}: {len(events)} events")
                if events:
                    print(f"     Event 1: {len(events[0])} spikes")
                    
    except Exception as e:
        print(f"   Error in event alignment: {e}")
    
    # Example 6: Get cluster waveforms
    print("\n6. Extracting cluster waveforms:")
    try:
        if good_clusters and ks_data.templates is not None:
            cluster_id = good_clusters[0]
            waveform = ks_data.get_cluster_waveform(cluster_id)
            print(f"   Waveform for cluster {cluster_id}: shape {waveform.shape}")
            
    except Exception as e:
        print(f"   Error getting waveform: {e}")
    
    # Example 7: Save processed data
    print("\n7. Saving processed data:")
    try:
        output_path = Path("./processed_kilosort_data")
        ks_data.save_processed_data(output_path)
        print(f"   Saved processed data to: {output_path}")
        
    except Exception as e:
        print(f"   Error saving data: {e}")


def multi_session_example():
    """
    Example of loading and analyzing multiple sessions for multi-animal studies.
    """
    
    print("\n=== Multi-Session Analysis Example ===\n")
    
    # Configuration for multiple sessions
    session_configs = [
        {
            "data_path": "./example_kilosort_output", 
            "animal_id": "rat001", 
            "session_id": "day1"
        },
        # Add more sessions as needed
        # {
        #     "data_path": "path/to/rat001/day2", 
        #     "animal_id": "rat001", 
        #     "session_id": "day2"
        # },
        # {
        #     "data_path": "path/to/rat002/day1", 
        #     "animal_id": "rat002", 
        #     "session_id": "day1"
        # },
    ]
    
    try:
        # Load multiple sessions
        sessions = load_multiple_sessions(session_configs)
        print(f"Loaded {len(sessions)} sessions successfully")
        
        # Analyze across sessions
        all_firing_rates = []
        for session in sessions:
            try:
                rates = session.firing_rates
                all_firing_rates.extend(rates.values)
                print(f"  {session.animal_id}/{session.session_id}: "
                      f"{len(rates)} clusters, "
                      f"mean rate: {rates.mean():.2f} Hz")
            except Exception as e:
                print(f"  Error analyzing session: {e}")
        
        if all_firing_rates:
            print(f"\nOverall statistics:")
            print(f"  Total clusters: {len(all_firing_rates)}")
            print(f"  Mean firing rate: {np.mean(all_firing_rates):.2f} Hz")
            print(f"  Std firing rate: {np.std(all_firing_rates):.2f} Hz")
        
    except Exception as e:
        print(f"Error in multi-session analysis: {e}")


if __name__ == "__main__":
    example_usage()
    multi_session_example()
    
    print("\n=== Integration with Behavioral Analysis ===")
    print("""
    The KilosortData class is designed to integrate seamlessly with behavioral analysis:
    
    1. Continuous Features (position, velocity, etc.):
       - Use bin_spike_times() with appropriate bin_size matching behavioral sampling rate
       - Example: bin_size=0.025 for 40 Hz position tracking
       
    2. Discrete Events (interactions, vocalizations, etc.):
       - Use get_event_aligned_spikes() to extract spikes around specific events
       - Analyze pre/post event activity patterns
       
    3. Multi-animal Analysis:
       - Load sessions from different animals using load_multiple_sessions()
       - Filter clusters by anatomical location using channel information
       - Compare neural activity patterns across animals and conditions
    
    4. Quality Control:
       - Use firing rate filters to exclude low-quality clusters
       - Access cluster quality metrics through cluster_info DataFrame
       - Visualize waveforms using get_cluster_waveform()
    
    5. Data Management:
       - Efficient memory usage through lazy loading and caching
       - Save processed data for quick reloading
       - Structured metadata for reproducible analysis
    """)