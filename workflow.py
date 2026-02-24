import ingestion.data_paths as data_paths
import ingestion.kilosort_data_import as kdi
import video.tracking_import as ti
import ingestion.ephys_sync as es
from video.plot_trajectory import plot_animal_path, plot_multiple_paths


animal_id = "613"
session_id = "20251210"
kilosort_path = data_paths.get_kilosort_path(animal_id, session_id)
kilosort_data = kdi.KilosortData(kilosort_path)

# Load and parse tracking data
tracking_path = data_paths.get_tracking_files_by_date(session_id)[0]
tracking_df = ti.load_tracking_data(tracking_path)
animals = ti.parse_tracking(tracking_df)

# Plot individual animal
fig1 = plot_animal_path(animals['rat613'], 'rat613')

# Compare multiple animals
fig2 = plot_multiple_paths(animals)

# Load timestamps for a tracking file
timestamps = ti.load_timestamps(tracking_path)

TSESync, TSBSync, system_time_at_creation = es.load_ephys_sync(animal_id, session_id, 1)
# Get sync mapping
mapping = es.find_sync_mapping(TSBSync, TSESync, system_time_at_creation)

# Plot results with full context
# fig = es.plot_sync_results(mapping, TSESync=TSESync, TSBSync=TSBSync)
