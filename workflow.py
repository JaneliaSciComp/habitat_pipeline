import ingestion.data_paths as data_paths
import ingestion.kilosort_data_import as kdi
import video.tracking_import as ti

animal_id = "613"
session_id = "20251210"
kilosort_path = data_paths.get_kilosort_path(animal_id, session_id)
kilosort_data = kdi.KilosortData(kilosort_path)


tracking_path = data_paths.get_tracking_files_by_date(session_id)[0]
tracking_df = ti.load_tracking_data(tracking_path)