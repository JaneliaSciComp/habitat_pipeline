# habitat_pipeline
Integrated Multi-Animal Electrophysiology and Behavior Analysis Pipeline

## 1. Purpose and Scope
A modular, scalable, and standards-compliant software platform designed to process and analyze large-scale electrophysiology and behavioral data from multiple freely behaving animals recorded simultaneously. The system supports the full experimental lifecycle: raw data ingestion, preprocessing, quality control, spike sorting, multimodal synchronization, advanced multi-animal analysis, visualization, and reproducible deployment. The architecture emphasizes structured and standardized data management, and extensibility for social and multi-brain neuroscience.

## 2. Architectural Principles
Modularity: Each processing and analysis stage is implemented as an independent, composable module.
Scalability: Designed for large Neuropixels datasets and multi-animal (up to 12 rats) experiments; supports parallel and distributed execution.
Reproducibility: All transformations are versioned, parameterized, and auditable.
Interoperability: Clear APIs for integration with existing lab tools and community software.

## Requirements for electrophysiology data analysis
Data exploration and quality check is a key feature of the pipeline. A key requirement here is to have an interactive GUI where researchers can examine neural–behavioral relationships. Specifically the GUI should have browsing of raw video with superimposed extracted animal positions and spikes from selected cells recorded in the same session / animal.

One of the key pipeline outputs are the plots characterizing encoding (representation) of behavioral features by the activity of individual cells or populations. Although plots may have various types of data and analysis shown they will all require similar steps in data preparation. The spike times belonging to individual cells can be considered as a minimum independent and indivisible unit of analysis. There are several ways of how the cells may be selected for specific analyses, e.g. from a single session, multiple sessions belonging to the same animal or multiple sessions and multiple animals. Under this assumption the data should be stored and organized in a way allowing for fast and easy subselection of cells based on combination of the following filters: animal, session, anatomical location (based on channel number), average firing rate (frequency of spikes).

Behavioral features may be selected independently of cells and are derived from the video and audio data. The features may include continuous (in time) variables such as positions (x, y) of multiple individual animals or pairwise distances, discrete events such as interactions (fight, chasing, food caching, etc) or vocalizations. 

After both cells and features are selected we load the data for further analysis. For continuous features the spike time data should be binned to match the sampling rate of the feature (e.g. 40 Hz for position) such that both binned spikes and behavioral feature are represented in the same time reference frame as vectors of the same length and can be used for further statistical and decoding analyses. For discrete events the spike time data is subselected in temporal windows around the times of the individual events and spike times are then aligned relative to the corresponding instances of the event and used for further analyses. 
The preprocessing of spike data for each cell may be done in advance and stored on the hard drive for speeding up the analysis and efficient memory usage. 
