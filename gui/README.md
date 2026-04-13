# Habitat Pipeline GUI

A Streamlit web app for exploring RatCity electrophysiology and behavioral data. Select a cohort, session, and animal, then browse analyses across four tabs.

## Setup

Install the GUI dependency if you haven't already:

```bash
pip install -e ".[gui]"
```

## Running

Always run from the **project root** (not from inside `gui/`):

```bash
streamlit run gui/app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

### 1. Select a session
Use the sidebar to choose:
- **Cohort / Config** — which cohort's data paths to use (Cohort 7 or Cohort 5)
- **Session** — recording date
- **Animal** — implanted animal for neural analyses

Then press **Load & Process Session**. Spike data, behavioral events, and clock sync are loaded once and cached in memory — switching tabs is instant.

### 2. Configure decoding / geometry parameters
Also in the sidebar, before or after loading:

| Parameter | Description |
|---|---|
| Behavior type | Event type to align neural activity to (e.g. EC = encounter) |
| Window start / end | Peri-event time window in seconds |
| Bin size | Spike count bin width in seconds |
| CV folds | Number of cross-validation folds for LDA decoding |
| Min events / class | Minimum events required per opponent to include them; upper limit updates automatically based on the loaded session |

### 3. Browse tabs

#### Tracking & Spatial
Visualize animal movement from video tracking data. Plot types:
- **Individual Path** — single animal's trajectory with time colormap
- **All Paths** — all animals overlaid
- **Heatmap** — spatial occupancy density
- **Territorial Occupancy** — which animal occupied each region most
- **Voronoi Territories** — territory map derived from occupancy
- **Proximity Network** — interaction graph weighted by time spent near each other (configurable distance threshold)

> Tracking data is not available for every session. A warning is shown when absent.

#### Behavioral Events
Visualize social behavioral events logged during the session. Plot types:
- **Interaction Heatmap** — event count matrix between all rat pairs, filterable by event type
- **Per-Rat Heatmap** — behavior breakdown for a specific rat
- **Event Timeline** — chronological view of all events

> Requires a behavioral events CSV for the session.

#### Neural Decoding
Cross-validated LDA decoding of opponent identity from single-cell and population firing rates.

Because decoding can take several minutes, it is **not run automatically**. Press **Run Decoding** to start. Results are saved to `.gui_cache/` and loaded instantly on revisit — changing any parameter automatically invalidates the cache and prompts a re-run.

Plot views once decoding is complete:
- **Accuracy Distribution** — per-cell accuracy histogram and boxplot
- **Best Cells** — bar chart of top-N cells + confusion matrix
- **Summary** — 6-panel dashboard
- **Top Cells Firing Rates** — peri-event time histograms for the best cells split by opponent

#### Population Geometry
Dimensionality reduction of population firing rate trajectories aligned to behavioral events.

Press **Run Population Geometry Analysis** to compute. Also cached to disk.

Parameters (set within the tab):
- **Method** — PCA or UMAP
- **Components** — number of dimensions (2–10)
- **Normalization** — none, z-score, or baseline subtraction
- **Alignment** — align to event start, end, or center

Plot views:
- **Population Dynamics** — neural trajectories in reduced space per opponent class (optionally show individual trials)
- **PCA Summary** — explained variance + component projections
- **Normalized Population Matrix** — heatmap of population activity

## Caching

| Data | How it's cached |
|---|---|
| Session manifest | `st.cache_data` (per cohort config) |
| Spike data, behavioral events, clock sync | `st.cache_resource` (in memory, per session) |
| Decoding results | Disk pickle in `.gui_cache/` |
| Population geometry results | Disk pickle in `.gui_cache/` |

The `.gui_cache/` directory is created automatically and is excluded from git. To force a recompute, delete the relevant `.pkl` file from `.gui_cache/` and press Run again.
