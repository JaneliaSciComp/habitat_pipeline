# Habitat Pipeline GUI

Two complementary web apps for exploring RatCity electrophysiology and behavioral data:

| App | File | Framework | Best for |
|---|---|---|---|
| Static explorer | `gui/app.py` | Streamlit | Browsing all analyses across tabs |
| Interactive explorer | `gui/interactive_app.py` | Panel + Bokeh + Plotly | Linked timeline ↔ Rastermap ↔ PCA |

## Setup

Install the GUI dependencies if you haven't already:

```bash
pip install -e ".[gui]"
pip install rastermap  # required for the interactive app
```

## Running

Always run from the **project root** (not from inside `gui/`).

**Static explorer (Streamlit):**
```bash
streamlit run gui/app.py
```
Opens at `http://localhost:8501`.

**Interactive explorer (Panel):**
```bash
panel serve gui/interactive_app.py --show
```
Opens at `http://localhost:5006/interactive_app`.

---

## Static Explorer (`gui/app.py`)

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

---

## Interactive Explorer (`gui/interactive_app.py`)

Three vertically-linked panels that update together as you navigate the recording.

### 1. Load a session

Use the **Session** section of the sidebar to pick a cohort, session date, and animal, then press **Load Session**. This loads spike data, fits Rastermap on quality-filtered cells, and synchronises behavioral events to the ephys clock. Loading takes roughly 1–2 minutes; all subsequent interactions are instant.

The loaded data is held in a process-level cache. Toggling the dark/light theme reloads the page but restores the session automatically — no need to press Load Session again.

### 2. Configure behavioral filters

| Parameter | Description |
|---|---|
| Behavior type | Event type to display (e.g. EC — Encounter) |
| Min events / opponent | Opponents with fewer events than this are hidden from the timeline and PCA |
| PCA bin size (s) | Temporal bin width used to build the firing-rate matrix for PCA |
| Rastermap bin size (s) | Temporal bin width used to build the Rastermap image |

Changing behavior type or min-events updates the timeline and PCA immediately — no reload required.

### 3. Three linked panels

#### Behavioral event timeline (top)
Bokeh plot showing all events of the selected type that involve the chosen animal, plotted in ephys time (seconds). Each opponent gets a distinct color.

- **y-axis** — rat ID (initiator = filled dot, victim = hollow dot)
- **Zoom/pan** — use the horizontal wheel-zoom or x-pan tools; zooming here filters which events appear on the PCA plot below
- Only opponents meeting the **Min events** threshold are shown

#### Rastermap (middle)
Firing-rate heatmap of quality-filtered cells sorted by activity similarity (via Rastermap). The x-axis is locked to the same range as the timeline — zooming either panel zooms both simultaneously.

#### PCA trajectory (bottom, Plotly 3D)
Full-session population trajectory in the top-3 PCA components, computed on raw z-scored firing rates of quality cells.

- **Background line** — entire session trajectory colored by time (Viridis colorscale), shown at low opacity
- **Colored dots** — events within the current timeline view window, one color per opponent (matching the timeline)
- **Updates automatically** every ~600 ms when the timeline zoom/pan changes

The PCA space is fixed (fit once on the full recording); panning the timeline just filters which event dots are visible — the axes never shift.

### 4. Theme
Use the theme toggle (top-right) to switch between light and dark mode. The Plotly PCA panel adapts its background automatically.

---

## Caching

### Static explorer

| Data | How it's cached |
|---|---|
| Session manifest | `st.cache_data` (per cohort config) |
| Spike data, behavioral events, clock sync | `st.cache_resource` (in memory, per session) |
| Decoding results | Disk pickle in `.gui_cache/` |
| Population geometry results | Disk pickle in `.gui_cache/` |

The `.gui_cache/` directory is created automatically and is excluded from git. To force a recompute, delete the relevant `.pkl` file from `.gui_cache/` and press Run again.

### Interactive explorer

| Data | How it's cached |
|---|---|
| Spike data, Rastermap image, behavioral events | `pn.state.cache` (process-level, in memory) |

The process-level cache survives page reloads (e.g. theme toggle) but is cleared when the Panel server process is restarted. To force a fresh load, restart the server and press **Load Session** again.
