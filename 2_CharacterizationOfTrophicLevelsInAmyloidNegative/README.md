# Characterization of Trophic Levels in Amyloid-Negative Subjects

This directory contains the analysis pipeline used to characterize **trophic hierarchy organization in amyloid-negative healthy controls**.

The analysis uses the **trophic levels (TL)** and **generative effective connectivity (GEC)** matrices obtained in **1_FromRawDataToTrophic/**


The pipeline performs:

1. Data harmonization across acquisition sites
2. Computation of graph-theoretic metrics from GEC
3. Regression analysis relating graph metrics to trophic levels
4. Visualization of trophic hierarchy structure

---

# Directory Structure
```
2_CharacterizationOfTrophicLevelsInAmyloidNegative/

├── src/
│ └── functions_pipeline/
│ ├── init.py
│ ├── harmonization.py
│ ├── metrics.py
│ └── plotting.py
│
└── main_pipeline.py
```


---

# Overview of the Pipeline

Input data:

- Trophic levels per subject and region
- Generative effective connectivity matrices
- Subject covariates (age, gender, education)
- Site/scanner information

Processing steps:
```
Load TL + GEC
↓
Harmonization (ComBat)
↓
Graph metric computation
↓
Regression analysis
↓
Visualization
```


---

# Input Data

The pipeline expects the following files:
```
TL_ADNI3_HCn_dbs80.mat
GEC_ADNI3_HCn_dbs80.mat
Demographics.xlsx
Sites.xlsx
```


### File descriptions

**TL_ADNI3_HCn_dbs80.mat**
TL : subjects × regions

Trophic level for each brain region.

---

**GEC_ADNI3_HCn_dbs80.mat**
GEC : subjects × regions × regions

Generative effective connectivity matrices.

---

**Demographics.xlsx**

Subject covariates used for harmonization:

- Age
- Gender
- Education

---

**Sites.xlsx**

Scanner/site information used in harmonization.

---

# 1. Data Harmonization

Module:
```
src/functions_pipeline/harmonization.py
```


The pipeline applies **ComBat harmonization** using the Python package:
```
neuroHarmonize
```


Harmonization removes **site/scanner effects** while preserving biological variability.

Both datasets are harmonized:

- trophic levels
- effective connectivity matrices

---

# 2. Graph Metrics from Effective Connectivity

Module:
```
src/functions_pipeline/metrics.py
```


For each subject's GEC matrix, the pipeline computes:

### Degree-based metrics

- **In-degree**
- **Out-degree**
- **Out/In degree ratio**

### Network topology metrics

- **Clustering coefficient**
- **Mean path length**
- **Betweenness centrality**

Libraries used:

- `networkx`
- `bctpy`

---

# 3. Regression Analysis

The pipeline evaluates the relationship between **trophic hierarchy and connectivity topology**.

A multiple linear regression is performed:
```
TL ~ out/in degree ratio
clustering
path length
betweenness
```
Using:
```
statsmodels OLS
```

Metrics are flattened across **subjects and regions** before regression.

---

# 4. Visualization

Module:
```
src/functions_pipeline/plotting.py
```

The following figures are generated.

### Trophic level distribution
```
Histogram.png
```

Brain regions are categorized into three groups:

- **Sinks** (lowest TL)
- **Intermediate**
- **Sources** (highest TL)

based on percentile thresholds (33% and 66%).

---

### Correlation plots

Scatter plots showing the relationship between trophic levels and connectivity metrics:
```
OutInRatio.png
MeanPathLength.png
```

Each point represents a **region-subject pair**, while larger colored points indicate **regional means**.

---

# Running the Pipeline

Execute:
```
python main_pipeline.py
--data-dir /path/to/data
--out-dir /path/to/results
```

Example:
```
python main_pipeline.py
--data-dir data/amyloid_negative
--out-dir results/characterization
```



