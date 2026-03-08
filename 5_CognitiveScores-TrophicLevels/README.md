# Cognitive Scores and Trophic Levels

This directory contains the analysis pipeline used to investigate the relationship between **brain trophic hierarchy organization** and **cognitive performance** across the Alzheimer's disease spectrum.

The pipeline evaluates how **regional trophic levels** relate to cognitive scores using **multiple linear regression models controlling for demographic covariates**.

Analyses are performed at two spatial scales:

- **regional level**
- **functional network level**

---

# Directory Structure
```
5_CognitiveScores-TrophicLevels/

├── src/
│
│ ├── data/
│ │ ├── dbs80_labels.txt
│ │ └── reg2net.json
│ │
│ └── functions_pipeline/
│ ├── init.py
│ ├── assessments_filtering.py
│ ├── MLR_assessments.py
│
└── main_pipeline.py
```


---

# Overview of the Pipeline

The analysis pipeline performs the following steps:

```
Load trophic levels
↓
Filter and preprocess cognitive assessments
↓
Harmonize trophic levels across imaging sites
↓
Aggregate regions into functional networks
↓
Run region-wise multiple linear regression models
↓
Apply FDR correction for multiple comparisons
↓
Generate statistical brain maps and figures
```


---

# Input Data

The pipeline requires the following datasets.

---

# 1. Trophic Levels
```
TL_ADNI3_<GROUP>_dbs80.mat
```

These values must be generated using the pipeline 1_FromRawDataToTrophic


---

# 2. Cognitive Assessments

The pipeline supports several neuropsychological tests from ADNI.

### ADAS-Cog
```
ADAS_original.csv
```

Filtered scores:

- **ADAS-Cog-11**
- **ADAS-Cog-13**

---

### Clinical Dementia Rating
```
CDR_original.csv
```

Filtered scores:

- **CDR-G** (global score)
- **CDR-SB** (sum of boxes)

---

### Montreal Cognitive Assessment
```
MoCA_original.csv
```

Computed cognitive domains include:

- MoCA total score
- Visuospatial ability
- Naming
- Attention
- Language
- Abstraction
- Delayed recall
- Orientation

---

### Mini-Mental State Examination
```
MMSE_original.csv
```

Computed domains include:

- Language
- Orientation
- Registration
- Attention/Calculation
- Recall
- Total MMSE score

---

# 3. Demographic Data
```
Demographics.xlsx
```

Includes:

- Subject ID
- Age
- Gender
- Education
- APOE4 carrier status

---

# 4. Scanner / Site Information
```
Sites.xlsx
```

Used for **ComBat harmonization** of trophic levels.

---

# 5. Region–Network Mapping
```
src/data/reg2net.json
```

Defines the mapping between brain regions and **functional networks**.

Networks used:

- VN – Visual Network
- SMN – Somatomotor Network
- DAN – Dorsal Attention Network
- SAN – Salience Network
- LN – Limbic Network
- CN – Control Network
- DMN – Default Mode Network

---

# 1. Cognitive Assessment Filtering

Module:
```
src/functions_pipeline/harmonization.py
```

Trophic levels are harmonized across imaging sites using **ComBat harmonization** implemented in **neuroHarmonize**

Covariates used during harmonization include:

- age
- gender
- education
- scanner/site information

---

# 3. Network-Level Aggregation

Regional trophic levels are averaged within functional networks defined in **reg2net.json**


Subcortical regions are excluded from network averages:
```
hippocampus
amygdala
thalamus
caudate
accumbens
putamen
gpe
gpi
stn
```


---

# 4. Multiple Linear Regression Analysis

Module:
```
src/functions_pipeline/MLR_assessments.py
```

For each brain region, the following regression model is estimated:
```
CognitiveScore ~ TrophicLevel + age + gender + education + APOE4
```

This analysis tests whether **regional trophic hierarchy position predicts cognitive performance**.

---

# 5. Multiple Comparison Correction

To control for testing across many regions, **False Discovery Rate (FDR)** correction is applied using:

```
Benjamini–Hochberg procedure
```


---

# Running the Pipeline

Execute the main pipeline using:
```
python main_pipeline.py
--data-dir /path/to/data
--deriv-dir /path/to/derivatives
--out-dir /path/to/results
```

Example:

```
python main_pipeline.py
--data-dir data/cognitive
--deriv-dir derivatives
--out-dir results/Cognition_TL
```