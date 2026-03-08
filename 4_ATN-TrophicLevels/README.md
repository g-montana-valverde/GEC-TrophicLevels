# ATN Pathology and Trophic Levels

This directory contains the analysis pipeline used to investigate the relationship between **trophic hierarchy organization** and **ATN biomarkers** across the Alzheimer’s disease spectrum.

The analysis evaluates how **Amyloid-β (A), Tau (T), and Neurodegeneration (N)** relate to **trophic levels (TL)** at both:

- **regional level**
- **functional network level**

using **ridge-regularized linear mixed-effects models**.

---

# Directory Structure
```
4_ATN-TrophicLevels/

├── src/
│
│ ├── data/
│ │ ├── dbs80_labels.txt
│ │ └── reg2net.json
│ │
│ └── functions_pipeline/
│ ├── init.py
│ ├── harmonization.py
│ ├── LME_Ridge.py
│
└── main_pipeline.py
```


---

# Overview of the Pipeline

The pipeline performs the following steps:
```
Load trophic levels and ATN biomarkers
↓
Harmonize site/scanner effects (ComBat)
↓
Aggregate regions into functional networks
↓
Ridge-regularized mixed-effects modeling
↓
FDR-corrected statistical testing
↓
Visualization of significant relationships
```

---

# Input Data

The pipeline expects the following datasets.

## Trophic Levels
```
TL_ADNI3_<GROUP>_dbs80.mat
```

Matrix: subjects x regions


Computed from the previous pipeline: 1_FromRawDataToTrophic/


---

## Amyloid-β PET
```
ADNI3_AmyloidBeta-PET_Centiloids.csv
```

Regional amyloid burden measured in **Centiloid units**.

---

## Tau PET
```
ADNI3_Tau-PET_SURV.csv
```

Regional tau deposition values.

---

## Structural MRI
```
GMV.csv
```

Regional **gray matter volume (GMV)** used as the **N (neurodegeneration)** component of the ATN framework.

---

## Demographics
```
Demographics.xlsx
```

Contains:

- Age
- Gender
- Education
- APOE4 carrier status
- Subject ID

---

## Scanner / Site Information
```
Sites.xlsx
```

Used to harmonize measurements across imaging sites.

---

## Region–Network Mapping
```
src/data/reg2net.json
```

Maps brain regions to **large-scale functional networks**.

Networks used:

- VN – Visual Network
- SMN – Somatomotor Network
- DAN – Dorsal Attention Network
- SAN – Salience Network
- LN – Limbic Network
- CN – Control Network
- DMN – Default Mode Network

---

# 1. Data Harmonization

Module:
```
src/functions_pipeline/harmonization.py
```


The pipeline removes **site and scanner effects** using **ComBat harmonization** via:
```
neuroHarmonize
```

Harmonization is applied to:

- trophic levels
- amyloid PET
- tau PET
- gray matter volume

while preserving biological variability.

---

# 2. Network-Level Aggregation

Regional measures are averaged within **functional networks** defined in:
```
reg2net.json
```

Subcortical regions are excluded:
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

This produces **network-level trophic levels and ATN biomarkers**.

---

# 3. Ridge-Regularized Mixed-Effects Model

Module:
```
src/functions_pipeline/LME_Ridge.py
```

The pipeline uses a **ridge regression model with cross-validation** combined with **linear mixed-effects testing**.

## Fixed effects
```
TrophicLevel ~ ABeta + Tau + Volume + age + gender + edu + APOE4
```

## Random effects
```
SubjectID
```


This model accounts for:

- repeated measurements across brain regions
- inter-subject variability

---

# 4. Region-Specific Significance Tests

After ridge regression identifies the best regularization parameter, **region-wise mixed-effects models** are estimated.

Statistical significance is evaluated for:

- **Aβ (Amyloid)**
- **Tau**
- **Volume (Neurodegeneration)**

across all brain regions.

---

# 5. Multiple Comparison Correction

P-values are corrected using **False Discovery Rate (FDR)**:
```
Benjamini–Hochberg procedure
```


---

# 6. Visualization

For each significant region, the pipeline generates scatter plots showing Trophic Level vs ATN biomarker


Plots include:

- regression lines derived from mixed-effects models
- group-specific subject points
- significance markers

Special visualization is generated for **bilateral hippocampus volume effects**.

---

# Output Files

Results are saved in: <out_dir>


---

# Running the Pipeline

Execute:
```
python main_pipeline.py
--data-dir /path/to/data
--deriv-dir /path/to/derivatives
--out-dir /path/to/results
```

Example:
```
python main_pipeline.py
--data-dir data/ATN
--deriv-dir derivatives
--out-dir results/ATN_TL
```
