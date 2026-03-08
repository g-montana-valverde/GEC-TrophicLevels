# From Raw Data to Trophic Levels

This directory contains the full pipeline used to compute **generative effective connectivity (GEC)** and **trophic hierarchy measures** from raw neuroimaging data.

The workflow starts from raw fMRI and diffusion MRI data and ends with the computation of **trophic levels and trophic directedness** derived from effective connectivity networks.

The pipeline consists of four main stages:

1. Functional MRI preprocessing
2. fMRI time series denoising and extraction
3. Structural connectome reconstruction
4. Generative effective connectivity and trophic level estimation

---

# Pipeline Overview

Raw MRI data  
↓  
**fMRIPrep preprocessing**  
↓  
**Denoised regional time series extraction**  
↓  
**Structural connectome reconstruction (MRtrix3)**  
↓  
**Generative Effective Connectivity (Hopf model)**  
↓  
**Trophic levels and coherence**

---

# Directory Structure
1_FromRawDataToTrophic/

├── Preprocessing/
│ └── fMRIprep_group.sh
│
├── Denoising/
│ ├── SingleSubject_TimeSeries_Denoising.py
│ └── Group_Denoised_TimeSeries.py
│
├── StructuralConnectome/
│ ├── Step1_DWI_preproc_and_streamlines.sh
│ └── Step2_CreateConnectome.sh
│
└── TrophicLevels/
├── Step1_fdiff_group.m
└── Step2_dbs80_par.m



---

# 1. Functional MRI Preprocessing

Script:

```
Preprocessing/fMRIprep_group.sh
```

Preprocessing is performed using **fMRIPrep 21.0.1** inside a Singularity container and executed as a **SLURM job array**.

Main steps performed by fMRIPrep:

- BIDS validation
- motion correction
- susceptibility distortion correction
- spatial normalization to MNI152NLin2009cAsym
- brain masking
- confound estimation

### Inputs

BIDS dataset directory:

```
$HOME/ADNI_dataset/
```

### Outputs

Preprocessed dataset:

```
$HOME/ADNI3_preprocessed/
```

---

# 2. Time Series Denoising and Extraction

## Single Subject Denoising

Script:

```
Denoising/SingleSubject_TimeSeries_Denoising.py
```

This script:

1. Loads preprocessed fMRI data
2. Applies confound regression using **nilearn**
3. Removes motion-contaminated volumes (scrubbing)
4. Extracts regional time series using the **dbs80 atlas**
5. Standardizes signals

Confound strategy:

- high-pass filtering
- motion parameters
- WM/CSF regressors
- scrubbing (FD > 0.5 mm)

Output per subject:

```
sub-XX_dbs80_denoised.mat
```

containing:

```
ts : regional time series
TR : repetition time
```

---

## Group Time Series Assembly

Script:

```
Denoising/Group_Denoised_TimeSeries.py
```

This script aggregates all individual subject time series into a single MATLAB structure:

```
ADNI3_GROUP_dbs80.mat
```

Structure:

```
data{subject} → ROI × time
```

---

# 3. Structural Connectome Reconstruction

Directory:

```
StructuralConnectome/
```

Processing is performed using **MRtrix3**.

## Step 1 — DWI preprocessing and tractography

Script:

```
Step1_DWI_preproc_and_streamlines.sh
```

Steps:

1. Convert DWI to `.mif`
2. Denoising
3. Eddy current and motion correction
4. Bias field correction
5. Brain mask generation
6. Multi-tissue response function estimation
7. Fiber orientation distribution (FOD) estimation
8. Anatomically constrained tractography (ACT)
9. Generate **5 million streamlines**
10. Apply **SIFT2 weighting**

Output:

```
tracks_SUB_5M.tck
sift_SUB_5M.txt
```

---

## Step 2 — Structural Connectivity Matrix

Script:

```
Step2_CreateConnectome.sh
```

This script:

1. Aligns the **dbs80 atlas** to diffusion space
2. Generates the structural connectome using `tck2connectome`

Output:

```
SUB_connectome_dbs80.csv
```

These matrices are later combined into:

```
GROUP_connectome.mat
```

which is used as the structural constraint for the GEC model.

---

# 4. Generative Effective Connectivity and Trophic Levels

Directory:

```
TrophicLevels/
```

MATLAB is used to estimate effective connectivity using the **Hopf whole-brain model**.

---

## Step 1 — Intrinsic Frequency Estimation

Script:

```
Step1_fdiff_group.m
```

Steps:

1. Band-pass filtering (0.008–0.08 Hz)
2. Power spectrum estimation
3. Dominant frequency extraction for each region

Output:

```
fdiff_GROUP_dbs80.mat
```

---

## Step 2 — Generative Effective Connectivity

Script:

```
Step2_dbs80_par.m
```

This script:

1. Fits the **Hopf model** to empirical functional connectivity
2. Uses the structural connectome as anatomical constraint
3. Estimates **Generative Effective Connectivity (GEC)** for each subject

Optimization targets:

- functional connectivity (FC)
- time-lagged covariance (COVτ)

Output:

```
Ceff (subject-level GEC matrices)
```

---

## Step 3 — Trophic Level Computation

From the effective connectivity matrix \(C_{eff}\):

1. Compute trophic levels
2. Compute trophic directedness

Definitions:

```
Lambda = diag(u) - A - A'
gamma = Lambda^{-1} v
```

Outputs:

```
TL_GROUP_dbs80.mat
```

containing:

```
hierarchicallevels
directedness
```

---

# Expected Final Outputs

The pipeline produces:

- Structural connectivity matrices
- Generative effective connectivity matrices
- Trophic hierarchy measures

Key outputs:

```
TL_GROUP_dbs80.mat
fdiff_GROUP_dbs80.mat
GROUP_connectome.mat
```


```
Run order:

1_Preprocessing/fMRIprep_group.sh
2_Denoising/SingleSubject_TimeSeries_Denoising.py
2_Denoising/Group_Denoised_TimeSeries.py
3_StructuralConnectome/Step1_DWI_preproc_and_streamlines.sh
3_StructuralConnectome/Step2_CreateConnectome.sh
4_TrophicLevels/Step1_fdiff_group.m
4_TrophicLevels/Step2_dbs80_par.m
```

---

