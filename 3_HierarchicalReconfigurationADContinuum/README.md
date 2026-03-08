# Hierarchical Reconfiguration Across the Alzheimer's Disease Continuum

This directory contains the analysis pipeline used to investigate **changes in brain hierarchical organization across the Alzheimer's disease continuum**.

The analysis compares trophic hierarchy metrics across four groups:

* HCn — amyloid-negative healthy controls
* HCp — amyloid-positive healthy controls
* MCIp — mild cognitive impairment with amyloid-positive
* ADp — Alzheimer's disease patients with amyloid-positive

The pipeline evaluates **regional trophic levels** and **global directedness**, and performs **group comparisons using permutation-based statistical tests controlling for demographic covariates**.

---

# Directory Structure

```
3_HierarchicalReconfigurationADContinuum/

├── src/
│   ├── data/
│   │   ├── dbs80_label.txt
│   │   └── reg2net.json
│   │
│   └── functions_pipeline/
│       ├── __init__.py
│       ├── harmonization.py
│       ├── Permutation_group_comparisons.py
│       └── plotting.py
│
└── main_pipeline.py
```

---

# Overview of the Pipeline

Input data:

* trophic levels per subject and brain region
* directedness values per subject
* demographic covariates
* site/scanner information

Processing workflow:

```
Load trophic levels and directedness
↓
Harmonization across acquisition sites
↓
Compute network-level trophic levels
↓
Permutation-based group comparisons
↓
Visualization of hierarchical reconfiguration
```

---

# Input Data

The pipeline expects the following files in the input directory.

```
TL_ADNI3_GROUP_dbs80.mat
DIR_ADNI3_GROUP_dbs80.mat
Demographics.xlsx
Sites.xlsx
dbs80_labels.txt
reg2net.json
```

### File descriptions

**TL_ADNI3_GROUP_dbs80.mat**

```
TL : subjects × regions
```

Regional trophic levels computed from effective connectivity.

---

**DIR_ADNI3_GROUP_dbs80.mat**

```
directedness : subjects
```

Global directedness metric derived from trophic hierarchy.

---

**dbs80_labels.txt**

Mapping between atlas indices and brain region names.

---

**reg2net.json**

Mapping between atlas regions and large-scale functional networks:

* VN – Visual Network
* SMN – Somatomotor Network
* DAN – Dorsal Attention Network
* SAN – Salience Network
* LN – Limbic Network
* CN – Control Network
* DMN – Default Mode Network

---

# 1. Data Harmonization

Module:

```
src/functions_pipeline/harmonization.py
```

Harmonization is performed using **ComBat** through the Python package:

```
neuroHarmonize
```

The following variables are harmonized:

* trophic levels (regional values)
* directedness (subject-level metric)

Covariates used in harmonization:

* age
* gender
* education
* acquisition site

---

# 2. Network-Level Trophic Levels

Regional trophic levels are aggregated into **functional networks**.

The mapping between regions and networks is defined in:

```
reg2net.json
```

For each subject, the trophic level of a network is defined as the **mean trophic level of its cortical regions**.

Subcortical structures are excluded:

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

# 3. Group Comparisons

Statistical comparisons are performed between all pairs of groups:

```
HCn vs HCp
HCn vs MCIp
HCn vs ADp
HCp vs MCIp
HCp vs ADp
MCIp vs ADp
```

Tests are implemented in:

```
src/functions_pipeline/Permutation_group_comparisons.py
```

### Statistical model

For each brain region (or network), the following regression model is evaluated:

```
metric ~ group + age + gender + education
```

Significance is assessed using **permutation testing**.

Key characteristics:

* 10,000 permutations
* Freedman–Lane permutation scheme
* two-sided tests
* FDR correction across regions

Outputs:

* t-statistics
* corrected p-values

---

# 4. Visualization

Module:

```
src/functions_pipeline/plotting.py
```

The following figures are produced.

---

## Directedness Group Comparison

```
DirectednessGroupComparison.png
```

Visualization of directedness differences between groups using **RainCloud plots**.

Statistical significance between group pairs is displayed above the distributions.

---

## Network-Level Trophic Level Comparison

```
NetworksGroupComparison.png
```

RainCloud plots showing trophic levels across functional networks for each diagnostic group.

Significant group differences are indicated with significance markers.

---

# 5. Brain Rendering Outputs

Regional group differences are exported as MATLAB files for brain surface visualization.

Example output:

```
RegionsGroupComparison_HCn_ADp.mat
```

These files contain regional statistical maps for rendering in external visualization software.

---

# Running the Pipeline

Execute:

```
python main_pipeline.py \
    --data-dir /path/to/data \
    --deriv-dir /path/to/derivatives \
    --out-dir /path/to/results
```

Example:

```
python main_pipeline.py \
    --data-dir data/adni \
    --deriv-dir derivatives/adni \
    --out-dir results/hierarchy_reconfiguration
```

---

# Output Files

The pipeline generates:

```
DirectednessGroupComparison.png
NetworksGroupComparison.png
RegionsGroupComparison_GROUPA_GROUPB.mat
```

These outputs summarize **hierarchical reconfiguration across the Alzheimer's disease continuum**.





