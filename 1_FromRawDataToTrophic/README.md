# From Raw Data to Trophic Levels

This directory contains the full pipeline used to compute **generative effective connectivity (GEC)** and **trophic hierarchy measures** from raw neuroimaging data.

The workflow starts from raw fMRI and diffusion MRI data and ends with the computation of **trophic levels and trophic coherence** derived from effective connectivity networks.

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

