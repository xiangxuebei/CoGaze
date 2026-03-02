# CoGaze: Closed-Loop Eye-Face Alignment for Multi-Center Mobile Gaze-Based Cognitive Screening in Older Adults

## Abstract

Mobile screening is increasingly needed for frequent, low-burden assessment of cognitive impairment in communities and households, and eye tracking offers scalable oculomotor signatures for this purpose. However, mobile gaze estimation is fragile under coupled shifts from aging-related appearance changes, uncontrolled capture noise, and fluctuating engagement, which can corrupt downstream oculomotor features.  
We propose CoGaze, a closed-loop gaze reasoning framework for mobile videos of older adults. CoGaze formulates gaze estimation as structured alignment between ocular appearance and facial geometric context. It stabilizes ocular representations with a prototype-anchor dictionary retrieved by normalized cosine attention, refines eye-face evidence via reciprocal cross-attention, and enforces temporal alignment using differential spatiotemporal attention with a segment-level consistency objective to reduce jitter and drift without blurring true saccades.  
We collect a deployment-oriented multi-center dataset from 7 sites (July-December 2025) with 119 gaze-task subjects and 114 clinically labeled assessments. CoGaze achieves mean error of 1.29 cm/2.03 cm on phone/tablet and improves screening to 85.1% accuracy (5-fold CV).

## Project Overview

CoGaze is a two-stage framework designed for mobile scenarios involving older adults:

1. `gaze stage`: predicts the gaze trajectory (PoR) from face/eye image sequences.
2. `screening stage`: extracts oculomotor features from gaze trajectories and performs subject-level cognitive screening (e.g., NCI/MCI/Dementia).

The core modules include multi-scale face representation, ocular prior dictionary enhancement, bidirectional Eye-Face Cross-Attention (EFCA), and differential spatiotemporal modeling (ADM).

## Environment Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Recommended versions: Python `>=3.10`, PyTorch `>=2.0`.

## Evaluation Metrics

### Gaze Estimation

- `fix_err_cm`, `dot_err_cm`, `rmse_cm`
- `pix100_acc`, `pix200_acc`

### Cognitive Screening

- `acc`, `f1`
- `macro_auc`, `micro_auc`
- `sensitivity`, `specificity`
