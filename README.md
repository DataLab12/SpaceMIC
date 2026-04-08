# SpaceMIC
Automated Space Micorbial-Induced Corrosion (MIC) Detection 

## Overview

This repository contains the implementation for a computer-vision pipeline for microbial-induced corrosion (MIC) region segmentation in scanning electron microscopy (SEM) imagery.

The project benchmarks classical and deep learning methods for MIC segmentation and includes two proposed deep learning models:
- Enhanced SAM2
- Lightweight PH-FPNSeg (Prompt- and Heatmap-Guided FPN-based Lightweight Segmentation Model)

## Motivation
Microbial-induced corrosion (MIC) threatens the structural integrity of metals in critical environments, including aerospace and terrestrial infrastructure.  Accurate and efficient segmentation of MIC regions in SEM imagery can support monitoring and maintenance workflows.

## Contributions
- An expanded, expertly annotated dataset of 331 SEM images for MIC region segmentation
- A benchmarking pipeline for MIC region segmentation in SEM images
- Evaluation of both classical and deep learning approaches
- An enhanced SAM2 model for MIC segmentation
- A Lightweight PH-FPNSeg model for MIC segmentation

## Dataset

The experiments use our proposed annotated SEM dataset of microbial-induced corrosion on stainless steel.

Dataset access: https://doi.org/10.18738/T8/0LQVXW

## Results

The enhanced SAM2 and Lightweight PH-FPNSeg provide strong segmentation performance, achieving average Dice and IoU scores of approximately 82% and 70%, respectively.

## Repository Structure

```text
.
├── Classicial_CLF_method/        # Proposed classical method Clustering with Local Features
├── DeepLabV3+/                   # DeepLabV3+ deep learning baseline
├── SAM2_model/                   # Enhanced Segment Anything Model 2 (deep learning)
├── sam_original/                 # Segment Anything Model (SAM) deep learning baseline
├── LightweightPH_FPNSeg_model/   # Proposed PH-FPNSeg model (deep learning)
```

## Usage
Each method is self-contained. Refer to the README file inside folders for instructions.

## Notes

This repository accompanies a manuscript currently under review.

## Citation
If you use this repository or dataset, please cite:

```bibtex
@data{T8/0LQVXW_2026,
author = {McLean, Robert and Tešić, Jelena and Amber Dowell Busboom and Akter Tani, Tanzina and Thornhill, Starla},
publisher = {Texas Data Repository},
title = {{SpaceMIC}},
year = {2026},
version = {V1},
doi = {10.18738/T8/0LQVXW},
url = {https://doi.org/10.18738/T8/0LQVXW}
}

@article{anonymous2026mic,
title={Benchmarking Automated MIC Detection: The AI-Ready Space Biology SEM Dataset and Advanced Detection Methods},
author={Akter Tani, Tanzina and Amber Dowell Busboom and McLean, Robert and Tešić, Jelena},
journal={Under Review},
year={2026} }
(Replace this with the final paper citation and DOI once available.)
