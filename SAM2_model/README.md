# SAM2_Model Setup and Running Instructions

To run the SAM2_model, users need to use official SAM2 https://github.com/facebookresearch/sam2.

## Installation

1. Clone the git with the command:
   ```bash
   git clone https://github.com/facebookresearch/sam2.git sam2_repo
   ```

2. Inside the sam2_repo, use command:
   ```bash
   pip install -e .
   ```

3. Download the checkpoint using the command:
   ```bash
   cd checkpoints && \
   ./download_ckpts.sh && \
   cd ..
   ```

That's all.

## Prerequisites

Note: to use SAM2, you will need Python 3.10.

## Running the MIC-Based Modified SAM2_Model

Finally, to run the MIC-based modified SAM2_Model, use the slum script, or just run:
- For training:
  ```bash
  python train_sam2.py
  ```
- For testing:
  ```bash
  python test_sam2.py
  ```
