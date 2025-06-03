# FAISS Index Builder

This script builds a FAISS index for efficient similarity search using DINOv2 features.

## Arguments

### Model Configuration
- `--backbone` (str, default: 'vits14')
  - Specifies the DINOv2 backbone model size
  - Choices: 'vits14', 'vitb14', 'vitl14', 'vitg14'
  - 'vits14': Small model (21M parameters)
  - 'vitb14': Base model (86M parameters)
  - 'vitl14': Large model (300M parameters)
  - 'vitg14': Giant model (1.1B parameters)

- `--register` (bool, default: True)
  - Determines whether to use the model with registers
  - When True, uses the model with additional register tokens
  - When False, uses the standard model without registers

### Data Configuration
- `--num-data` (int, default: 3)
  - Specifies the number of data samples to use for building the FAISS index

- `--data-path` (str, default: 'BME_Capstone1/FAISS')
  - Path to the directory containing the data for building the index
  - Should contain the image data organized by slice direction

- `--slice` (str, default: 'axial') ‼
  - Specifies the direction of slices to use for building the index
  - Common values: 'axial', 'coronal', 'sagittal'
  - ‼ ONLY "axial" is implemented ‼

## Usage Example

```bash
python build_faiss.py \
    --backbone vits14 \
    --register True \
    --num-data 3 \
    --data-path /path/to/data \
    --slice axial
```