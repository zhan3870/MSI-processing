# Image Denoising Script

This Python script performs image denoising on large TIFF images using a pre-trained TensorFlow model from N2N denoising. It splits the input image into smaller patches, processes each patch using the model, and then stitches the denoised patches back together to produce a final output image.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [How It Works](#how-it-works)

## Features

- Denoises large TIFF images using a TensorFlow model.
- Handles large images by splitting them into smaller patches.
- Blends the denoised patches seamlessly to create the final image.
- Supports customizable overlap for patch blending.

## Requirements

- Python 3.x
- TensorFlow 2.x (with TensorFlow 1.x compatibility mode)
- NumPy
- PIL (Python Imaging Library)
- tifffile (for handling TIFF files)
- glob, re, os, time (standard Python libraries)

## Usage

To run the script, use the following command:

```bash
python denoise.py input_image.tif,output_directory/
```

- **`input_image.tif`**: The path to the TIFF image you want to denoise.
- **`output_directory/`**: The directory where the denoised image will be saved.

**Note**: The input file must be a **single-layer** TIFF image (2D).

### Example

```bash
python denoise.py /path/to/input_image.tif,/path/to/output_directory/
```

The script will output the denoised image in the specified output directory with the same filename as the input image, but with a `.tif` extension.

## How It Works

1. **Model Loading**: The script loads a pre-trained TensorFlow model from a `.pb` file.
2. **Image Preprocessing**: The input image is split into smaller patches with padding to ensure that each patch is of the correct size for the model.
3. **Denoising**: Each patch is processed by the model to remove noise.
4. **Blending**: The denoised patches are stitched back together, with overlapping regions blended smoothly to avoid visible seams.
5. **Output**: The final denoised image is saved as a TIFF file in the specified output directory.
