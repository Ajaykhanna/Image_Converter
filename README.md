# Streamlit Multi-Format Image Converter

Developer: Ajay Khanna
Place: LANL
Date: May 12, 2025

## 🌟 Overview

This Streamlit web application provides a user-friendly interface for converting multiple image files between various formats. It supports common raster image types (PNG, JPG, TGA, BMP, GIF, WEBP) and optionally offers SVG input conversion if the necessary dependencies are met. Users can upload files via drag-and-drop, preview their selections, remove individual files from the queue, and apply resizing or rescaling options before conversion. Converted images are then made available for download, either individually (for single successful conversions) or as a collective ZIP archive.

## ✨ Features

- **Multi-Format Support**: Convert between multiple image formats, including PNG, JPG, TGA, BMP, GIF, WEBP, and SVG (if CairoSVG is installed).
    **Multiple Input Formats**: Supports PNG, JPG/JPEG, TGA (Targa), BMP, GIF, WEBP.

    **SVG Input Support (Optional)**: Converts SVG vector files to selected raster output formats if CairoSVG is installed.

    **Multiple Output Formats**: Convert to PNG (default), JPEG, WEBP, BMP, GIF, TGA.

    **Multi-File Upload**: Easy drag-and-drop interface for uploading multiple images at once.

    **Image Preview Queue**:

        View thumbnails of uploaded images.

        Remove individual files from the conversion queue.

        Option to clear the entire upload queue.

    **Resizing & Rescaling**:

        Resize: Specify exact target pixel dimensions (width & height).

        Rescale: Adjust image size by a percentage.

        Choice of resampling filters (e.g., Lanczos, Bicubic) for quality control during raster image resizing.

    **User-Friendly Interface**: All options are conveniently located in a sidebar.

    **Error Handling**: Gracefully handles common issues like corrupted files or unsupported formats.

    **Download Options**:

        Download a single converted file directly if only one image is processed.

        Download all successfully converted images as a single ZIP archive.

    **Modular Code Structure**: Organized into functions for UI, processing, and helpers for better readability and maintenance.

## 🚀 Live Demo

Try the live application hosted on Streamlit Community Cloud:
🔗 Launch Image Converter App

## 🛠️ Prerequisites

    Python 3.7+

    pip (Python package installer)

## ⚙️ Installation & Setup

    Clone the Repository (if applicable):

    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

    Create a Virtual Environment (Recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install Dependencies:
    A requirements.txt file should be included in the repository.

```bash
pip install -r requirements.txt
```
If you don't have a requirements.txt yet, you can create one from the main dependencies:

### Requirements.txt
```
streamlit
Pillow
# CairoSVG is optional, see below

```
And then install: pip install streamlit Pillow

## ▶️ How to Run

Navigate to the directory where you saved the image_converter_app.py file (or the cloned repository).

Run the Streamlit application from your terminal:
```bash
streamlit run image_converter_app.py
```
The application will open in your default web browser.

## 🔮 Optional: Enabling SVG Support

For converting .svg files as input, you need to install CairoSVG and its system-level dependencies.

**Install CairoSVG Python package**:
```
pip install CairoSVG
```

**Install System Dependencies for Cairo**:
```
macOS (using Homebrew):

brew install cairo

Debian/Ubuntu Linux:

sudo apt-get update
sudo apt-get install libcairo2-dev

Fedora Linux:

sudo yum install cairo-devel

Windows: This can be more complex. You may need to install GTK+ for Windows or find pre-compiled binaries for Cairo. Refer to the CairoSVG documentation or community resources for specific Windows installation guidance.
```
If CairoSVG or its dependencies are not found, the application will still run but SVG input functionality will be disabled, and a message will be printed to your console.

Enjoy converting your images!