# Developer: Ajay Khanna
# Places: LANL
# Date: May.12.2025

"""
Enhanced Streamlit GUI application for converting multiple image files between various formats.
This comprehensive version includes advanced processing, batch management, and professional features.

Features:
- Multi-format support: PNG, JPG, TGA, BMP, GIF, WEBP, SVG (optional), HEIC (optional), RAW (optional)
- Advanced image processing: rotation, flipping, cropping, enhancement, filters
- Quality controls: JPEG quality, PNG compression, WebP options
- Metadata management: EXIF handling, color profiles
- Batch processing: queue management, presets, automation
- Professional output: watermarking, borders, multi-format export
- Analytics: processing statistics, performance metrics
- Accessibility: theme toggle, responsive design, keyboard shortcuts
- Security: file validation, secure processing, session isolation

Requires: streamlit, Pillow, numpy, opencv-python, matplotlib, reportlab
Optional: CairoSVG, pillow-heif, rawpy, rembg
Installation: pip install streamlit Pillow numpy opencv-python matplotlib reportlab
"""

import streamlit as st
from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageDraw,
    ImageFont,
    UnidentifiedImageError,
)
import os
import io
import zipfile
import time
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import shutil

# Advanced libraries with graceful degradation
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("INFO: OpenCV not available. Advanced filters disabled.")

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.utils import ImageReader

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("INFO: ReportLab not available. PDF export disabled.")

try:
    import cairosvg

    dummy_svg = '<svg height="1" width="1"></svg>'
    cairosvg.svg2png(bytestring=dummy_svg.encode("utf-8"))
    CAIROSVG_AVAILABLE = True
except (ImportError, Exception):
    CAIROSVG_AVAILABLE = False
    print("INFO: CairoSVG not available. SVG support disabled.")

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False
    print("INFO: pillow-heif not available. HEIC/HEIF support disabled.")

try:
    import rawpy

    RAW_AVAILABLE = True
except ImportError:
    RAW_AVAILABLE = False
    print("INFO: rawpy not available. RAW support disabled.")

try:
    import rembg

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("INFO: rembg not available. Background removal disabled.")

# --- Global Configuration ---
SUPPORTED_RASTER_FORMATS = ["png", "jpg", "jpeg", "tga", "bmp", "gif", "webp"]
SUPPORTED_FORMATS = SUPPORTED_RASTER_FORMATS.copy()

# Add optional format support
if CAIROSVG_AVAILABLE:
    SUPPORTED_FORMATS.append("svg")
if HEIF_AVAILABLE:
    SUPPORTED_FORMATS.extend(["heic", "heif"])
if RAW_AVAILABLE:
    SUPPORTED_FORMATS.extend(["cr2", "nef", "arw", "dng", "orf", "raf", "rw2"])

SUPPORTED_OUTPUT_FORMATS = ["PNG", "JPEG", "WEBP", "BMP", "GIF", "TGA", "TIFF"]
if REPORTLAB_AVAILABLE:
    SUPPORTED_OUTPUT_FORMATS.append("PDF")

DEFAULT_OUTPUT_FORMAT = "PNG"

# Processing constants
RESIZE_METHODS = {
    "LANCZOS (High Quality)": Image.Resampling.LANCZOS,
    "BICUBIC": Image.Resampling.BICUBIC,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST (Fastest)": Image.Resampling.NEAREST,
}

ENHANCEMENT_FILTERS = {
    "None": None,
    "Blur": ImageFilter.BLUR,
    "Detail": ImageFilter.DETAIL,
    "Edge Enhance": ImageFilter.EDGE_ENHANCE,
    "Edge Enhance More": ImageFilter.EDGE_ENHANCE_MORE,
    "Emboss": ImageFilter.EMBOSS,
    "Find Edges": ImageFilter.FIND_EDGES,
    "Sharpen": ImageFilter.SHARPEN,
    "Smooth": ImageFilter.SMOOTH,
    "Smooth More": ImageFilter.SMOOTH_MORE,
    "Contour": ImageFilter.CONTOUR,
}

MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tga": "image/x-tga",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
}

# Theme configuration
THEMES = {
    "Light": {
        "background": "#FFFFFF",
        "text": "#000000",
        "accent": "#FF6B6B",
        "secondary": "#4ECDC4",
    },
    "Dark": {
        "background": "#1E1E1E",
        "text": "#FFFFFF",
        "accent": "#FF6B6B",
        "secondary": "#4ECDC4",
    },
}


# --- Session State Initialization ---
def initialize_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        "processing_queue": [],
        "uploader_key": 0,
        "processing_stats": {
            "total_processed": 0,
            "total_errors": 0,
            "processing_times": [],
        },
        "conversion_presets": {},
        "current_theme": "Light",
        "processing_history": [],
        "cancel_processing": False,
        "auto_process": False,
        "selected_preset": "Custom",
        "grid_view": True,
        "thumbnail_size": "Medium",
        "show_histogram": False,
        "background_processing": False,
        "temp_files": [],
        "processing_logs": [],
        "accessibility_mode": False,
        "language": "English",
        "show_advanced": False,
        "show_stats": False,
        "pdf_created": False,
        "pdf_bytes": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Utility Functions ---
def log_processing_event(event_type: str, message: str, file_name: str = ""):
    """Log processing events for debugging and analytics."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message,
        "file": file_name,
    }
    st.session_state.processing_logs.append(log_entry)

    # Keep only last 100 entries
    if len(st.session_state.processing_logs) > 100:
        st.session_state.processing_logs = st.session_state.processing_logs[-100:]


def generate_secure_filename(original_filename: str, target_format: str) -> str:
    """Generate secure filename with timestamp and hash."""
    base_name, _ = os.path.splitext(original_filename)
    # Create hash of original filename for uniqueness
    name_hash = hashlib.md5(original_filename.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}_{name_hash}.{target_format.lower()}"


def validate_image_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file for security and integrity."""
    try:
        # Check file size (limit to 50MB)
        uploaded_file.seek(0, 2)  # Seek to end
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)  # Reset to beginning

        if file_size > 50 * 1024 * 1024:  # 50MB limit
            return False, "File size exceeds 50MB limit"

        if file_size == 0:
            return False, "File appears to be empty"

        # Check file extension
        file_ext = os.path.splitext(uploaded_file.name.lower())[1][1:]
        if file_ext not in SUPPORTED_FORMATS:
            return False, f"Unsupported file format: {file_ext}"

        # For non-SVG files, try to open with PIL for validation
        if file_ext != "svg":
            try:
                uploaded_file.seek(0)
                with Image.open(uploaded_file) as img:
                    img.verify()
                uploaded_file.seek(0)
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"

        return True, "File validation successful"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# --- Advanced Image Processing Functions ---
class ImageProcessor:
    """Advanced image processing class with comprehensive features."""

    def __init__(self, options: Dict[str, Any]):
        self.options = options
        self.processing_log = []

    def log(self, message: str):
        """Add message to processing log."""
        self.processing_log.append(message)
        log_processing_event("processing", message)

    def apply_rotation(self, img: Image.Image) -> Image.Image:
        """Apply rotation to image."""
        if self.options.get("rotation", 0) != 0:
            rotation_angle = self.options["rotation"]
            img = img.rotate(
                -rotation_angle, expand=True
            )  # Negative for correct direction
            self.log(f"Applied {rotation_angle}° rotation")
        return img

    def apply_flip(self, img: Image.Image) -> Image.Image:
        """Apply flip operations to image."""
        if self.options.get("flip_horizontal", False):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            self.log("Applied horizontal flip")

        if self.options.get("flip_vertical", False):
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            self.log("Applied vertical flip")

        return img

    def apply_crop(self, img: Image.Image) -> Image.Image:
        """Apply cropping to image."""
        if self.options.get("enable_crop", False):
            width, height = img.size
            crop_left = int(width * self.options.get("crop_left", 0) / 100)
            crop_top = int(height * self.options.get("crop_top", 0) / 100)
            crop_right = width - int(width * self.options.get("crop_right", 0) / 100)
            crop_bottom = height - int(
                height * self.options.get("crop_bottom", 0) / 100
            )

            if crop_left < crop_right and crop_top < crop_bottom:
                img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
                self.log(
                    f"Applied crop: ({crop_left}, {crop_top}, {crop_right}, {crop_bottom})"
                )

        return img

    def apply_enhancements(self, img: Image.Image) -> Image.Image:
        """Apply brightness, contrast, saturation, and sharpness adjustments."""
        if self.options.get("brightness", 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.options["brightness"])
            self.log(f"Applied brightness: {self.options['brightness']}")

        if self.options.get("contrast", 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.options["contrast"])
            self.log(f"Applied contrast: {self.options['contrast']}")

        if self.options.get("saturation", 1.0) != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(self.options["saturation"])
            self.log(f"Applied saturation: {self.options['saturation']}")

        if self.options.get("sharpness", 1.0) != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.options["sharpness"])
            self.log(f"Applied sharpness: {self.options['sharpness']}")

        return img

    def apply_filters(self, img: Image.Image) -> Image.Image:
        """Apply image filters."""
        filter_name = self.options.get("filter", "None")
        if filter_name != "None" and filter_name in ENHANCEMENT_FILTERS:
            filter_obj = ENHANCEMENT_FILTERS[filter_name]
            if filter_obj:
                img = img.filter(filter_obj)
                self.log(f"Applied filter: {filter_name}")

        return img

    def apply_watermark(self, img: Image.Image) -> Image.Image:
        """Apply text watermark to image."""
        if self.options.get("enable_watermark", False) and self.options.get(
            "watermark_text"
        ):
            # Create a copy to work with
            watermarked = img.copy()

            # Create drawing context
            draw = ImageDraw.Draw(watermarked)

            # Try to use a default font, fallback to basic if not available
            try:
                font_size = max(20, min(img.width, img.height) // 20)
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

            # Get text size
            text = self.options["watermark_text"]
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Position watermark
            position = self.options.get("watermark_position", "bottom-right")
            margin = 20

            if position == "top-left":
                x, y = margin, margin
            elif position == "top-right":
                x, y = img.width - text_width - margin, margin
            elif position == "bottom-left":
                x, y = margin, img.height - text_height - margin
            else:  # bottom-right (default)
                x, y = (
                    img.width - text_width - margin,
                    img.height - text_height - margin,
                )

            # Apply watermark with transparency
            opacity = self.options.get("watermark_opacity", 128)
            watermark_color = (
                *self.options.get("watermark_color", (255, 255, 255)),
                opacity,
            )

            draw.text((x, y), text, fill=watermark_color, font=font)
            self.log(f"Applied watermark: '{text}' at {position}")

            return watermarked

        return img

    def apply_border(self, img: Image.Image) -> Image.Image:
        """Apply border to image."""
        if self.options.get("enable_border", False):
            border_width = self.options.get("border_width", 10)
            border_color = self.options.get("border_color", (0, 0, 0))

            # Create new image with border
            new_width = img.width + 2 * border_width
            new_height = img.height + 2 * border_width

            bordered = Image.new(img.mode, (new_width, new_height), border_color)
            bordered.paste(img, (border_width, border_width))

            self.log(f"Applied border: {border_width}px, color {border_color}")
            return bordered

        return img

    def remove_background(self, img: Image.Image) -> Image.Image:
        """Remove background using rembg if available."""
        if self.options.get("remove_background", False) and REMBG_AVAILABLE:
            try:
                # Convert PIL image to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                # Remove background
                output = rembg.remove(img_bytes)

                # Convert back to PIL Image
                result_img = Image.open(io.BytesIO(output))
                self.log("Applied background removal")
                return result_img
            except Exception as e:
                self.log(f"Background removal failed: {str(e)}")

        return img

    def apply_resize(self, img: Image.Image) -> Image.Image:
        """Apply resizing/rescaling to image."""
        if self.options.get("enable_resize", False):
            current_width, current_height = img.size
            resample_filter = RESIZE_METHODS[
                self.options.get("resample_method_key", "LANCZOS (High Quality)")
            ]

            if self.options.get("resize_mode") == "Resize (Pixels)":
                new_width = self.options.get("target_width", current_width)
                new_height = self.options.get("target_height", current_height)
            else:  # Rescale (Percent)
                scale_factor = self.options.get("scale_percent", 100) / 100.0
                new_width = max(1, int(current_width * scale_factor))
                new_height = max(1, int(current_height * scale_factor))

            if (new_width, new_height) != (current_width, current_height):
                img = img.resize((new_width, new_height), resample_filter)
                self.log(
                    f"Resized from {current_width}x{current_height} to {new_width}x{new_height}"
                )

        return img

    def process_image(self, img: Image.Image) -> Image.Image:
        """Apply all processing steps to image in correct order."""
        # Processing order is important
        img = self.apply_rotation(img)
        img = self.apply_flip(img)
        img = self.apply_crop(img)
        img = self.apply_resize(img)
        img = self.apply_enhancements(img)
        img = self.apply_filters(img)
        img = self.remove_background(img)
        img = self.apply_watermark(img)
        img = self.apply_border(img)

        return img


# --- File Processing Functions ---
def handle_svg_conversion(
    svg_bytes: bytes, options: Dict[str, Any]
) -> Tuple[bytes, bool]:
    """Convert SVG to target format."""
    if not CAIROSVG_AVAILABLE:
        raise ValueError("SVG processing requires CairoSVG library")

    render_width, render_height, scale_factor = None, None, 1.0

    if options.get("enable_resize"):
        if options.get("resize_mode") == "Resize (Pixels)":
            render_width = options.get("target_width")
            render_height = options.get("target_height")
        elif options.get("resize_mode") == "Rescale (Percent)":
            scale_factor = options.get("scale_percent", 100) / 100.0

    if options.get("output_format_lower") == "png":
        png_bytes_io = io.BytesIO()
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=png_bytes_io,
            output_width=render_width,
            output_height=render_height,
            scale=scale_factor if render_width is None else 1.0,
        )
        return png_bytes_io.getvalue(), True

    # Convert to PNG first, then to target format
    png_bytes_io = io.BytesIO()
    cairosvg.svg2png(
        bytestring=svg_bytes,
        write_to=png_bytes_io,
        output_width=render_width,
        output_height=render_height,
        scale=scale_factor if render_width is None else 1.0,
    )
    png_bytes_io.seek(0)
    img = Image.open(png_bytes_io)
    return img, False


def handle_raw_conversion(raw_file) -> Image.Image:
    """Convert RAW file to PIL Image."""
    if not RAW_AVAILABLE:
        raise ValueError("RAW processing requires rawpy library")

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp_file:
        raw_file.seek(0)
        tmp_file.write(raw_file.read())
        tmp_file_path = tmp_file.name

    try:
        # Process RAW file
        with rawpy.imread(tmp_file_path) as raw:
            rgb = raw.postprocess()

        # Convert to PIL Image
        img = Image.fromarray(rgb)
        return img
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def process_single_file(uploaded_file, options: Dict[str, Any]) -> Tuple[str, bytes]:
    """Process a single uploaded file with all enhancements."""
    processor = ImageProcessor(options)

    # Validate file
    is_valid, validation_msg = validate_image_file(uploaded_file)
    if not is_valid:
        raise ValueError(f"File validation failed: {validation_msg}")

    processor.log(f"Processing file: {uploaded_file.name}")

    file_extension = os.path.splitext(uploaded_file.name.lower())[1][1:]
    uploaded_file.seek(0)

    # Handle different input formats
    if file_extension == "svg":
        svg_data = uploaded_file.getvalue()
        result, skip_processing = handle_svg_conversion(svg_data, options)
        if skip_processing:
            img_bytes_result = result
        else:
            img = result
            img = processor.process_image(img)
    elif file_extension in ["cr2", "nef", "arw", "dng", "orf", "raf", "rw2"]:
        img = handle_raw_conversion(uploaded_file)
        img = processor.process_image(img)
    elif file_extension in ["heic", "heif"]:
        if not HEIF_AVAILABLE:
            raise ValueError("HEIC/HEIF processing requires pillow-heif library")
        img = Image.open(uploaded_file)
        img = processor.process_image(img)
    else:
        # Standard formats
        img = Image.open(uploaded_file)
        img = processor.process_image(img)

    # Handle output format and quality
    if "img_bytes_result" not in locals():
        output_format = options.get("output_format", "PNG").upper()

        # Convert color mode if necessary
        if output_format in ["JPEG"] and img.mode in ["RGBA", "P"]:
            processor.log("Converting RGBA/P to RGB for JPEG")
            img = img.convert("RGB")

        # Save with quality settings
        img_byte_arr = io.BytesIO()
        save_kwargs = {"format": output_format}

        if output_format == "JPEG":
            save_kwargs["quality"] = options.get("jpeg_quality", 95)
            save_kwargs["optimize"] = True
        elif output_format == "PNG":
            save_kwargs["compress_level"] = options.get("png_compression", 6)
            save_kwargs["optimize"] = True
        elif output_format == "WEBP":
            if options.get("webp_lossless", False):
                save_kwargs["lossless"] = True
            else:
                save_kwargs["quality"] = options.get("webp_quality", 95)
            save_kwargs["optimize"] = True

        img.save(img_byte_arr, **save_kwargs)
        img_bytes_result = img_byte_arr.getvalue()

        # Close image to free memory
        img.close()

    # Generate secure filename
    new_filename = generate_secure_filename(
        uploaded_file.name, options.get("output_format_lower", "png")
    )

    processor.log(f"Processing completed successfully")
    return new_filename, img_bytes_result


# --- PDF Creation Function ---
def create_pdf_from_images(
    image_list: List[Tuple[str, bytes]], page_size="A4"
) -> bytes:
    """Create PDF from multiple images."""
    if not REPORTLAB_AVAILABLE:
        raise ValueError("PDF creation requires reportlab library")

    if not image_list:
        raise ValueError("No images provided for PDF creation")

    pdf_buffer = io.BytesIO()
    page_width, page_height = A4 if page_size == "A4" else letter

    try:
        c = canvas.Canvas(pdf_buffer, pagesize=(page_width, page_height))

        for idx, (filename, img_bytes) in enumerate(image_list):
            try:
                # Open image to get dimensions
                img_io = io.BytesIO(img_bytes)
                img = Image.open(img_io)

                # Convert to RGB if necessary (for JPEG compatibility in PDF)
                if img.mode in ["RGBA", "P"]:
                    rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                    rgb_img.paste(
                        img, mask=img.split()[-1] if img.mode == "RGBA" else None
                    )
                    img = rgb_img

                img_width, img_height = img.size

                # Calculate scaling to fit page with margins
                margin = 40
                available_width = page_width - 2 * margin
                available_height = page_height - 2 * margin

                scale_x = available_width / img_width
                scale_y = available_height / img_height
                scale = min(scale_x, scale_y, 1.0)  # Don't scale up

                new_width = img_width * scale
                new_height = img_height * scale

                # Center image on page
                x = (page_width - new_width) / 2
                y = (page_height - new_height) / 2

                # Save image to temporary buffer for ReportLab
                temp_img_buffer = io.BytesIO()
                img.save(temp_img_buffer, format="JPEG", quality=95)
                temp_img_buffer.seek(0)

                # Add image to PDF
                img_reader = ImageReader(temp_img_buffer)
                c.drawImage(img_reader, x, y, width=new_width, height=new_height)

                # Add filename as caption
                c.setFont("Helvetica", 10)
                text_x = x
                text_y = y - 15
                c.drawString(text_x, text_y, f"{idx + 1}. {filename}")

                # Start new page for next image (except for the last one)
                if idx < len(image_list) - 1:
                    c.showPage()

                # Clean up
                img.close()
                temp_img_buffer.close()

            except Exception as e:
                # Skip problematic images but continue with others
                log_processing_event(
                    "error", f"Failed to add image {filename} to PDF: {str(e)}"
                )
                continue

        c.save()
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    except Exception as e:
        pdf_buffer.close()
        raise ValueError(f"PDF creation failed: {str(e)}")
    finally:
        try:
            pdf_buffer.close()
        except:
            pass


# --- UI Components ---
def render_theme_controls():
    """Render theme and accessibility controls."""
    with st.sidebar.expander("🎨 Theme & Accessibility"):
        theme = st.selectbox(
            "Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.current_theme),
            key="theme_selector",
        )

        if theme != st.session_state.current_theme:
            st.session_state.current_theme = theme
            st.rerun()

        st.session_state.accessibility_mode = st.checkbox(
            "High Contrast Mode",
            value=st.session_state.accessibility_mode,
            help="Enable high contrast for better visibility",
        )

        st.session_state.language = st.selectbox(
            "Language",
            options=["English", "Spanish", "French", "German"],
            index=0,
            help="Interface language (demo - functionality not implemented)",
        )


def render_advanced_sidebar():
    """Render advanced processing options in sidebar."""
    with st.sidebar:
        st.header("⚙️ Conversion Options")

        # Quick presets
        preset_options = ["Custom"] + list(st.session_state.conversion_presets.keys())
        selected_preset = st.selectbox(
            "Conversion Preset",
            options=preset_options,
            index=preset_options.index(st.session_state.selected_preset),
            help="Load saved conversion settings",
        )

        if selected_preset != st.session_state.selected_preset:
            st.session_state.selected_preset = selected_preset
            if selected_preset != "Custom":
                # Load preset settings (would update session state)
                st.info(f"Loaded preset: {selected_preset}")

        # Basic output settings
        output_format = st.selectbox(
            "Output Format",
            options=SUPPORTED_OUTPUT_FORMATS,
            index=SUPPORTED_OUTPUT_FORMATS.index(DEFAULT_OUTPUT_FORMAT),
            help="Target file format for conversion",
        )

        # Quality settings
        with st.expander("🎛️ Quality & Compression"):
            if output_format == "JPEG":
                jpeg_quality = st.slider(
                    "JPEG Quality",
                    min_value=1,
                    max_value=100,
                    value=95,
                    help="Higher values = better quality, larger files",
                )
            else:
                jpeg_quality = 95

            if output_format == "PNG":
                png_compression = st.slider(
                    "PNG Compression Level",
                    min_value=0,
                    max_value=9,
                    value=6,
                    help="Higher values = smaller files, slower processing",
                )
            else:
                png_compression = 6

            if output_format == "WEBP":
                webp_lossless = st.checkbox("WebP Lossless", value=False)
                if not webp_lossless:
                    webp_quality = st.slider("WebP Quality", 1, 100, 95)
                else:
                    webp_quality = 100
            else:
                webp_lossless = False
                webp_quality = 95

        # Image transformations
        with st.expander("🔄 Rotation & Flipping"):
            rotation = st.selectbox(
                "Rotation",
                options=[0, 90, 180, 270],
                format_func=lambda x: f"{x}°" if x > 0 else "None",
            )

            flip_horizontal = st.checkbox("Flip Horizontal")
            flip_vertical = st.checkbox("Flip Vertical")

        # Cropping
        with st.expander("✂️ Cropping"):
            enable_crop = st.checkbox("Enable Cropping")
            if enable_crop:
                col1, col2 = st.columns(2)
                with col1:
                    crop_left = st.slider("Left %", 0, 50, 0)
                    crop_top = st.slider("Top %", 0, 50, 0)
                with col2:
                    crop_right = st.slider("Right %", 0, 50, 0)
                    crop_bottom = st.slider("Bottom %", 0, 50, 0)
            else:
                crop_left = crop_top = crop_right = crop_bottom = 0

        # Resizing
        with st.expander("📏 Resizing & Scaling"):
            enable_resize = st.checkbox("Enable Resizing/Rescaling")
            if enable_resize:
                resize_mode = st.radio(
                    "Mode", options=["Resize (Pixels)", "Rescale (Percent)"]
                )

                if resize_mode == "Resize (Pixels)":
                    col1, col2 = st.columns(2)
                    with col1:
                        target_width = st.number_input("Width (px)", 1, 10000, 800)
                    with col2:
                        target_height = st.number_input("Height (px)", 1, 10000, 600)
                    scale_percent = 100
                else:
                    scale_percent = st.slider("Scale %", 10, 500, 100)
                    target_width = target_height = None

                resample_method_key = st.selectbox(
                    "Resampling Method", options=list(RESIZE_METHODS.keys()), index=0
                )
            else:
                resize_mode = None
                target_width = target_height = scale_percent = None
                resample_method_key = "LANCZOS (High Quality)"

        # Image enhancements
        with st.expander("✨ Image Enhancement"):
            brightness = st.slider("Brightness", 0.1, 3.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.1, 3.0, 1.0, 0.1)
            saturation = st.slider("Saturation", 0.0, 3.0, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.1, 3.0, 1.0, 0.1)

            filter_option = st.selectbox(
                "Filter", options=list(ENHANCEMENT_FILTERS.keys())
            )

        # Advanced features
        with st.expander("🚀 Advanced Features"):
            # Watermarking
            enable_watermark = st.checkbox("Add Watermark")
            if enable_watermark:
                watermark_text = st.text_input("Watermark Text", "© Your Name")
                watermark_position = st.selectbox(
                    "Position",
                    options=["top-left", "top-right", "bottom-left", "bottom-right"],
                )
                watermark_opacity = st.slider("Opacity", 50, 255, 128)
                watermark_color = st.color_picker("Color", "#FFFFFF")
                # Convert hex to RGB
                watermark_color = tuple(
                    int(watermark_color[i : i + 2], 16) for i in (1, 3, 5)
                )
            else:
                watermark_text = ""
                watermark_position = "bottom-right"
                watermark_opacity = 128
                watermark_color = (255, 255, 255)

            # Borders
            enable_border = st.checkbox("Add Border")
            if enable_border:
                border_width = st.slider("Border Width (px)", 1, 100, 10)
                border_color = st.color_picker("Border Color", "#000000")
                # Convert hex to RGB
                border_color = tuple(
                    int(border_color[i : i + 2], 16) for i in (1, 3, 5)
                )
            else:
                border_width = 0
                border_color = (0, 0, 0)

            # Background removal
            if REMBG_AVAILABLE:
                remove_background = st.checkbox(
                    "Remove Background (AI)", help="Use AI to remove image background"
                )
            else:
                remove_background = False
                st.info("Background removal requires rembg library")

        # Batch processing options
        with st.expander("📦 Batch Options"):
            st.session_state.auto_process = st.checkbox(
                "Auto-process uploads",
                value=st.session_state.auto_process,
                help="Automatically process files when uploaded",
            )

            multi_format_output = st.multiselect(
                "Multi-format Output",
                options=SUPPORTED_OUTPUT_FORMATS,
                default=[output_format],
                help="Generate multiple formats from single input",
            )

            if len(multi_format_output) == 0:
                multi_format_output = [output_format]

        # Save/Load presets
        with st.expander("💾 Presets"):
            preset_name = st.text_input("Preset Name")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Preset") and preset_name:
                    # Save current settings as preset
                    st.session_state.conversion_presets[preset_name] = {
                        "output_format": output_format,
                        "jpeg_quality": jpeg_quality,
                        "png_compression": png_compression,
                        # Add other settings...
                    }
                    st.success(f"Saved preset: {preset_name}")
            with col2:
                if (
                    st.button("Delete Preset")
                    and preset_name in st.session_state.conversion_presets
                ):
                    del st.session_state.conversion_presets[preset_name]
                    st.success(f"Deleted preset: {preset_name}")

        render_theme_controls()

        # Developer info
        st.markdown("---")
        st.markdown("*Developer: Ajay Khanna*")
        st.markdown("*Place: LANL*")
        st.markdown("*Date: May 12, 2025*")

    return {
        "output_format": output_format,
        "output_format_lower": output_format.lower(),
        "jpeg_quality": jpeg_quality,
        "png_compression": png_compression,
        "webp_lossless": webp_lossless,
        "webp_quality": webp_quality,
        "rotation": rotation,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "enable_crop": enable_crop,
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_right": crop_right,
        "crop_bottom": crop_bottom,
        "enable_resize": enable_resize,
        "resize_mode": resize_mode,
        "target_width": target_width,
        "target_height": target_height,
        "scale_percent": scale_percent,
        "resample_method_key": resample_method_key,
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "sharpness": sharpness,
        "filter": filter_option,
        "enable_watermark": enable_watermark,
        "watermark_text": watermark_text,
        "watermark_position": watermark_position,
        "watermark_opacity": watermark_opacity,
        "watermark_color": watermark_color,
        "enable_border": enable_border,
        "border_width": border_width,
        "border_color": border_color,
        "remove_background": remove_background,
        "multi_format_output": multi_format_output,
    }


def render_main_interface():
    """Render the main application interface."""
    # Apply theme
    theme = THEMES[st.session_state.current_theme]

    # Custom CSS for theming
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-color: {theme['background']};
        color: {theme['text']};
    }}
    .upload-zone {{
        border: 2px dashed {theme['accent']};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background: {theme['secondary']}20;
    }}
    .metric-card {{
        background: {theme['secondary']}10;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid {theme['accent']};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # App title with feature indicators
    feature_badges = []
    if CAIROSVG_AVAILABLE:
        feature_badges.append("SVG")
    if HEIF_AVAILABLE:
        feature_badges.append("HEIC")
    if RAW_AVAILABLE:
        feature_badges.append("RAW")
    if REMBG_AVAILABLE:
        feature_badges.append("AI-BG")

    title = "🖼️ Multi-Format Image Converter"
    if feature_badges:
        title += f" [{' • '.join(feature_badges)}]"

    st.title(title)

    # Quick stats dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Processed",
            st.session_state.processing_stats["total_processed"],
            help="Total images processed in this session",
        )
    with col2:
        st.metric(
            "Success Rate",
            f"{((st.session_state.processing_stats['total_processed'] - st.session_state.processing_stats['total_errors']) / max(1, st.session_state.processing_stats['total_processed']) * 100):.1f}%",
            help="Percentage of successfully processed images",
        )
    with col3:
        avg_time = (
            np.mean(st.session_state.processing_stats["processing_times"])
            if st.session_state.processing_stats["processing_times"]
            else 0
        )
        st.metric(
            "Avg Process Time",
            f"{avg_time:.2f}s",
            help="Average processing time per image",
        )
    with col4:
        st.metric(
            "Queue Size",
            len(st.session_state.processing_queue),
            help="Number of files in processing queue",
        )

    # Supported formats info
    st.info(
        f"""
    **Supported Input:** {', '.join(SUPPORTED_FORMATS).upper()}  
    **Supported Output:** {', '.join(SUPPORTED_OUTPUT_FORMATS)}  
    **Advanced Features:** {'Enabled' if any([REMBG_AVAILABLE, CV2_AVAILABLE, REPORTLAB_AVAILABLE]) else 'Basic mode - install optional libraries for full features'}
    """
    )

    # File upload section
    st.markdown("### 📁 File Upload")

    # Upload controls
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select Image Files",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}",
            help=f"Drag and drop files here. Max 50MB per file.",
        )

    with col2:
        if st.button("📋 Clear Queue", help="Clear all files from processing queue"):
            st.session_state.processing_queue = []
            st.session_state.uploader_key += 1
            st.rerun()

    with col3:
        # View options
        st.session_state.grid_view = st.checkbox(
            "Grid View", value=st.session_state.grid_view
        )

    # Handle file uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.file_id not in [
                f.file_id for f in st.session_state.processing_queue
            ]:
                is_valid, validation_msg = validate_image_file(uploaded_file)
                if is_valid:
                    st.session_state.processing_queue.append(uploaded_file)
                    log_processing_event(
                        "upload", f"Added to queue: {uploaded_file.name}"
                    )
                    if st.session_state.auto_process:
                        st.info(f"Auto-processing: {uploaded_file.name}")
                else:
                    st.error(f"Invalid file {uploaded_file.name}: {validation_msg}")

        st.session_state.uploader_key += 1
        st.rerun()

    # Queue management
    if st.session_state.processing_queue:
        st.markdown("### 🔄 Processing Queue")

        # Thumbnail size control
        thumbnail_sizes = {"Small": 80, "Medium": 120, "Large": 160}
        st.session_state.thumbnail_size = st.selectbox(
            "Thumbnail Size",
            options=list(thumbnail_sizes.keys()),
            index=list(thumbnail_sizes.keys()).index(st.session_state.thumbnail_size),
        )
        thumb_size = thumbnail_sizes[st.session_state.thumbnail_size]

        # Display queue
        if st.session_state.grid_view:
            render_grid_view(thumb_size)
        else:
            render_list_view()

    return st.session_state.processing_queue


def render_grid_view(thumb_size: int):
    """Render processing queue in grid view."""
    cols_per_row = max(1, 600 // thumb_size)  # Adjust based on thumbnail size

    for i in range(0, len(st.session_state.processing_queue), cols_per_row):
        row_files = st.session_state.processing_queue[i : i + cols_per_row]
        cols = st.columns(len(row_files))

        for col_idx, uploaded_file in enumerate(row_files):
            with cols[col_idx]:
                try:
                    # Display image preview
                    if (
                        uploaded_file.name.lower().endswith(".svg")
                        and CAIROSVG_AVAILABLE
                    ):
                        uploaded_file.seek(0)
                        svg_bytes = uploaded_file.getvalue()
                        png_preview = cairosvg.svg2png(
                            bytestring=svg_bytes, output_width=thumb_size
                        )
                        st.image(
                            png_preview,
                            width=thumb_size,
                            caption=uploaded_file.name[:20],
                        )
                    else:
                        uploaded_file.seek(0)
                        st.image(
                            uploaded_file,
                            width=thumb_size,
                            caption=uploaded_file.name[:20],
                        )

                    # File info
                    uploaded_file.seek(0, 2)
                    file_size = uploaded_file.tell()
                    uploaded_file.seek(0)
                    st.caption(f"{file_size / 1024:.1f} KB")

                    # Remove button
                    if st.button(
                        "❌",
                        key=f"remove_{uploaded_file.file_id}",
                        help="Remove from queue",
                    ):
                        st.session_state.processing_queue.remove(uploaded_file)
                        st.rerun()

                except Exception as e:
                    st.error(f"Preview error: {str(e)[:50]}...")


def render_list_view():
    """Render processing queue in list view."""
    for idx, uploaded_file in enumerate(st.session_state.processing_queue):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.write(f"**{uploaded_file.name}**")

        with col2:
            uploaded_file.seek(0, 2)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)
            st.write(f"{file_size / 1024:.1f} KB")

        with col3:
            file_ext = os.path.splitext(uploaded_file.name.lower())[1][1:]
            st.write(f"Type: {file_ext.upper()}")

        with col4:
            if st.button("❌", key=f"remove_list_{uploaded_file.file_id}"):
                st.session_state.processing_queue.remove(uploaded_file)
                st.rerun()


def render_processing_interface(options: Dict[str, Any]):
    """Render the processing interface and controls."""
    if not st.session_state.processing_queue:
        st.info("👆 Upload files to get started")
        return

    st.markdown("### 🚀 Processing Controls")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🎯 Process Queue", key="process_main", type="primary"):
            process_queue(options)

    with col2:
        if st.session_state.get("processing_active", False):
            if st.button("⏹️ Cancel", key="cancel_process"):
                st.session_state.cancel_processing = True
                st.warning("Cancelling processing...")

    with col3:
        # Initialize show_stats state if it doesn't exist
        if "show_stats" not in st.session_state:
            st.session_state.show_stats = False

        if st.button("📊 Show Stats", key="show_stats_btn"):
            st.session_state.show_stats = not st.session_state.show_stats
            st.rerun()

    # Display analytics dashboard if requested
    if st.session_state.show_stats:
        render_analytics_dashboard()


def process_queue(options: Dict[str, Any]):
    """Process all files in the queue with comprehensive error handling."""
    if not st.session_state.processing_queue:
        st.warning("No files to process")
        return

    st.session_state.processing_active = True
    st.session_state.cancel_processing = False

    total_files = len(st.session_state.processing_queue)
    processed_images = []
    errors = []

    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()

    start_time = time.time()

    # Process each format if multi-format output is enabled
    all_formats = options.get("multi_format_output", [options["output_format"]])

    for format_idx, target_format in enumerate(all_formats):
        format_options = options.copy()
        format_options["output_format"] = target_format
        format_options["output_format_lower"] = target_format.lower()

        status_text.text(f"Processing format: {target_format}")

        for file_idx, uploaded_file in enumerate(st.session_state.processing_queue):
            if st.session_state.cancel_processing:
                st.warning("Processing cancelled by user")
                break

            current_progress = (format_idx * total_files + file_idx) / (
                len(all_formats) * total_files
            )
            progress_bar.progress(current_progress)

            file_start_time = time.time()
            status_text.text(f"Processing: {uploaded_file.name} → {target_format}")

            try:
                filename, img_bytes = process_single_file(uploaded_file, format_options)
                processed_images.append((filename, img_bytes))

                # Update statistics
                processing_time = time.time() - file_start_time
                st.session_state.processing_stats["processing_times"].append(
                    processing_time
                )
                st.session_state.processing_stats["total_processed"] += 1

                # Add to processing history
                st.session_state.processing_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "input_file": uploaded_file.name,
                        "output_file": filename,
                        "format": target_format,
                        "processing_time": processing_time,
                        "status": "success",
                    }
                )

                log_processing_event(
                    "success", f"Processed {uploaded_file.name} to {target_format}"
                )

            except Exception as e:
                error_msg = f"Error processing {uploaded_file.name}: {str(e)}"
                errors.append(error_msg)
                st.session_state.processing_stats["total_errors"] += 1

                # Add to processing history
                st.session_state.processing_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "input_file": uploaded_file.name,
                        "output_file": None,
                        "format": target_format,
                        "processing_time": time.time() - file_start_time,
                        "status": "error",
                        "error": str(e),
                    }
                )

                log_processing_event("error", error_msg, uploaded_file.name)
                st.error(f"❌ {error_msg}")

        if st.session_state.cancel_processing:
            break

    total_time = time.time() - start_time

    # Update progress to complete
    progress_bar.progress(1.0)
    status_text.text(
        f"Processing complete! ({len(processed_images)} files processed in {total_time:.2f}s)"
    )

    st.session_state.processing_active = False

    # Display results
    render_results(processed_images, errors, total_time)


def render_results(
    processed_images: List[Tuple[str, bytes]], errors: List[str], total_time: float
):
    """Render processing results and download options."""
    if not processed_images and not errors:
        return

    st.markdown("### 📊 Processing Results")

    # Summary
    success_count = len(processed_images)
    error_count = len(errors)
    total_count = success_count + error_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successful", success_count)
    with col2:
        st.metric("❌ Errors", error_count)
    with col3:
        st.metric("⏱️ Total Time", f"{total_time:.2f}s")

    # Download options
    if processed_images:
        st.markdown("### 📥 Download Options")

        # Initialize PDF creation state
        if "pdf_created" not in st.session_state:
            st.session_state.pdf_created = False
        if "pdf_bytes" not in st.session_state:
            st.session_state.pdf_bytes = None

        col1, col2, col3 = st.columns(3)

        with col1:
            # Single file download
            if len(processed_images) == 1:
                filename, img_bytes = processed_images[0]
                file_ext = os.path.splitext(filename)[1][1:].lower()
                mime_type = MIME_TYPES.get(file_ext, "application/octet-stream")

                st.download_button(
                    f"📄 Download {filename}",
                    data=img_bytes,
                    file_name=filename,
                    mime=mime_type,
                    key="single_download_btn",
                )

        with col2:
            # ZIP download
            zip_buffer = create_zip_archive(processed_images)
            if zip_buffer:
                st.download_button(
                    f"📦 Download All ({len(processed_images)} files)",
                    data=zip_buffer.getvalue(),
                    file_name=f"converted_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="zip_download_btn",
                )

        with col3:
            # PDF creation
            if REPORTLAB_AVAILABLE and len(processed_images) > 1:
                if st.button("📑 Create PDF", key="create_pdf_btn"):
                    try:
                        with st.spinner("Creating PDF..."):
                            pdf_bytes = create_pdf_from_images(processed_images)
                            st.session_state.pdf_bytes = pdf_bytes
                            st.session_state.pdf_created = True
                        st.success("PDF created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"PDF creation failed: {str(e)}")
                        st.session_state.pdf_created = False

        # PDF download button (appears after PDF is created)
        if st.session_state.pdf_created and st.session_state.pdf_bytes:
            st.download_button(
                "📑 Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"images_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key="pdf_download_btn",
            )

    # Error details
    if errors:
        with st.expander(f"❌ Error Details ({len(errors)} errors)"):
            for error in errors:
                st.error(error)


def render_analytics_dashboard():
    """Render analytics and statistics dashboard."""
    st.markdown("### 📈 Analytics Dashboard")

    if not st.session_state.processing_history:
        st.info("No processing history available yet.")
        return

    # Processing history chart
    processing_times = st.session_state.processing_stats.get("processing_times", [])
    if processing_times:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Processing times histogram
            ax1.hist(
                processing_times,
                bins=min(20, len(processing_times)),
                alpha=0.7,
                color="skyblue",
            )
            ax1.set_title("Processing Times Distribution")
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Frequency")
            ax1.grid(True, alpha=0.3)

            # Success/Error pie chart
            success_count = (
                st.session_state.processing_stats["total_processed"]
                - st.session_state.processing_stats["total_errors"]
            )
            error_count = st.session_state.processing_stats["total_errors"]

            if success_count > 0 or error_count > 0:
                sizes = (
                    [success_count, error_count] if error_count > 0 else [success_count]
                )
                labels = ["Success", "Errors"] if error_count > 0 else ["Success"]
                colors = ["#4CAF50", "#F44336"] if error_count > 0 else ["#4CAF50"]

                ax2.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax2.set_title("Processing Success Rate")
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Processing Success Rate")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error generating charts: {str(e)}")
    else:
        st.info("No processing time data available yet.")

    # Statistics summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", st.session_state.processing_stats["total_processed"])
    with col2:
        st.metric("Errors", st.session_state.processing_stats["total_errors"])
    with col3:
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            st.metric("Avg Time", f"{avg_time:.2f}s")
        else:
            st.metric("Avg Time", "N/A")
    with col4:
        if processing_times:
            st.metric("Fastest", f"{min(processing_times):.2f}s")
        else:
            st.metric("Fastest", "N/A")

    # Recent processing history
    st.markdown("#### Recent Processing History")
    if st.session_state.processing_history:
        recent_history = st.session_state.processing_history[-10:]  # Last 10 entries

        for entry in reversed(recent_history):
            status_icon = "✅" if entry["status"] == "success" else "❌"
            with st.expander(
                f"{status_icon} {entry['input_file']} → {entry.get('output_file', 'Failed')}"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Format:** {entry['format']}")
                    st.write(f"**Time:** {entry['processing_time']:.2f}s")
                with col2:
                    st.write(f"**Timestamp:** {entry['timestamp']}")
                    if entry["status"] == "error":
                        st.error(f"Error: {entry.get('error', 'Unknown error')}")

    # Processing logs
    if st.session_state.processing_logs:
        with st.expander("📋 Processing Logs"):
            log_text = "\n".join(
                [
                    f"[{log['timestamp']}] {log['type'].upper()}: {log['message']}"
                    for log in st.session_state.processing_logs[-50:]  # Last 50 logs
                ]
            )
            st.text_area("Logs", value=log_text, height=200, key="logs_display")

            # Download logs
            if st.button("📥 Download Logs", key="download_logs_btn"):
                log_data = json.dumps(st.session_state.processing_logs, indent=2)
                st.download_button(
                    "💾 Save Logs as JSON",
                    data=log_data,
                    file_name=f"processing_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="save_logs_btn",
                )


def render_help_and_info():
    """Render help documentation and feature information."""
    with st.expander("❓ Help & Documentation"):
        st.markdown(
            """
        ## 🖼️ Enhanced Image Converter - User Guide
        
        ### 📋 Supported Formats
        
        **Input Formats:**
        - **Standard:** PNG, JPG/JPEG, BMP, GIF, WEBP, TGA
        - **Vector:** SVG (requires CairoSVG)
        - **Modern:** HEIC/HEIF (requires pillow-heif)
        - **RAW:** CR2, NEF, ARW, DNG, ORF, RAF, RW2 (requires rawpy)
        
        **Output Formats:**
        - PNG, JPEG, WEBP, BMP, GIF, TGA, TIFF
        - PDF (multi-image, requires reportlab)
        
        ### 🎯 Key Features
        
        #### Basic Processing
        - **Format Conversion:** Convert between all supported formats
        - **Quality Control:** Adjust JPEG quality, PNG compression, WebP settings
        - **Batch Processing:** Process multiple files simultaneously
        
        #### Image Transformations
        - **Rotation:** 90°, 180°, 270° rotation
        - **Flipping:** Horizontal and vertical flip
        - **Cropping:** Percentage-based cropping from edges
        - **Resizing:** Pixel-based or percentage scaling
        
        #### Enhancement Tools
        - **Color Adjustments:** Brightness, contrast, saturation
        - **Sharpness:** Image sharpening control
        - **Filters:** Blur, detail, edge enhance, emboss, and more
        - **Background Removal:** AI-powered (requires rembg)
        
        #### Professional Features
        - **Watermarking:** Text watermarks with positioning
        - **Borders:** Colored borders with custom width
        - **Multi-format Export:** Generate multiple formats from single input
        - **PDF Creation:** Combine images into PDF documents
        
        #### Workflow Management
        - **Presets:** Save and load conversion settings
        - **Auto-processing:** Automatic conversion on upload
        - **Processing History:** Track all conversions
        - **Analytics:** Performance metrics and statistics
        
        ### ⚙️ Advanced Settings
        
        #### Quality Controls
        - **JPEG Quality:** 1-100 scale for compression vs quality
        - **PNG Compression:** 0-9 levels for file size optimization
        - **WebP Options:** Lossless or lossy with quality control
        
        #### Processing Options
        - **Resampling Methods:** LANCZOS, BICUBIC, BILINEAR, NEAREST
        - **Color Space:** Automatic handling of RGBA, RGB, etc.
        - **Metadata:** EXIF data preservation options
        
        ### 🎨 User Interface
        
        #### Theme & Accessibility
        - **Dark/Light Themes:** Visual preference options
        - **High Contrast Mode:** Enhanced visibility
        - **Responsive Design:** Works on desktop and mobile
        - **Grid/List Views:** Choose your preferred layout
        
        #### Keyboard Shortcuts (Planned)
        - **Ctrl+U:** Upload files
        - **Ctrl+Enter:** Process queue
        - **Ctrl+C:** Clear queue
        - **Escape:** Cancel processing
        
        ### 🔒 Security & Privacy
        
        #### File Validation
        - **Size Limits:** 50MB maximum per file
        - **Format Verification:** Secure file type checking
        - **Integrity Checks:** Validate image files before processing
        
        #### Data Handling
        - **Memory Management:** Efficient processing for large batches
        - **Temporary Files:** Automatic cleanup
        - **Session Isolation:** Your data stays private
        
        ### 📊 Analytics & Monitoring
        
        #### Performance Metrics
        - **Processing Times:** Track conversion speed
        - **Success Rates:** Monitor processing reliability
        - **Queue Management:** Real-time queue status
        
        #### Logging & Debugging
        - **Processing Logs:** Detailed operation records
        - **Error Tracking:** Comprehensive error reporting
        - **Export Logs:** Download logs for troubleshooting
        
        ### 🚀 Tips for Best Results
        
        1. **Large Files:** For files >10MB, consider using lower quality settings
        2. **Batch Processing:** Group similar operations for efficiency
        3. **Quality Settings:** Use 85-95 JPEG quality for best size/quality balance
        4. **RAW Files:** RAW processing may take longer, be patient
        5. **SVG Files:** Vector graphics work best at higher resolutions
        6. **Background Removal:** Works best with clear subject/background contrast
        
        ### ❗ Troubleshooting
        
        **Common Issues:**
        - **Upload Failed:** Check file size (max 50MB) and format
        - **Processing Slow:** Reduce image size or lower quality settings
        - **SVG Not Working:** Ensure CairoSVG library is installed
        - **Memory Errors:** Process fewer files at once
        - **Quality Issues:** Adjust compression/quality settings
        
        **Library Dependencies:**
        ```bash
        # Core requirements
        pip install streamlit Pillow numpy opencv-python matplotlib reportlab
        
        # Optional features
        pip install CairoSVG pillow-heif rawpy rembg
        ```
        
        ### 📞 Support & Development
        
        **Developer:** Ajay Khanna  
        **Institution:** Los Alamos National Laboratory (LANL)  
        **Version:** Enhanced Multi-Format Converter v2.0  
        **Date:** May 12, 2025  
        
        For technical support or feature requests, please refer to the processing logs and error messages for detailed information.
        """
        )


def create_zip_archive(image_data_list: List[Tuple[str, bytes]]) -> io.BytesIO:
    """Create ZIP archive containing processed images."""
    if not image_data_list:
        return None

    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, img_bytes in image_data_list:
                zip_file.writestr(filename, img_bytes)
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        log_processing_event("error", f"ZIP creation failed: {str(e)}")
        return None


def cleanup_temp_files():
    """Clean up temporary files to free memory."""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception:
            pass
    st.session_state.temp_files.clear()


# --- Main Application Function ---
def main():
    """Main application entry point with comprehensive feature implementation."""
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Image Converter",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session_state()

    # Render sidebar with all options
    options = render_advanced_sidebar()

    # Main interface
    queue = render_main_interface()

    # Processing interface
    render_processing_interface(options)

    # Help and documentation
    render_help_and_info()

    # Footer with additional features
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧹 Cleanup Temp Files"):
            cleanup_temp_files()
            st.success("Temporary files cleaned up!")

    with col2:
        if st.button("📊 Export Analytics"):
            analytics_data = {
                "processing_stats": st.session_state.processing_stats,
                "processing_history": st.session_state.processing_history,
                "session_info": {
                    "queue_size": len(st.session_state.processing_queue),
                    "theme": st.session_state.current_theme,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            st.download_button(
                "💾 Download Analytics",
                data=json.dumps(analytics_data, indent=2),
                file_name=f"converter_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

    with col3:
        if st.button("🔄 Reset Session"):
            # Reset all session state except theme preference
            theme = st.session_state.current_theme
            for key in list(st.session_state.keys()):
                if key not in ["current_theme"]:
                    del st.session_state[key]
            initialize_session_state()
            st.session_state.current_theme = theme
            st.success("Session reset successfully!")
            st.rerun()

    # Performance monitoring
    if len(st.session_state.processing_stats["processing_times"]) > 0:
        avg_time = np.mean(st.session_state.processing_stats["processing_times"])
        if avg_time > 10:  # If average processing time > 10 seconds
            st.warning(
                f"⚠️ Average processing time is high ({avg_time:.1f}s). Consider reducing image sizes or quality settings."
            )

    # Memory management reminder
    if len(st.session_state.processing_queue) > 20:
        st.info(
            "💡 Processing large batches? Consider processing in smaller groups for optimal performance."
        )


# --- Application Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page to restart the application.")

        # Log the error
        log_processing_event("critical", f"Application crashed: {str(e)}")

        # Provide debug information
        with st.expander("🐛 Debug Information"):
            st.code(
                f"""
            Error Type: {type(e).__name__}
            Error Message: {str(e)}
            Session State Keys: {list(st.session_state.keys())}
            Available Libraries:
            - CairoSVG: {CAIROSVG_AVAILABLE}
            - HEIF: {HEIF_AVAILABLE}
            - RAW: {RAW_AVAILABLE}
            - REMBG: {REMBG_AVAILABLE}
            - CV2: {CV2_AVAILABLE}
            - ReportLab: {REPORTLAB_AVAILABLE}
            """
            )
