# Developer: Ajay Khanna
# Places: LANL
# Date: May.12.2025

"""
Streamlit GUI application for converting multiple image files between various formats.
This version is refactored for better modularity and includes an image preview queue.

Features:
- Supports multiple input formats including PNG, JPG, TGA, BMP, GIF, WEBP.
- Optionally supports SVG input if the 'CairoSVG' library and its dependencies are installed.
- Image preview queue for uploaded files with option to remove individual files or clear queue.
- Allows selecting output format (PNG, JPEG, WEBP, BMP, GIF, TGA), default is PNG.
  (Note: SVG output is not supported).
- Multi-file drag-and-drop upload.
- Optional resizing (to specific pixel dimensions) or rescaling (by percentage).
- Choice of resampling filters for resizing/rescaling (applies to raster images).
- Options configured in the sidebar.
- Handles errors gracefully during processing.
- Packages converted images into a downloadable ZIP archive.
- Provides a direct download button if only one image is converted successfully.

Requires: streamlit, Pillow
Optional for SVG support: CairoSVG
Installation: pip install streamlit Pillow CairoSVG
Note: CairoSVG may require system dependencies (Cairo graphics library). If not found,
      SVG support will be disabled, and a message will be printed to the console.
"""

import streamlit as st
from PIL import Image, UnidentifiedImageError
import os
import io
import zipfile
import time  # To provide progress feedback
import sys  # To check for modules

# --- Try importing CairoSVG and set a flag ---
cairosvg_available = False
try:
    import cairosvg

    # Perform a minimal check to see if the underlying library is likely present
    dummy_svg = '<svg height="1" width="1"></svg>'
    cairosvg.svg2png(bytestring=dummy_svg.encode("utf-8"))
    cairosvg_available = True
    print("CairoSVG library found. SVG input support enabled.", file=sys.stdout)
except ImportError:
    print(
        "INFO: 'CairoSVG' library not found. SVG input support will be disabled.",
        file=sys.stdout,
    )
    print(
        "      To enable SVG support, install it: pip install CairoSVG", file=sys.stdout
    )
    print(
        "      You may also need system dependencies (e.g., libcairo2-dev on Debian/Ubuntu, brew install cairo on macOS).",
        file=sys.stdout,
    )
except Exception as e:
    print(
        f"WARNING: CairoSVG imported but failed runtime check. SVG input support disabled. Error: {e}",
        file=sys.stdout,
    )
    cairosvg_available = False

# --- Global Configuration ---
SUPPORTED_RASTER_INPUT_FORMATS = ["png", "jpg", "jpeg", "tga", "bmp", "gif", "webp"]
SUPPORTED_INPUT_FORMATS = (
    SUPPORTED_RASTER_INPUT_FORMATS + ["svg"]
    if cairosvg_available
    else SUPPORTED_RASTER_INPUT_FORMATS
)
SUPPORTED_OUTPUT_FORMATS = ["PNG", "JPEG", "WEBP", "BMP", "GIF", "TGA"]
DEFAULT_OUTPUT_FORMAT = "PNG"

RESIZE_METHODS = {
    "LANCZOS (High Quality)": Image.Resampling.LANCZOS,
    "BICUBIC": Image.Resampling.BICUBIC,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST (Fastest)": Image.Resampling.NEAREST,
}

MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tga": "image/x-tga",
}


# --- Helper Functions ---
def create_zip_archive(image_data_list):
    """Creates a ZIP archive in memory containing processed images."""
    if not image_data_list:
        return None
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            for filename, img_bytes in image_data_list:
                zip_file.writestr(filename, img_bytes)
    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        return None
    return zip_buffer


def generate_new_filename(original_filename, target_format_lower):
    """Generates a new filename for the converted image."""
    base_name, _ = os.path.splitext(original_filename)
    return f"{base_name}_converted.{target_format_lower}"


# --- UI Rendering Functions ---
def render_sidebar():
    """Renders all sidebar widgets and returns their selected values as a dictionary."""
    with st.sidebar:
        st.header("⚙️ Conversion Options")

        output_format_selected = st.selectbox(
            "Select Output Format:",
            options=SUPPORTED_OUTPUT_FORMATS,
            index=SUPPORTED_OUTPUT_FORMATS.index(DEFAULT_OUTPUT_FORMAT),
            help="Choose the file format for the converted images (SVG output not supported).",
        )
        output_format_lower_selected = output_format_selected.lower()

        # --- Quality options for PNG/JPEG ---
        png_compression_level = 6  # Pillow default
        jpeg_quality = 85  # Pillow default

        if output_format_lower_selected == "png":
            png_compression_level = st.slider(
                "PNG Compression Level (0=none, 9=max)",
                min_value=0,
                max_value=9,
                value=6,
                help="Lower value = less compression, larger file, faster save. 0 is highest quality.",
            )
        elif output_format_lower_selected in ("jpeg", "jpg"):
            jpeg_quality = st.slider(
                "JPEG Quality (1-100)",
                min_value=1,
                max_value=100,
                value=85,
                help="Higher value = better quality, larger file. 85 is standard.",
            )

        st.markdown("---")
        st.subheader("Resizing & Rescaling")
        resize_help_text = "Check this box to enable image dimension changes."
        if cairosvg_available:
            resize_help_text += " For SVG input, this sets the rendering size."
        enable_resize_selected = st.checkbox(
            "Enable Resizing/Rescaling", value=False, help=resize_help_text
        )

        resize_mode_selected = None
        target_width_selected = None
        target_height_selected = None
        scale_percent_selected = 100
        resample_method_key_selected = "LANCZOS (High Quality)"

        if enable_resize_selected:
            resize_mode_selected = st.radio(
                "Select Mode:",
                ("Resize (Pixels)", "Rescale (Percent)"),
                key="resize_mode_radio",
                help="Choose whether to set exact dimensions or scale by a percentage.",
            )
            resample_method_help = (
                "Algorithm used for resizing raster images (PNG, JPG, etc.)."
            )
            if cairosvg_available:
                resample_method_help += " Not directly used for SVG rendering size."
            resample_method_key_selected = st.selectbox(
                "Resampling Method (Raster only):",
                options=list(RESIZE_METHODS.keys()),
                index=0,
                help=resample_method_help,
            )
            if resize_mode_selected == "Resize (Pixels)":
                col1, col2 = st.columns(2)
                with col1:
                    target_width_selected = st.number_input(
                        "Target Width (px):",
                        min_value=1,
                        value=800,
                        step=1,
                        key="target_width_input",
                    )
                with col2:
                    target_height_selected = st.number_input(
                        "Target Height (px):",
                        min_value=1,
                        value=600,
                        step=1,
                        key="target_height_input",
                    )
            elif resize_mode_selected == "Rescale (Percent)":
                scale_help = "Adjust the image size relative to its original dimensions (for raster)."
                if cairosvg_available:
                    scale_help += " Sets render scale (for SVG)."
                scale_percent_selected = st.slider(
                    "Scale Percentage:",
                    1,
                    500,
                    100,
                    1,
                    "%d%%",
                    key="scale_percent_slider",
                    help=scale_help,
                )

        st.markdown("---")
        st.markdown(
            f"*Developer: Ajay Khanna*\n\n*Place: LANL*\n\n*Date: May 12, 2025*"
        )
        st.markdown("---")
        req_caption = "Requires: `streamlit`, `Pillow`"
        req_caption += (
            ", `CairoSVG` (+ system dependencies)"
            if cairosvg_available
            else " (Optional: `CairoSVG` for SVG support)"
        )
        st.caption(req_caption)

    return {
        "output_format": output_format_selected,
        "output_format_lower": output_format_lower_selected,
        "enable_resize": enable_resize_selected,
        "resize_mode": resize_mode_selected,
        "target_width": target_width_selected,
        "target_height": target_height_selected,
        "scale_percent": scale_percent_selected,
        "resample_method_key": resample_method_key_selected,
        "png_compression_level": png_compression_level,
        "jpeg_quality": jpeg_quality,
    }


def render_main_page_content():
    """Renders the main page title, intro, file uploader, and preview queue. Returns files for processing."""
    app_title = "🖼️ Multi-Format Image Converter" + (
        " (with SVG Input)" if cairosvg_available else ""
    )
    st.title(app_title)
    st.markdown(
        f"""
    Upload one or more images (including **TGA** files{' and **SVG**' if cairosvg_available else ''}).
    Configure your desired conversion options in the sidebar on the left.
    Supported input formats: **{", ".join(SUPPORTED_INPUT_FORMATS).upper()}**.
    Supported output formats: **{", ".join(SUPPORTED_OUTPUT_FORMATS)}**.
    Click 'Convert Images' to process and download the results.
    """
    )
    if not cairosvg_available:
        st.warning(
            "SVG input support is disabled because the 'CairoSVG' library or its dependencies were not found.",
            icon="⚠️",
        )
    st.markdown("---")

    # Initialize session state for the processing queue if it doesn't exist
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []

    if "uploader_key" not in st.session_state:  # Changed key name for clarity
        st.session_state.uploader_key = 0

    uploader_label = (
        f"Upload Image Files ({', '.join(SUPPORTED_INPUT_FORMATS).upper()})"
    )
    uploader_help = "You can drag and drop multiple files here." + (
        " (including SVG)" if cairosvg_available else ""
    )

    # File uploader widget - its key changes to allow "resetting" its visual state
    newly_uploaded_files = st.file_uploader(
        uploader_label,
        type=SUPPORTED_INPUT_FORMATS,
        accept_multiple_files=True,
        help=uploader_help,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # Add newly uploaded files to our processing_queue, avoiding duplicates
    if newly_uploaded_files:  # This means user just uploaded new files via the widget
        files_were_added_to_queue = False
        # Get IDs of files already in our managed queue
        existing_file_ids_in_queue = {
            f.file_id for f in st.session_state.processing_queue
        }

        for new_file_obj in newly_uploaded_files:
            if new_file_obj.file_id not in existing_file_ids_in_queue:
                st.session_state.processing_queue.append(new_file_obj)
                existing_file_ids_in_queue.add(
                    new_file_obj.file_id
                )  # Update set for current batch
                files_were_added_to_queue = True

        if files_were_added_to_queue:
            # Important: After adding to our queue, we "clear" the uploader widget
            # by changing its key. This makes it ready for the *next* distinct upload action.
            st.session_state.uploader_key += 1
            st.rerun()  # Rerun to reflect new files in queue and clear uploader display

    # "Clear Upload Queue" button
    if st.button("Clear Upload Queue", key="clear_all_button_main"):
        st.session_state.processing_queue = []
        st.session_state.uploader_key += 1  # Reset uploader as well
        st.rerun()

    # Display previews if the processing queue is not empty
    if st.session_state.processing_queue:
        st.subheader("Image Preview Queue")

        cols_per_row = 5

        # Create a list of lists, where each inner list is a row of files from our managed queue
        rows_of_files = [
            st.session_state.processing_queue[i : i + cols_per_row]
            for i in range(0, len(st.session_state.processing_queue), cols_per_row)
        ]

        for row_idx, row_files in enumerate(rows_of_files):
            cols = st.columns(len(row_files))
            for col_idx, file_obj_in_queue in enumerate(row_files):
                with cols[col_idx]:
                    file_identifier_key = file_obj_in_queue.file_id
                    file_ext = os.path.splitext(file_obj_in_queue.name.lower())[1]

                    try:
                        # Ensure file pointer is at the beginning for preview
                        file_obj_in_queue.seek(0)
                        if file_ext == ".svg" and cairosvg_available:
                            svg_bytes = file_obj_in_queue.getvalue()  # Reads the file
                            file_obj_in_queue.seek(0)  # Reset pointer after getvalue
                            png_preview_bytes = cairosvg.svg2png(
                                bytestring=svg_bytes, output_width=120
                            )
                            st.image(
                                png_preview_bytes,
                                caption=(
                                    f"{file_obj_in_queue.name[:15]}..."
                                    if len(file_obj_in_queue.name) > 15
                                    else file_obj_in_queue.name
                                ),
                            )
                        elif file_ext in [
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".bmp",
                            ".gif",
                            ".webp",
                            ".tga",
                        ]:
                            # st.image can take the UploadedFile object directly.
                            # It's generally good about not consuming it if it's an UploadedFile.
                            st.image(
                                file_obj_in_queue,
                                width=120,
                                caption=(
                                    f"{file_obj_in_queue.name[:15]}..."
                                    if len(file_obj_in_queue.name) > 15
                                    else file_obj_in_queue.name
                                ),
                            )
                        else:
                            st.info(f"No preview for {file_obj_in_queue.name}")

                        if st.button(
                            "✖ Remove",
                            key=f"remove_{file_identifier_key}",
                            help=f"Remove {file_obj_in_queue.name}",
                        ):
                            st.session_state.processing_queue = [
                                f
                                for f in st.session_state.processing_queue
                                if f.file_id != file_identifier_key
                            ]
                            # Only rerun to update the preview list. Do not change uploader key here.
                            st.rerun()
                    except Exception as e:
                        st.error(f"Preview error for {file_obj_in_queue.name}: {e}")
        st.markdown("---")

    return st.session_state.processing_queue


# --- Image Processing Functions ---
def handle_svg_conversion(svg_bytes, options):
    """Converts SVG bytes to target format bytes using CairoSVG."""
    render_width, render_height, scale_factor = None, None, 1.0

    if options["enable_resize"]:
        if options["resize_mode"] == "Resize (Pixels)":
            if (
                not options["target_width"]
                or not options["target_height"]
                or options["target_width"] < 1
                or options["target_height"] < 1
            ):
                raise ValueError("Invalid target width or height for SVG rendering.")
            render_width, render_height = (
                options["target_width"],
                options["target_height"],
            )
        elif options["resize_mode"] == "Rescale (Percent)":
            if options["scale_percent"] <= 0:
                raise ValueError("Scale percentage must be positive.")
            scale_factor = options["scale_percent"] / 100.0

    if options["output_format_lower"] == "png":
        png_bytes_io = io.BytesIO()
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=png_bytes_io,
            output_width=render_width,
            output_height=render_height,
            scale=scale_factor if render_width is None else 1.0,
        )
        return png_bytes_io.getvalue(), True

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
    if img is None:
        raise ValueError("Failed to load intermediate PNG from SVG.")
    return img, False


def handle_raster_conversion(image_file, options):
    """Converts a raster image file using Pillow, applying transformations."""
    image_file.seek(0)  # Ensure pointer is at the start before Image.open
    img = Image.open(image_file)

    if options["enable_resize"]:
        current_width, current_height = img.size
        resample_filter = RESIZE_METHODS[options["resample_method_key"]]
        new_width, new_height = current_width, current_height

        if options["resize_mode"] == "Resize (Pixels)":
            if (
                not options["target_width"]
                or not options["target_height"]
                or options["target_width"] < 1
                or options["target_height"] < 1
            ):
                raise ValueError("Invalid target width or height for resizing.")
            new_width, new_height = options["target_width"], options["target_height"]
        elif options["resize_mode"] == "Rescale (Percent)":
            if options["scale_percent"] <= 0:
                raise ValueError("Scale percentage must be positive.")
            scale_factor = options["scale_percent"] / 100.0
            new_width, new_height = max(1, int(current_width * scale_factor)), max(
                1, int(current_height * scale_factor)
            )

        if (new_width, new_height) != (current_width, current_height):
            try:
                img = img.resize((new_width, new_height), resample_filter)
            except Exception as resize_err:
                raise ValueError(f"Failed to resize: {resize_err}")
    return img, False


def process_single_file(uploaded_file, options):
    """
    Processes a single uploaded file (SVG or raster).
    Returns a tuple: (new_filename, image_bytes) or raises an exception.
    Manages closing of PIL.Image object if created.
    """
    img_pil_object = None
    img_bytes_result = None
    skip_pillow_save = False
    file_extension = os.path.splitext(uploaded_file.name.lower())[1]

    try:
        # Ensure file pointer is at the beginning before any processing
        uploaded_file.seek(0)

        if file_extension == ".svg":
            if not cairosvg_available:
                raise ValueError(
                    "SVG processing skipped: CairoSVG library not available."
                )
            svg_data = uploaded_file.getvalue()  # Reads the file
            uploaded_file.seek(0)  # Reset pointer after getvalue for potential reuse
            if not svg_data:
                raise ValueError("SVG file appears to be empty.")

            result, skip_pillow_save = handle_svg_conversion(svg_data, options)
            if skip_pillow_save:
                img_bytes_result = result
            else:
                img_pil_object = result
        else:
            # For raster, handle_raster_conversion itself will do seek(0) before Image.open
            img_pil_object, skip_pillow_save = handle_raster_conversion(
                uploaded_file, options
            )

        if not skip_pillow_save:
            if img_pil_object is None:
                raise ValueError("Image object is missing before final save.")
            if options["output_format_lower"] in ["jpeg", "jpg"] and (
                img_pil_object.mode == "RGBA" or img_pil_object.mode == "P"
            ):
                st.warning(
                    f"'{uploaded_file.name}' has transparency; converting to RGB for JPEG output.",
                    icon="⚠️",
                )
                img_pil_object = img_pil_object.convert("RGB")

            img_byte_arr = io.BytesIO()
            try:
                if options["output_format_lower"] == "png":
                    img_pil_object.save(
                        img_byte_arr,
                        format="PNG",
                        compress_level=options.get(
                            "png_compression_level", 6
                        ),  # User-selected
                    )
                elif options["output_format_lower"] in ["jpeg", "jpg"]:
                    img_pil_object.save(
                        img_byte_arr,
                        format="JPEG",
                        quality=options.get("jpeg_quality", 85),  # User-selected
                    )
                else:
                    img_pil_object.save(
                        img_byte_arr, format=options["output_format"].upper()
                    )
            except KeyError:
                raise ValueError(
                    f"Selected output format '{options['output_format'].upper()}' is not supported by Pillow."
                )
            except Exception as save_err:
                raise ValueError(
                    f"Failed to save image as {options['output_format'].upper()}: {save_err}"
                )
            img_bytes_result = img_byte_arr.getvalue()

        if img_bytes_result:
            new_filename = generate_new_filename(
                uploaded_file.name, options["output_format_lower"]
            )
            return new_filename, img_bytes_result
        raise ValueError("Failed to generate final image bytes.")
    finally:
        if img_pil_object:
            try:
                img_pil_object.close()
            except Exception:
                pass


def run_conversion_process(files_to_process, options):
    """Manages the conversion process for all uploaded files."""
    processed_images_list = []
    error_messages_list = []
    success_count = 0
    total_count = len(files_to_process)

    if total_count == 0:
        st.info("No files in the queue to process.")
        return [], [], 0, 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file_obj in enumerate(files_to_process):
        status_text.text(
            f"Processing '{uploaded_file_obj.name}' ({i+1}/{total_count})..."
        )
        progress_bar.progress((i + 1) / total_count)

        try:
            new_filename, img_bytes = process_single_file(uploaded_file_obj, options)
            processed_images_list.append((new_filename, img_bytes))
            success_count += 1
        except UnidentifiedImageError:
            msg = f"❌ Error: '{uploaded_file_obj.name}' is not a valid or supported image file (excluding SVG issues), or it might be corrupted."
            error_messages_list.append(msg)
            st.error(msg, icon="❗")
        except ValueError as ve:
            msg = f"❌ Error processing '{uploaded_file_obj.name}': {ve}"
            error_messages_list.append(msg)
            if "SVG processing skipped" in str(ve):
                st.warning(
                    f"Skipped '{uploaded_file_obj.name}': SVG support disabled.",
                    icon="ℹ️",
                )
            else:
                st.error(msg, icon="❗")
        except Exception as e:
            msg = f"❌ An unexpected error occurred with '{uploaded_file_obj.name}': {type(e).__name__} - {e}"
            error_messages_list.append(msg)
            st.error(msg, icon="❗")

    processing_duration = time.time() - st.session_state.start_time
    status_text.text(f"Processing complete in {processing_duration:.2f} seconds.")
    return processed_images_list, error_messages_list, success_count, total_count


# --- Output and Summary Functions ---
def display_results_and_downloads(
    processed_images, errors, successes, total_files, output_format_lower
):
    """Displays the conversion summary and download buttons."""
    st.markdown("---")
    st.subheader("📊 Conversion Summary")

    if successes > 0:
        st.success(f"Successfully converted {successes} out of {total_files} image(s).")
    if errors:
        st.warning(
            f"Encountered errors or skipped files for {len(errors)} item(s). See messages above/below."
        )
        with st.expander("Show Error/Skipped File Details"):
            for msg in errors:
                if "skipped" in msg.lower() or "Skipped" in msg:
                    st.info(msg)
                else:
                    st.error(msg)

    if processed_images:
        if successes == 1:
            filename, file_bytes = processed_images[0]
            mime = MIME_TYPES.get(output_format_lower, "application/octet-stream")
            st.download_button(
                f"📥 Download Converted File ({filename})",
                file_bytes,
                filename,
                mime,
                key="download_single_file_btn",
            )
            st.markdown("*(Or download as a ZIP archive below)*")

        zip_buffer = create_zip_archive(processed_images)
        if zip_buffer:
            st.download_button(
                f"📥 Download All {successes} Converted Image(s) (ZIP)",
                zip_buffer.getvalue(),
                f"converted_images_{output_format_lower}.zip",
                "application/zip",
                key="download_zip_archive_btn",
            )
        elif successes > 0:
            st.error("Could not create the ZIP archive for multiple files.")
    elif total_files > 0 and successes == 0:
        st.warning("No images were successfully processed to download.")


# --- Main Application ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Image Converter")

    sidebar_options = render_sidebar()
    files_in_queue_for_conversion = render_main_page_content()

    if files_in_queue_for_conversion:
        if st.button("🚀 Convert Images in Queue", key="convert_button_main"):
            if not files_in_queue_for_conversion:
                st.warning("Upload queue is empty. Please add files to convert.")
            else:
                st.session_state.start_time = time.time()
                processed_images, errors, successes, total = run_conversion_process(
                    files_in_queue_for_conversion, sidebar_options
                )
                display_results_and_downloads(
                    processed_images,
                    errors,
                    successes,
                    total,
                    sidebar_options["output_format_lower"],
                )
    else:
        st.info("☝️ Upload image files using the uploader above to get started.")


if __name__ == "__main__":
    main()
