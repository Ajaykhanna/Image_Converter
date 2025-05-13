# Blueprint: Streamlit Multi-Format Image Converter

This document outlines the development steps for creating a Streamlit application that allows users to convert images between various formats, with options for resizing, rescaling, and previewing uploaded files.

Developer: Ajay Khanna
Places: LANL
Date: May 12, 2025

## I. Project Setup & Initial Configuration

**Create Project Files**:

    image_converter_app.py: Main Python script for the Streamlit application.

    (Optional, for GitHub) requirements.txt: To list project dependencies.

    (Optional, for GitHub) README.md: Project description and instructions.

**Install Core Libraries**:

    streamlit: For building the web application UI.

    Pillow: For raster image processing (opening, resizing, saving various formats like PNG, JPG, TGA, BMP, GIF, WEBP).

    Command: pip install streamlit Pillow

**Optional Library for SVG Support**:

    CairoSVG: For converting SVG (vector) images to raster formats (like PNG).

    Command: pip install CairoSVG

    Note System Dependencies: CairoSVG relies on the Cairo 2D graphics library. This needs to be installed on the system:

        macOS (Homebrew): brew install cairo

        Debian/Ubuntu: sudo apt-get install libcairo2-dev

        Fedora: sudo yum install cairo-devel

        Windows: Requires manual installation of GTK+ or finding pre-compiled Cairo binaries.

**Import Necessary Modules in image_converter_app.py**:

    streamlit as st

    PIL.Image, PIL.UnidentifiedImageError

    os (for path manipulation, e.g., getting file extensions)

    io (for handling bytes in memory, e.g., io.BytesIO)

    zipfile (for creating downloadable ZIP archives of multiple converted images)

    time (for tracking processing duration)

    sys (for printing messages to standard output, e.g., CairoSVG availability)

    CairoSVG Availability Check (Graceful Degradation):

**At the beginning of the script, attempt to import cairosvg**.

    Use a try-except block:

        try: Import cairosvg and perform a minimal runtime check (e.g., convert a tiny dummy SVG string to PNG bytes). If successful, set a global boolean flag like cairosvg_available = True. Print a console message indicating SVG support is enabled.

        except ImportError: If cairosvg is not installed, set cairosvg_available = False. Print an informative message to sys.stdout guiding the user on how to install it and its dependencies for SVG support.

        except Exception as e: If cairosvg is imported but the runtime check fails (e.g., missing system library), set cairosvg_available = False. Print a warning to sys.stdout about the failure.

    This flag will be used to conditionally enable SVG-related features in the UI and processing logic.

**Global Configurations**:

    SUPPORTED_RASTER_INPUT_FORMATS: List of raster file extensions (e.g., ['png', 'jpg', 'tga']).

    SUPPORTED_INPUT_FORMATS: Dynamically created list. Start with SUPPORTED_RASTER_INPUT_FORMATS. If cairosvg_available is True, add 'svg' to this list.

    SUPPORTED_OUTPUT_FORMATS: List of output raster file extensions (e.g., ['PNG', 'JPEG', 'TGA']). SVG will not be an output option.

    DEFAULT_OUTPUT_FORMAT: Set to 'PNG'.

    RESIZE_METHODS: A dictionary mapping user-friendly names (e.g., "LANCZOS (High Quality)") to Image.Resampling constants from Pillow (e.g., Image.Resampling.LANCZOS).

    MIME_TYPES: A dictionary mapping lowercase file extensions to their corresponding MIME types (e.g., 'png': 'image/png') for the single file download feature.

## II. Helper Functions

    create_zip_archive(image_data_list):

        Input: image_data_list (a list of tuples, where each tuple is (filename_str, image_bytes)).

        Functionality:

            If image_data_list is empty, return None.

            Create an io.BytesIO object to act as an in-memory buffer for the ZIP file.

            Use zipfile.ZipFile to create a new ZIP archive in this buffer (mode 'w', compression zipfile.ZIP_DEFLATED).

            Iterate through image_data_list. For each (filename, image_bytes) pair, add the image to the archive using zip_file.writestr(filename, image_bytes).

            Handle potential exceptions during ZIP creation and display an error using st.error().

        Output: The io.BytesIO object containing the ZIP data, or None on error.

    generate_new_filename(original_filename, target_format_lower):

        Input: original_filename (string), target_format_lower (string, e.g., "png").

        Functionality:

            Use os.path.splitext(original_filename) to get the base name without the original extension.

            Construct the new filename: f"{base_name}_converted.{target_format_lower}".

        Output: The new filename string.

## III. UI Rendering Functions

    render_sidebar():

        Functionality: Creates all widgets within st.sidebar.

        Add a header: st.header("⚙️ Conversion Options").

        Output Format: st.selectbox for choosing the target format from SUPPORTED_OUTPUT_FORMATS, defaulting to DEFAULT_OUTPUT_FORMAT.

        Resizing/Rescaling Toggle: st.checkbox to enable/disable resizing features.

        Conditional Resizing/Rescaling Options (if checkbox is checked):

            st.radio to select mode: "Resize (Pixels)" or "Rescale (Percent)".

            st.selectbox for "Resampling Method (Raster only)" from RESIZE_METHODS.keys(). Add help text explaining its applicability (raster only, not for SVG rendering size).

            If "Resize (Pixels)": Two st.number_input widgets (for Width and Height), with appropriate min_value, default value, and step. Use st.columns(2) for layout.

            If "Rescale (Percent)": st.slider for Percentage (e.g., 1% to 500%).

        Developer Information: Display developer name, place, and date using st.markdown.

        Requirements Caption: Display required/optional libraries.

        Output: A dictionary containing all selected sidebar options (e.g., {"output_format": "PNG", "enable_resize": True, ...}).

    render_main_page_content():

        Functionality: Renders the main application area, including title, instructions, file uploader, and the preview queue.

        Set the application title using st.title(), dynamically including "(with SVG Input)" if cairosvg_available.

        Display introductory markdown text explaining app usage, dynamically listing supported input formats.

        If cairosvg_available is False, display an st.warning about disabled SVG support.

        Session State Initialization:

            Initialize st.session_state.processing_queue = [] if it doesn't exist (this list will hold UploadedFile objects ready for conversion).

            Initialize st.session_state.uploader_key = 0 if it doesn't exist (used to reset the st.file_uploader widget).

        File Uploader:

            Create an st.file_uploader widget.

                Set label dynamically based on SUPPORTED_INPUT_FORMATS.

                Set type=SUPPORTED_INPUT_FORMATS.

                Set accept_multiple_files=True.

                Set key=f"file_uploader_{st.session_state.uploader_key}". This dynamic key allows the widget to be "reset" by incrementing st.session_state.uploader_key.

        Adding Files to Processing Queue:

            If the st.file_uploader returns new files:

                Iterate through these newly_uploaded_files.

                For each new file, check if its file_id is already in st.session_state.processing_queue (to avoid duplicates).

                If not a duplicate, append the new file object to st.session_state.processing_queue.

                If any new files were added to the queue, increment st.session_state.uploader_key and call st.rerun(). This clears the st.file_uploader's display (as it gets a new key) and makes it ready for the next distinct upload action, while our processing_queue retains all chosen files.

        "Clear Upload Queue" Button:

            An st.button("Clear Upload Queue").

            If clicked:

                Set st.session_state.processing_queue = [].

                Increment st.session_state.uploader_key.

                Call st.rerun().

        Image Preview Queue (if st.session_state.processing_queue is not empty):

            Display a subheader: st.subheader("Image Preview Queue").

            Arrange previews in rows (e.g., using st.columns with 4-5 columns per row).

            For each file_obj_in_queue in st.session_state.processing_queue:

                Ensure file_obj_in_queue.seek(0) before reading for preview.

                Display a thumbnail using st.image() (width around 120px).

                    For SVGs (if cairosvg_available): Read file_obj_in_queue.getvalue(), convert to PNG bytes using cairosvg.svg2png(output_width=120), then display. Remember to seek(0) again after getvalue().

                    For raster images: Pass file_obj_in_queue directly to st.image().

                Display a truncated filename as a caption.

                Add an st.button("✖ Remove", key=f"remove_{file_obj_in_queue.file_id}").

                    If clicked: Remove the file from st.session_state.processing_queue by filtering based on file_id. Call st.rerun() to update the preview display. Do not change st.session_state.uploader_key here.

                Include try-except for preview generation to catch errors.

        Output: The current st.session_state.processing_queue (list of UploadedFile objects).

## IV. Image Processing Functions

    handle_svg_conversion(svg_bytes, options):

        Input: svg_bytes (raw bytes of the SVG file), options (dictionary from render_sidebar()).

        Functionality:

            Determine rendering dimensions (render_width, render_height) or scale_factor based on options (resizing/rescaling settings). Validate inputs.

            If options["output_format_lower"] == 'png':

                Use cairosvg.svg2png(bytestring=svg_bytes, write_to=io.BytesIO(), output_width/height/scale=...) to get PNG bytes directly.

                Return (png_bytes, True) where True indicates skip_pillow_save.

            Else (for other output formats like JPEG, WEBP):

                Convert SVG to intermediate PNG bytes using cairosvg.svg2png().

                Open these PNG bytes into a Pillow Image object: img = Image.open(io.BytesIO(png_bytes)).

                Return (img, False) where False indicates Pillow processing is still needed for final format saving.

        Output: A tuple (result, skip_pillow_save). result is either bytes (if PNG output) or a Pillow Image object.

    handle_raster_conversion(image_file_obj, options):

        Input: image_file_obj (Streamlit UploadedFile object for a raster image), options (dictionary).

        Functionality:

            Call image_file_obj.seek(0).

            Open the image: img = Image.open(image_file_obj).

            If options["enable_resize"] is true:

                Calculate new_width, new_height based on options["resize_mode"] (Pixels or Percent), options["target_width/height"] or options["scale_percent"]. Validate inputs.

                Get the resampling filter: resample_filter = RESIZE_METHODS[options["resample_method_key"]].

                Resize: img = img.resize((new_width, new_height), resample_filter). Handle exceptions.

        Output: A tuple (img, False). img is the (potentially resized) Pillow Image object. False indicates Pillow saving is needed.

    process_single_file(uploaded_file_obj, options):

        Input: uploaded_file_obj (Streamlit UploadedFile object), options (dictionary).

        Functionality:

            Initialize img_pil_object = None, img_bytes_result = None, skip_pillow_save = False.

            Call uploaded_file_obj.seek(0) before any read operation.

            Get file extension: file_extension = os.path.splitext(uploaded_file_obj.name.lower())[1].

            If file_extension == '.svg':

                Check cairosvg_available. If not, raise ValueError.

                Read SVG data: svg_data = uploaded_file_obj.getvalue(). Call uploaded_file_obj.seek(0) again.

                Call result, skip_pillow_save = handle_svg_conversion(svg_data, options).

                If skip_pillow_save is True, img_bytes_result = result. Else, img_pil_object = result.

            Else (raster image):

                Call img_pil_object, skip_pillow_save = handle_raster_conversion(uploaded_file_obj, options).

            If not skip_pillow_save (Pillow saving is needed):

                If img_pil_object is None, raise ValueError.

                If output is JPEG/JPG and img_pil_object has alpha (RGBA or P mode), convert to RGB: img_pil_object = img_pil_object.convert('RGB'). Display an st.warning about transparency loss.

                Save to an io.BytesIO buffer: img_pil_object.save(img_byte_arr, format=options["output_format"].upper()). Handle KeyError or other save exceptions.

                img_bytes_result = img_byte_arr.getvalue().

            If img_bytes_result is populated:

                Generate new_filename = generate_new_filename(uploaded_file_obj.name, options["output_format_lower"]).

                Return (new_filename, img_bytes_result).

            Else, raise ValueError("Failed to generate final image bytes.").

            Use a finally block to ensure img_pil_object.close() is called if img_pil_object was created.

        Output: Tuple (new_filename_str, image_bytes) or raises an exception.

## V. Main Conversion Orchestration

    run_conversion_process(files_to_process, options):

        Input: files_to_process (list of UploadedFile objects from st.session_state.processing_queue), options (dictionary).

        Functionality:

            Initialize processed_images_list = [], error_messages_list = [], success_count = 0.

            If files_to_process is empty, display st.info("No files in the queue...") and return empty results.

            Create st.progress(0) and status_text = st.empty().

            Store st.session_state.start_time = time.time().

            Iterate through files_to_process with an index i:

                Update status_text and progress_bar.

                Use a try-except block for each file:

                    try: Call new_filename, img_bytes = process_single_file(uploaded_file_obj, options). Append (new_filename, img_bytes) to processed_images_list, increment success_count.

                    except UnidentifiedImageError: Append error message to error_messages_list, display st.error.

                    except ValueError as ve: (Handles errors from process_single_file, including SVG skipping). Append error, display st.warning if "SVG processing skipped", else st.error.

                    except Exception as e: General error. Append, display st.error.

            Calculate processing_duration = time.time() - st.session_state.start_time.

            Update status_text with completion message and duration.

        Output: (processed_images_list, error_messages_list, success_count, total_files_processed).

## VI. Results Display

    display_results_and_downloads(processed_images, errors, successes, total_files, output_format_lower):

        Input: Results from run_conversion_process, and output_format_lower.

        Functionality:

            Display a subheader: st.subheader("📊 Conversion Summary").

            If successes > 0, display st.success(...).

            If errors list is not empty, display st.warning(...) and an st.expander to show detailed error messages.

            If processed_images is not empty:

                Single File Download (if successes == 1):

                    Get the single filename, file_bytes from processed_images[0].

                    Determine mime_type from MIME_TYPES dictionary.

                    Provide an st.download_button for the single file.

                    Add markdown: "(Or download as a ZIP archive below)".

                ZIP Archive Download (if successes > 0):

                    Call zip_buffer = create_zip_archive(processed_images).

                    If zip_buffer is not None, provide an st.download_button for the ZIP file.

                    Else (if zip_buffer is None but successes > 0), display st.error("Could not create ZIP archive...").

            Else if total_files > 0 and successes == 0: Display st.warning("No images successfully processed...").

## VII. Main Application Flow

    main() function:

        Call st.set_page_config(layout="wide", page_title="...").

        Call sidebar_options = render_sidebar().

        Call files_in_queue_for_conversion = render_main_page_content().

        If files_in_queue_for_conversion is not empty:

            Display an st.button("🚀 Convert Images in Queue").

            If the button is clicked:

                (Optional safeguard) Check again if files_in_queue_for_conversion is not empty.

                Call processed_images, errors, successes, total = run_conversion_process(files_in_queue_for_conversion, sidebar_options).

                Call display_results_and_downloads(processed_images, errors, successes, total, sidebar_options["output_format_lower"]).

        Else (queue is empty):

            Display st.info("☝️ Upload image files...").

    Script Execution Guard:

        Use if __name__ == "__main__": main().

This blueprint provides a comprehensive guide to rebuilding the application with its current features and modular structure.
