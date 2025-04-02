from pathlib import Path


def pdf_to_images(pdf_path: Path, zoom=2):
    """
    Convert each page of a PDF to an image.

    Args:
        pdf_path (Path): Path to the PDF file.
        zoom (int): Zoom factor to apply to the PDF pages.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)  # Define the zoom factor
    images = []

    print(f" 🖨️ Printing {len(doc)} pages from {pdf_path}...")
    for page_num in range(len(doc)):
        print(f"📄 Extracting page {page_num + 1}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


def build_vllm_pipeline_options():
    """Build VLLM pipeline options for docling Document conversion.
    Tries to detect the system architecture and set the appropriate VLM model.
    If running on Apple Silicon, it uses a fast implementation for SmolDocling-256M via MLX.
    If running on a different architecture, it defaults to SmolDocling-256M.
    It also sets the accelerator options based on the availability of CUDA.

    Returns:
        VlmPipelineOptions: The configured VLM pipeline options.
    """
    import platform

    import torch
    from docling.datamodel.pipeline_options import (  # granite_vision_vlm_conversion_options,
        AcceleratorDevice,
        smoldocling_vlm_conversion_options,
        smoldocling_vlm_mlx_conversion_options,
        VlmPipelineOptions,
    )

    pipeline_options = VlmPipelineOptions()

    # On GPU systems, enable flash_attention_2 with CUDA
    if torch.cuda.is_available():
        print("🚀 Using CUDA for GPU acceleration.")
        pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
        # pipeline_options.accelerator_options.cuda_use_flash_attention2 = True

    ## Pick a VLM model:
    if "arm" in platform.processor():
        # Fast Apple Silicon friendly implementation for SmolDocling-256M via MLX
        print("🍏 Using Apple Silicon implementation for SmolDocling-256M via MLX.")
        pipeline_options.vlm_options = smoldocling_vlm_mlx_conversion_options
    else:
        # Otherwise, we choose SmolDocling-256M by default
        print("🖥️ Using default SmolDocling-256M model for VLM conversion.")
        pipeline_options.vlm_options = smoldocling_vlm_conversion_options

    return pipeline_options
