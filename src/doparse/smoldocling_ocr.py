import tempfile
from pathlib import Path

import PIL
import torch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from fastapi import FastAPI, HTTPException, UploadFile
from ray import serve

from src.doparse import ExportFormat
from src.doparse.utils import build_vllm_pipeline_options

# Define a FastAPI app
# NOTE There's actually a docling server.
# Might be better to use that instead of this one.
# https://github.com/docling-project/docling-serve
app = FastAPI(title="smoldocling-ocr", root_path="/docling")


@serve.deployment(name="docling-ocr")
@serve.ingress(app)
class smolDoclingConverter:
    PROMPT_TEXT = "Convert page to Docling."
    CHAT_TEMPLATE = "<|im_start|>User:<image>{PROMPT_TEXT}<end_of_utterance>Assistant:"

    def __init__(self):
        print("🧙‍♂️ Initializing smolDocling converter...")
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=build_vllm_pipeline_options(),
                ),
            }
        )

    def _init_llm(
        self,
        model_id: str = "ds4sd/SmolDocling-256M-preview",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        from vllm import LLM, SamplingParams

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = (
            "bfloat16" if torch.cuda.get_device_properties(0).major >= 8 else "float16"
        )
        print(f"📠 Using device: {self.device} | dtype: {self.dtype}")

        # Initialize LLM
        self.llm = LLM(
            model=model_id,
            limit_mm_per_prompt={"image": 1},
            tensor_parallel_size=1,  # Use 1 GPU
            device=self.device,
            dtype=self.dtype,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens
        )

    def _image_to_doctags(self, image: PIL.Image):
        prompt = self.CHAT_TEMPLATE.format(PROMPT_TEXT=self.PROMPT_TEXT)
        llm_input = {"prompt": prompt, "multi_modal_data": {"image": image}}
        output = self.llm.generate([llm_input], sampling_params=self.sampling_params)[0]
        doctags = output.outputs[0].text

        return doctags

    def _convert_from_pdf_images(self, pdf_path: Path):
        from doparse.utils import pdf_to_images

        # Check it is a PDF file
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"❌ {pdf_path} is not a PDF file.")

        # Convert PDF to images
        images = pdf_to_images(pdf_path)

        doctags_list = []
        for i, image in enumerate(images):
            print(f"🔖 Converting page {i + 1} to doctags...")
            doctags = self._image_to_doctags(image)
            doctags_list.append(doctags)

        # To convert to Docling Document, MD, HTML, etc.
        print(f"🪄 Converting {pdf_path} to Docling Document...")
        doc_name = pdf_path.stem
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(doctags_list, images)
        doc = DoclingDocument(name=doc_name)
        doc.load_from_doctags(doctags_doc)

        return doc.export_to_markdown()

    def _convert(
        self, file_path: Path, export_format: ExportFormat = ExportFormat.Markdown
    ):
        print(f"Using docling to convert {file_path} ➡️ {export_format.name}")
        result = self.converter.convert(file_path)
        if export_format == ExportFormat.Markdown:
            return result.document.export_to_markdown()
        elif export_format == ExportFormat.HTML:
            return result.document.export_to_html()
        else:
            raise ValueError(f"Unsupported format: {export_format}")

    @app.post("/upload")
    async def convert_from_upload(self, file: UploadFile) -> str:
        """Upload PDF files and convert them to markdown
        This endpoint accepts multiple PDF files and converts each to markdown format.
        It uses the `smolDocling` model to perform the conversion.

        Args:
            files (UploadFile): The file to be converted.

        Returns:
            str: The conversion result, text markdown
        """
        try:
            # Create a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(
                delete=True, suffix=Path(file.filename).suffix
            ) as tmp_file:
                tmp_file.write(await file.read())
                return self._convert(Path(tmp_file.name))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"💥 Failed to convert {file} to markdown: {e}",
            )

    @app.get("/local")
    async def convert_from_local(self, pdf_path: str) -> str:
        """Convert PDF files to text
        This endpoint accepts multiple PDF file paths and converts each to markdown format.
        It uses the `smolDocling` model to perform the conversion.

        Args:
            pdf_paths (list[str]): The list of PDF file paths to be converted.

        Returns:
            dict: A dictionary mapping each file path to its conversion result.
        """
        try:
            return self._convert(Path(pdf_path))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"💥 Failed to convert {pdf_path} to markdown: {e}",
            )


# # Deploy (without parameters for now - default values are used)
# converter = smolDoclingConverter.options(
#     ray_actor_options={"num_gpus": 0.5, "num_cpus": 4},
# ).bind()
