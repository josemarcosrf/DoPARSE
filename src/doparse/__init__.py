from enum import Enum


class ServiceTypes(str, Enum):
    Marker: str = "marker"
    Docling: str = "docling"


class ExportFormat(str, Enum):
    Markdown: str = "md"
    HTML: str = "html"
