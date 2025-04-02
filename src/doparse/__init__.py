from enum import Enum


class ExportFormat(str, Enum):
    Markdown: str = "md"
    HTML: str = "html"
