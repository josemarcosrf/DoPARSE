from invoke import task

@task
def ray(c, head_port:int=6379, dashboard_port:int=8265):
    """Start a local Ray cluster."""
    c.run(
        f"ray start --head --port={head_port} --dashboard-port={dashboard_port}"
    )

@task
def marker(c):
    """Start the marker OCR service."""
    c.run(
        ".venv/bin/serve run -r src.doparse.marker_ocr:converter "
        f"--route-prefix '/marker' --name marker"
    )

@task
def docling(c):
    """Start the docling OCR service."""
    c.run(
        ".venv/bin/serve run -r src.doparse.smoldocling_ocr:converter "
        f"--route-prefix '/docling' --name docling"
    )
