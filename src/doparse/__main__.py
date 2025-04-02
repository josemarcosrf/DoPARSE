import click
from ray import serve

from src.doparse import ServiceTypes


@click.command()
@click.argument(
    "service", type=click.Choice(ServiceTypes), default=ServiceTypes.Docling
)
@click.option("--gpus", type=float, default=0.5, help="Number of GPUs to use.")
@click.option("--cpus", type=float, default=4, help="Number of CPUs to use.")
@click.option("--port", default=8000, help="Port to run the server on.")
def main(service: str, gpus: float, cpus: int, port: int):
    """Run the FastAPI app with Ray Serve."""
    if service == ServiceTypes.Docling:
        from src.doparse.smoldocling_ocr import smolDoclingConverter

        converter = smolDoclingConverter.options(
            ray_actor_options={"num_gpus": gpus, "num_cpus": cpus},
        ).bind()
    elif service == ServiceTypes.Marker:
        from src.doparse.marker_ocr import MarkerPDFConverter

        converter = MarkerPDFConverter.options(
            ray_actor_options={"num_gpus": gpus, "num_cpus": cpus},
        ).bind()

    # Start Ray Serve
    serve.run(
        converter,
        blocking=True,
        name=f"{service.value}-ocr",
        route_prefix=f"/{service.value}",
    )


if __name__ == "__main__":
    main()
