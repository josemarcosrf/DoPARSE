import click
import ray
from ray import serve
from ray.serve.config import HTTPOptions

from src.doparse import ServiceTypes


def ray_connect(ray_address: str, blocking: bool | None = None):
    # Connect to existing Ray cluster or start a new one
    started_new_cluster = False
    if not ray.is_initialized():
        try:
            ray.init(address=ray_address)
            if ray_address and ray_address != "auto":
                print(f"🔌 Connected to Ray cluster at {ray_address}")
            elif ray_address == "auto":
                # Check if we actually connected or started new
                try:
                    # If auto connected to existing,
                    # there will be existing deployments or info
                    existing_apps = serve.status().applications
                    print(
                        "🔌 Connected to existing Ray cluster. "
                        f"Found applications: {list(existing_apps.keys())}"
                    )
                except Exception:
                    print("💫 Started new Ray cluster")
                    started_new_cluster = True
            else:
                print("Started new Ray cluster")
                started_new_cluster = True
        except Exception as e:
            print(f"Failed to connect to Ray: {e}")
            print("💫 Starting new Ray cluster")
            ray.init()
            started_new_cluster = True
    else:
        print("Ray already initialized in this process")

    # Auto-determine blocking behavior if not explicitly set
    if blocking is None:
        blocking = started_new_cluster
        if blocking:
            print("Blocking mode: keeping process alive (started new cluster)")
        else:
            print(
                "Non-blocking mode: will exit after deployment "
                "(connected to existing cluster)"
            )

    return blocking


@click.command()
@click.argument(
    "service",
    type=click.Choice([e.value for e in ServiceTypes], case_sensitive=True),
    default=ServiceTypes.Docling,
)
@click.option("--gpus", type=float, default=0, help="Number of GPUs to use.")
@click.option("--cpus", type=float, default=4, help="Number of CPUs to use.")
@click.option("--port", default=8080, help="Port to run the server on.")
@click.option(
    "--ray-address",
    default="auto",
    help="Ray cluster address to connect to (e.g., 'auto' or 'ray://localhost:10001'). Use 'auto' to connect to existing cluster.",
)
@click.option(
    "--blocking/--no-blocking",
    default=None,
    help=(
        "Whether to block after deploying the service. "
        "If not set, blocks only when starting a new Ray cluster."
    ),
)
def main(
    service: str, gpus: float, cpus: int, port: int, ray_address: str, blocking: bool
):
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
    else:
        raise ValueError(f"Unknown service type: {service}")

    # Connect to Ray cluster and determine blocking behavior
    blocking = ray_connect(ray_address, blocking)

    # Start or connect to Ray Serve
    try:
        serve.start(http_options=HTTPOptions(port=port), _blocking=False)
        print(f"💫 Started Ray Serve on port {port}")
    except RuntimeError:
        print("🔌 Ray Serve already running, connecting to it")

    # Deploy the service
    serve.run(
        converter,
        blocking=blocking,
        name=f"{service}-ocr",
        route_prefix=f"/{service}",
    )
    print(f"Deployed {service} service at route /{service}")


if __name__ == "__main__":
    main()
