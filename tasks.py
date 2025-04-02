from invoke import task


@task
def ray(c, head_port: int = 6379, dashboard_port: int = 8265):
    """Start a local Ray cluster."""
    c.run(f"ray start --head --port={head_port} --dashboard-port={dashboard_port}")
