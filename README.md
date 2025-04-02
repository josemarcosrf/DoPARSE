# DoPARSE: **Do**cument **Pa**rsing via **R**ay **Se**rve

This repo contains document parsing APIs for various document types.
The APIs are built using [Ray](https://docs.ray.io/en/master/index.html) Serve, a scalable and flexible library for serving models and other Python objects.

## HowTo

### Requirements

- python 3.12

### 💿 Install

```bash
# Install pdm and uv
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configure PDM to use UV
pdm config use_uv true
pdm config python.install_root $(uv python dir)

# Install python deps
pdm sync -G :all
```

### 🏃‍➡️ Run

We use [invoke](https://www.pyinvoke.org/) to simplify deploying the services:

```bash
python -m src.doparse docling
```

If you want to run Ray without running the parser serving:

```bash
inv ray     # start ray local cluster
```

## Services

Currently the following APIs are available:

- Marker: [marker-pdf](https://github.com/VikParuchuri/marker)-based parsing

- Docling: [docling](https://docling-project.github.io/docling/#features)-based parsing. Uses [smolDocling](https://huggingface.co/ds4sd/SmolDocling-256M-preview) to create doc tags and docling to export a markdown version of the document.

Both Ray Serve deployments expose the following endpoints:

- `/upload`: Upload a documents to parse and return the parsed output as markdown
- `/convert`: Takes a local document path to parse and return the parsed output as markdown
