# Face Recognition Authentication

An educational local prototype for one-to-one face verification. Users will enroll with three
face images and later authenticate with an email address and a new face image.

> [!WARNING]
> This project is not production-ready or banking-grade authentication. The MVP does not include
> liveness detection and can be vulnerable to photograph or screen-replay attacks.

## Current status

Project bootstrap only. The repository currently provides the Python environment, shared paths,
quality tooling, and project documentation. ML training, API, and UI dependencies will be added in
their implementation milestones.

## Setup

Requirements:

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)

Create and synchronize the local environment:

```powershell
uv sync
```

Confirm the interpreter:

```powershell
uv run python --version
```

## Quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

## Planned structure

Python modules live directly in `src/`; there is no nested package directory.

```text
face-recognition/
├── data/                 # CelebA and generated manifests; ignored
├── models/               # YuNet model; ignored
├── checkpoints/          # Trained embedding checkpoint; ignored
├── notebooks/
├── src/
│   ├── config.py
│   ├── data.py           # planned
│   ├── vision.py         # planned
│   ├── model.py          # planned
│   ├── training.py       # planned
│   ├── evaluation.py     # planned
│   ├── engine.py         # planned
│   ├── storage.py        # planned
│   ├── security.py       # planned
│   ├── api.py            # planned
│   └── ui.py             # planned
└── tests/
```

Future entry points will run without installing a Python package:

```powershell
uv run python src/training.py
uv run fastapi dev src/api.py
uv run streamlit run src/ui.py
```

## Documentation

Local planning documents are kept in the ignored `context/` directory:

- `context/prd.md`
- `context/implementation-plan.md`

The project is local-only and intended for non-commercial learning and experimentation.
