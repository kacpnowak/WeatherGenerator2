# Tech Stack: WeatherGenerator

This document outlines the core technology stack used in the WeatherGenerator project.

## Core Language
- **Python (>=3.12, <3.13)**: The primary programming language for the project, balancing ease of use with a rich ecosystem of scientific computing libraries.

## Machine Learning & AI
- **PyTorch**: The core deep learning framework used for building and training the Earth system models.
- **flash-attn**: Used for fast and memory-efficient exact attention, crucial for large-scale model training on GPUs.

## Data Processing & Numerics
- **NumPy & SciPy**: Foundational libraries for numerical operations.
- **Pandas & Polars**: Used for efficient data manipulation and analysis, with Polars providing high-performance multithreaded processing.
- **Zarr**: Format for storage of chunked, compressed, N-dimensional arrays, critical for handling large geospatial datasets.
- **Dask**: Enables parallel computing and scaling data processing pipelines out to clusters.

## Project Management & Tooling
- **uv**: Extremely fast Python package installer and resolver, managing dependencies and virtual environments.
- **hatchling**: Modern, extensible Python build backend.

## Testing & Quality Assurance
- **Pytest**: The framework used for writing and running tests.
- **Ruff**: An extremely fast Python linter and code formatter.
- **Pylint**: Used for static code analysis alongside Ruff to catch additional code smells and errors.