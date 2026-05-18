# Tech Stack: WeatherGenerator

## 1. Core Language & Runtime
- **Python 3.12:** The primary programming language, utilizing modern features for performance and type safety.

## 2. Machine Learning Frameworks
- **PyTorch 2.6.0:** The foundational deep learning framework, configured for both CPU (testing) and NVIDIA GPU (training/inference) environments.
- **Flash-Attention:** Integrated for efficient attention mechanism computations, critical for high-resolution modeling.

## 3. Data Processing & Management
- **Zarr & NumPy:** Primary formats for large-scale, chunked multi-dimensional arrays.
- **Pandas & Polars:** Used for efficient tabular data manipulation and analysis.
- **Dask:** Employed for parallel and distributed computing across large datasets.
- **Kerchunk:** Used for unified access to diverse data formats (NetCDF, GRIB) via Zarr.
- **Anemoi-Datasets:** Specialist library for Earth system data handling.

## 4. Infrastructure & Tooling
- **UV:** Used for high-performance Python package and environment management.
- **Hatch:** The build system and project manager for the monorepo workspace.
- **OmegaConf:** Handles hierarchical configuration management.

## 5. Quality Assurance
- **Pytest:** The primary testing framework.
- **Ruff:** A fast, unified linter and formatter.
- **Pylint:** Used for additional static code analysis.
