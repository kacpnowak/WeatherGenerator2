# Product Guidelines: WeatherGenerator

## 1. Documentation Style
- **Scientific / Academic**: Maintain a formal tone throughout the documentation. Include detailed equations, algorithmic descriptions, and proper citations or references to underlying research papers to ensure scientific rigor.

## 2. Architectural Principles
- **Modular & Pluggable**: Design system components (models, datasets, readers, evaluation metrics) to be easily swappable and extensible.
- **High-Performance**: Prioritize computational efficiency, optimal GPU utilization, and scalable distributed training architectures.
- **Reproducible**: Enforce strict logging, fixed random seeds, and deterministic operations wherever possible to guarantee that scientific experiments can be perfectly replicated.
- **Keep It Simple, Stupid (KISS)**: Avoid unnecessary complexity. Strive for straightforward, understandable code and architectures over overly clever or intricate solutions.

## 3. Error Handling and Validation
- **Strict / Fail-Fast**: The system should crash early and loudly upon encountering misconfigurations, invalid data, or unexpected states. This approach prevents silent failures and ensures the correctness of scientific outputs.
