# Specification: Data Robustness - NaN and Filler Value Filtering

## Overview
This track addresses data quality issues in `data_reader_mesh.py` by implementing a filtering mechanism. The goal is to ensure that invalid data points (NaNs or configurable filler values) are identified and removed along with their corresponding coordinates before being passed to the model.

## Functional Requirements
- **NaN Detection**: Automatically identify and drop any data point containing a `NaN` value.
- **Configurable Filler Values**: Extend the configuration to allow users to specify a list of "filler" or sentinel values (e.g., `-9999`) that should also be dropped.
- **Coordinate Alignment**: When a data point is dropped, its associated coordinates in the mesh must also be removed to maintain spatial alignment.
- **Empty Sample Handling**: Gracefully handle scenarios where all points in a sample are invalid, allowing the reader to return an empty sample or skip it according to the pipeline's requirements.

## Acceptance Criteria
- **Strict Filtering**: A sample containing a mix of valid and invalid points is returned with *only* the valid points and their respective coordinates.
- **Configuration Flexibility**: Users can successfully define custom filler values in the YAML configuration.
- **Deterministic Behavior**: The filtering process is deterministic and does not introduce spatial offsets.
- **Robustness**: The data pipeline does not crash when encountering entirely invalid samples.

## Out of Scope
- Complex imputation strategies (e.g., k-NN interpolation).
- Temporal filtering across multiple time steps (focus is on spatial points within a single load).
