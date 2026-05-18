# Specification: Regular Sampling Mode in data_reader_mesh

## Overview
This track implements a new sampling mode called `regular` in the `data_reader_mesh`. This mode allows for systematic subsampling of data by selecting every Nth point in the mesh. This is particularly useful for reducing high-resolution data (e.g., 1/4 degree) to a lower resolution (e.g., 1 degree) for specific experiments or tests.

## Functional Requirements
1. **Configuration Support:**
   - Update the configuration schema to accept `sampling_mode: regular`.
   - Add a `sampling_step` parameter (integer N) to specify the subsampling rate.
2. **Subsampling Logic:**
   - Implement logic within `data_reader_mesh` to filter the mesh points based on the `sampling_step`.
   - Ensure the subsampling is deterministic and consistent across different data streams if they share the same mesh.
3. **Compatibility:**
   - The new mode must be compatible with existing data loading and sampling pipelines.
   - Other sampling modes (e.g., `random`, `all`) must remain unaffected.

## Non-Functional Requirements
1. **Performance:** Subsampling should be efficient and not significantly increase data loading time.
2. **Testability:** The logic must be covered by unit tests verifying different `sampling_step` values.
3. **Coverage:** Maintain >80% code coverage for the modified modules.

## Acceptance Criteria
- Setting `sampling_mode: regular` and `sampling_step: 4` results in exactly 1/4 of the points being sampled in a regular fashion.
- The implementation passes all unit tests.
- Existing features of `data_reader_mesh` continue to work as expected.
