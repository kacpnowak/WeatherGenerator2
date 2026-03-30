# Implementation Plan: Data Reader Performance Comparison (Anemoi vs Mesh)

## Hypothesis
The choice of dataloader (Anemoi vs Mesh) makes little impact on the overall model performance when using a sufficient number of workers (e.g., 16), because the loaders can feed data to the model faster than the model can process it (forward and backward passes).

## Phase 1: Setup Profiling Infrastructure
- [x] Task: Review current training loop
    - [x] Identify injection points for timing data loading, forward pass, and backward pass in `src/weathergen/train/trainer.py`. (7 characters of commit hash will be added later)
- [x] Task: Implement timing mechanism
    - [x] Write unit tests for timing utility functions (Red Phase).
    - [x] Implement the timing functionality (Green Phase).
    - [x] Ensure minimal overhead and toggleability.
- [x] Task: Integrate timing into the training loop
    - [x] Add configuration flag `profiling.enabled` in `config/default_config.yml`.
    - [x] Instrument data loading, forward pass, and backward pass with the timing mechanism.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Setup Profiling Infrastructure' (Protocol in workflow.md)

## Phase 2: Execute Performance Comparison and Reporting
- [x] Task: Configure test runs for worker scaling
    - [x] Set up configurations for the **Anemoi** reader using `config/test_anemoi_base.yml`.
    - [x] Set up configurations for the **Mesh** reader using existing mesh stream configs.
- [x] Task: Run profiling experiments with 16 workers
    - [x] Execute model with **Anemoi** reader (16 workers) and collect metrics.
    - [x] Execute model with **Mesh** reader (16 workers) and collect metrics.
- [x] Task: Run baseline experiments (e.g., 1 or 4 workers)
    - [x] Execute model with **Anemoi** reader and collect metrics.
    - [x] Execute model with **Mesh** reader and collect metrics.
- [x] Task: Analyze results and prove/refute hypothesis
    - [x] Compile timing metrics and compare "Wait for Data" vs. "Model Processing Time".
    - [x] Document findings, specifically focusing on the performance gap between loaders as worker count increases.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Execute Performance Comparison and Reporting' (Protocol in workflow.md)
