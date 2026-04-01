# WBHM Surrogate Modeling for Exercise Case (SBC 2025)

This repository contains a workflow to generate thermal simulation data from a Whole Body Heat Model (WBHM) in Ansys Fluent, preprocess that data, and train a PyTorch surrogate model to predict the core-body-temperature time response.

The project combines:
- **Physics-based simulation** in Fluent (journal-driven parameter sweeps + UDF behavior)
- **Data extraction/preprocessing** from generated Fluent outputs
- **Neural-network surrogate training** for faster prediction

## Repository Structure

```text
.
|-- cases/
|   `-- exercise/
|       `-- with_sweating/
|           `-- case_1/
|               |-- Exercise_Case_SteadyState.cas.h5
|               |-- Exercise_Case_SteadyState.dat.h5
|               |-- change_parameter.log
|               |-- steady_and_transient.log
|               `-- case1_parametrized.c
`-- src/
    |-- config.py
    |-- data_preprocessing/
    |   `-- data_preprocessor.py
    |-- model/
    |   `-- model.py
    |-- numerical_solution/
    |   `-- numerical_solver.py
    |-- training/
    |   `-- training.py
    `-- utils/
        |-- parameters.py
        `-- utils.py
```

## What Each Module Does

- `src/numerical_solution/numerical_solver.py`
  - Launches Fluent through `ansys.fluent.core` (PyFluent).
  - Iteratively updates Fluent journal files with sampled parameters.
  - Runs steady + transient solves and writes updated case/data files.

- `src/utils/parameters.py`
  - Samples parameter sets (metabolic rates, ambient temperature, convection coefficient) using uniform random ranges.
  - Builds a dictionary used by the numerical solver to replace placeholder values in `change_parameter.log`.

- `cases/exercise/with_sweating/case_1/case1_parametrized.c`
  - Fluent UDF that defines thermal source terms and boundary profile behavior (convection + sweating term).
  - Computes and stores internal thermal metrics (including core temperature signals).

- `src/data_preprocessing/data_preprocessor.py`
  - Reads generated `.cas.h5` and `report-def*.out` files.
  - Extracts model inputs from case parameters and output time histories from report files.
  - Saves tensors for training (for example: `input_data.pt`, `output.pt`).

- `src/model/model.py`
  - Defines a fully connected neural network (`5 -> ... -> 102`) with `Tanh` activations.

- `src/training/training.py`
  - Loads preprocessed tensors, normalizes features, and transforms outputs in the frequency domain (`rfft`).
  - Trains/evaluates the surrogate model.
  - Includes helper routines for qualitative comparison plots and metrics (`mse`, `mae`, correlation, `R^2`).

- `src/utils/utils.py`
  - Utility helpers for tensor save behavior, normalization, file sorting, and optional custom loss.

## Requirements

## 1) System/Software

- Windows environment (current setup uses Windows-style paths)
- **Ansys Fluent** installed and licensed
- Fluent-compatible Python environment (for `ansys.fluent.core`)

## 2) Python Dependencies

At minimum, install:
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`
- `wandb` (optional if tracking is enabled)
- `ansys-fluent-core` (PyFluent)

Example:

```bash
pip install torch numpy matplotlib tqdm wandb ansys-fluent-core
```

## Setup Notes (Important)

Before running scripts, verify these paths/settings:

- `src/config.py` points `CASE_DIR` to:
  - `cases/exercise/with_sweating/case1`
  - but the repository currently contains:
  - `cases/exercise/with_sweating/case_1`
  - Update one side so they match.

- `src/data_preprocessing/data_preprocessor.py` currently contains hard-coded local paths (`case_file_path`, `run_path`). Replace with your local project paths.

- `src/training/training.py` expects files like `input_data_3.pt` and `output_3.pt`. Ensure those files exist, or change the filenames to match your generated tensors.

- Import style in some scripts assumes execution from specific working directories (for example `import utils` or `from model import NeuralNetwork`). If imports fail, run scripts from the expected directory or convert imports to package-qualified style.

## End-to-End Workflow

## Step 1: Generate simulation cases in Fluent

Run:

```bash
python src/numerical_solution/numerical_solver.py
```

What it does:
- Loads base case (`Exercise_Case_SteadyState.cas.h5`)
- Updates input parameters in `change_parameter.log`
- Executes journals (`change_parameter.log`, `steady_and_transient.log`)
- Saves sequential solved cases

## Step 2: Build ML tensors from simulation outputs

Run:

```bash
python src/data_preprocessing/data_preprocessor.py
```

What it does:
- Parses generated case files and report outputs
- Builds:
  - input tensor: simulation/control parameters
  - output tensor: core temperature time trajectories
- Saves `.pt` tensors for model training

## Step 3: Train the surrogate model

Run:

```bash
python src/training/training.py
```

Typical flow in the script:
- Load tensors
- Normalize inputs/outputs
- FFT-based representation for output sequence learning
- Train fully connected neural network
- Save model (`model.pth`) when enabled

## Step 4: Evaluate and visualize

Inside `src/training/training.py`:
- Use `compare()` for qualitative curve plots
- Use `mse()` for quantitative metrics and inference time

## Data and Model Artifacts

Common generated files (names depend on current script settings):
- `input_data.pt` / `input_data_*.pt`
- `output.pt` / `output_*.pt`
- `model.pth`

Recommended practice:
- Keep generated large artifacts out of Git history.
- Use a local `results/` or `artifacts/` directory if you extend this project.

## Reproducibility Guidance

- Set a fixed random seed in `parameters.py` and training scripts if strict repeatability is needed.
- Record:
  - Fluent version
  - UDF version
  - parameter ranges and sample count (`size_data`)
  - train/test split
  - model hyperparameters (learning rate, epochs, scheduler)

## Known Limitations

- Some paths are currently machine-specific and should be generalized.
- Script coupling to folder names and file naming conventions is tight.
- Packaging/import layout is not fully standardized as an installable Python package.
- No automated tests are included yet.

## Suggested Next Improvements

- Add `requirements.txt` and pin tested package versions.
- Centralize all runtime paths into `config.py`.
- Introduce CLI arguments (for sample count, paths, epochs, output names).
- Add a small smoke test for preprocessing and model forward pass.
- Standardize Python package imports across `src/`.

## Citation / Context

If this repository supports your publication or report, cite the associated SBC 2025 work and include:
- WBHM setup assumptions
- Fluent/UDF configuration details
- surrogate model training configuration

