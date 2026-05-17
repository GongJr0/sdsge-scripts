# Output Reproduction and Development Test Scripts

__People intending to reproduce the outputs from "Diagnostics and Correction of Measurement Misspecification in DSGE Models for Policy Analysis" (Kıymaç, 2026):__

  These scripts are written for execution in the development repository of `SymbolicDSGE` ([GongJr0/SymbolicDSGE](https://github.com/GongJr0/SymbolicDSGE)).
  Please follow the reproduction instructions below to reproduce the paper's outputs without having to manage dependencies.

## Reproduction Instructions

__NOTE:__ This guide assumes you have access to `pip` (therefore a Python installation) in your local environment.
Please refer to the [Python Download page](https://www.python.org/downloads/) or another tool/package manager of your choice if you do not have a Python installation available.

1. Install `uv` for automatic Python version and dependency synchronization:
   ```bash
   # It is recommended to install uv with pipx
   pip install pipx
   pipx install uv

   # However direct installations are also functional
   pip install uv
   ```

2. Clone the `SymbolicDSGE` repository with the reproduction scripts:
    ```bash
    git clone --recurse-submodules=sdsge-scripts https://github.com/GongJr0/SymbolicDSGE.git
    ```

3. Navigate to the repository and synchronize with `uv`:
   ```bash
   # Navigate to the SymbolicDSGE repository
   cd "<path-to-SymbolicDSGE>"

   # Get the dependencies and correct Python version
   uv sync --all-extras --all-groups
   ```

4. Start a Jupyter server:
   ```bash
   # Initialize a Jupyter server to run the scripts
   # You can skip this step if you want to run them inside an IDE
   uv run jupyter notebook  
   ```

5. Execute desired scripts:
   ```bash
   # Below is the relative path of reproduction scripts within the repository
   # Navigate to this path in the Jupyter server interface or in your IDE to find the scripts
   ./sdsge-scripts/misspec_tests/
   ```


## Configuration

Following the steps above, you will find multiple Jupyter Notebooks (`uv` already installed the packages necessary for Jupyter) with the naming convention `r*`.
Each of these notebooks represent a pre-configured measurement noise level and whether it is known to the reference model.
You will also find a series of configuration knobs that will allow you to produce all outputs from the paper and any other test you might be interested to run with a pre-made version of the paper's workflow.

The first cell of each notebook will have the section:
```python
_KNOWN_R = False  # True if measurement is known to the reference model
_AUGMENTED_PARAM = 'Pi_coef'  # Name of the variable to augment the equations with (can be "Pi_coef"/"x_coef"/"r_coef")
_AUGMENTED_EQUATION = 'OutGap'  # Measurement equation to augment (can be "OutGap"/"Infl"/"Rate")
_AUGMENTED_CONFIG = augmented_config_path(_AUGMENTED_EQUATION)  # Do not modify this line
_MEAS_ERR_SCALE = 0.00  # Scale (or lambda as denoted in the paper) of measurement noise
_MC_SAMPLES = 1000  # N samples for the MC repetitions
_MC_ALPHA = 0.05  # Significance level for MC rejection rates
```

Subsequent cells of all `r*` notebooks except `r0.ipynb` (used for outputs before Section 9 of the paper) are identical, and they are kept as separate files to allow running multiple configurations in parallel.

## Contact

If you have any problems reproducing the outputs, feel free to contact me by either creating an issue on this GitHub repository or on its parent [GongJr0/SymbolicDSGE](https://github.com/GongJr0/SymbolicDSGE).
Alternatively, you can reach me via email at [guneykiymac@gmail.com](mailto:guneykiymac@gmail.com).
