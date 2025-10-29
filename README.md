# Moral Foundations Concept Vector Extraction

This project extracts concept vectors from LLMs to study how they represent different moral foundations. The current method is based on the Persona Vector approach, using simple mean differences to identify moral concept directions in the model's activation space.

## What This Does

The code analyzes how LLMs process moral scenarios by:
1. Feeding the model scenarios related to different moral foundations (care, fairness, loyalty, authority, sanctity, liberty)
2. Capturing the model's internal activations (how neurons fire)
3. Computing concept vectors that represent the difference between two moral foundations
4. Saving these vectors for later use in model steering or analysis

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create the environment from the provided file
conda env create -f environment.yaml

# Activate the environment
conda activate moral-expt
```

### Option 2: Using pip

```bash
# Create a virtual environment (optional but recommended)
python -m venv moral-expt-env
source moral-expt-env/bin/activate  # On Windows: moral-expt-env\Scripts\activate

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10
- PyTorch 2.5+ with CUDA 11.8+
- Transformers 4.48+
- CUDA-capable GPU (recommended)

## Project Structure

```
.
├── universal_moral_vector.py           # Step 1: Extract concept vectors
├── projection_test_cli_wasserstein.py  # Step 2: Test and validate vectors
├── MFV130Gen.json                      # Moral scenarios for training (generating Vetors)
├── unrelated_questions.json            # unrelated scenarios (test only)
├── MFV130_testP/                       # Test scenarios for Projection validation
│   ├── care.json
│   ├── fairness.json
│   ├── loyalty.json
│   └── ...
├── environment.yaml                    # Conda environment
├── requirements.txt                    # pip requirements
└── README.md
```


## Two-Step Workflow (exps start here:)

This project uses a two-step process:

### Step 1: Extract Concept Vectors

Use `universal_moral_vector.py` to extract concept vectors from the model by comparing two moral foundations.

### Step 2: Test and Validate Vectors

Use `projection_test_cli_wasserstein.py` to test how well the extracted vectors separate different concepts using independent test data.


### Step 3: Steering Test & SAEs
to be added here

-----

## Quick Start

### Step 1: Extract Concept Vectors

Extract concept vectors by comparing two moral foundations:

```bash
CUDA_VISIBLE_DEVICES=0 python universal_moral_vector.py \
  --model_name llama-3.1-8b-instruct \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --target_foundation care \
  --control_foundation loyalty \
  --n_samples 100
```

This will create a directory like:
```
MFV130/llama-3.1-8b-instruct_care_vs_loyalty_concept_vector/
└── concept_vectors/
    └── vectors_npy/  # ← Use this path in Step 2
```

### Step 2: Test the Extracted Vectors

Test how well your vectors separate concepts using independent test data:

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path /path/to/Meta-Llama-3.1-8B-Instruct \
  --vector_path MFV130/.../concept_vectors/vectors_npy \
  --test_file1 MFV130_testP/care.json \
  --test_file2 MFV130_testP/loyalty.json \
  --wass_d_min 0.3 \
  --wass_p_max 0.05
```

**Important:** The order of test files matters!
- `test_file1`: Scenarios that should project in the **positive direction** (concept1)
- `test_file2`: Scenarios that should project in the **negative direction** (concept2)

For a `care_vs_loyalty` vector:
- `test_file1` = care.json (should be on the right/positive side)
- `test_file2` = loyalty.json (should be on the left/negative side)

## Command Line Arguments

### Step 1: Vector Extraction (`universal_moral_vector.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_name` | str | `mistral-7b-instruct` | Name identifier for your model |
| `--model_path` | str | (required) | Path to the model files |
| `--target_foundation` | str | `fairness` | The moral foundation you want to study |
| `--control_foundation` | str | `social_norms` | The baseline foundation to compare against |
| `--n_samples` | int | 30 | Number of scenarios to use per foundation |
| `--monitoring_mode` | str | `comprehensive` | How many layers to monitor (`light`, `comprehensive`, `dense`, `full`) |
| `--enhanced_monitoring` | flag | off | Monitor additional components (attention, MLP, residual stream) |
| `--temperature` | float | 0.7 | Sampling temperature for generation |
| `--max_new_tokens` | int | 10 | Maximum tokens to generate per scenario |
| `--data_file` | str | `MFV130Gen.json` | JSON file with moral scenarios |

### Step 2: Vector Testing (`projection_test_cli_wasserstein.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | (required) | Path to the model (same as Step 1) |
| `--vector_path` | str | (required) | Path to `vectors_npy/` from Step 1 output |
| `--test_file1` | str | (required) | Test data for concept 1 (positive direction) |
| `--test_file2` | str | (required) | Test data for concept 2 (negative direction) |
| `--device` | str | `auto` | Device to use (`auto`, `cuda`, `cpu`) |
| `--wass_d_min` | float | 0.5 | Minimum Wasserstein distance for layer to pass |
| `--wass_p_max` | float | 0.01 | Maximum p-value for layer to pass |
| `--bh_alpha` | float | 0.05 | FDR correction alpha for multiple testing |

### Available Moral Foundations

- `care` - Care and compassion
- `fairness` - Fairness and justice
- `loyalty` - Loyalty to groups
- `authority` - Respect for authority
- `sanctity` - Purity and sanctity
- `liberty` - Freedom and autonomy
- `social_norms` - General social norms (good control baseline)
- `unrelated` - Non-moral questions (control)

## Input Data Format

The code expects a JSON file (`MFV130Gen.json`) with this structure:

```json
{
  "scenarios": [
    {
      "scenario": "A person refuses to help an injured stranger",
      "foundation": "Care",
      "wrongness_rating": 3
    },
    ...
  ]
}
```

## Output Structure

### Step 1 Output: Vector Extraction

Results are saved in `MFV130/{model_name}_{target}_vs_{control}_concept_vector/`:

```
MFV130/
└── llama-3.1-8b-instruct_care_vs_loyalty_concept_vector/
    ├── concept_vectors/
    │   ├── {model}_{target}_vs_{control}_complete.pkl    # Full results
    │   ├── {model}_{target}_vs_{control}_statistics.json # Vector stats
    │   ├── {model}_{target}_vs_{control}_report.txt      # Summary report
    │   └── vectors_npy/                                   # ← Use this in Step 2
    │       ├── *_vector.npy                               # Raw vectors
    │       └── *_normalized.npy                           # Normalized vectors
    ├── detailed_logs/
    │   └── interactions.csv                               # All model responses
    └── visualizations/
        └── *_concept_vectors.png                          # Analysis plots
```

### Step 2 Output: Vector Testing

Results are saved in `projection_results/{vector_name}/{test_comparison}_{timestamp}/`:

```
projection_results/
└── care_vs_loyalty/
    └── care_vs_loyalty_20241029_143022/
        ├── test_report.txt                    # ⭐ Main results with direction check
        ├── statistics.json                    # Detailed statistics per layer
        ├── visualizations/
        │   ├── all_layers_summary.png         # Overview of all layers
        │   └── layer_{N}_detail.png           # Per-layer analysis
        └── projections/
            ├── layer_{N}_Care.npy             # Raw projection values
            └── layer_{N}_Loyalty.npy
```

## How to Use the Results

### Loading Saved Vectors

```python
import numpy as np
import pickle

# Load a specific layer's vector
vector = np.load('MFV130/.../vectors_npy/model_layers_15_mlp_vector.npy')

# Load complete results
with open('MFV130/.../complete.pkl', 'rb') as f:
    results = pickle.load(f)
```

### Using Vectors for Model Steering

```python
# Add the concept vector to model activations
# alpha controls the strength (typically 0.5 to 2.0)
steering_strength = 1.5
modified_activation = original_activation + steering_strength * concept_vector
```

### Monitoring Concept Presence

```python
# Project activation onto concept vector to see how much of the concept is present
concept_score = np.dot(activation, concept_vector)
```

## Understanding the Method

### Step 1: Vector Extraction (Persona Vector Approach)

This implementation uses the **Persona Vector** approach:

1. **Collect activations**: Run the model on scenarios from two different moral foundations
2. **Compute means**: Calculate the average activation for each foundation
3. **Extract difference**: Subtract control mean from target mean to get the concept vector

The formula is simple:
```
concept_vector = mean(target_activations) - mean(control_activations)
```

This differs from statistical methods that use t-tests or other significance testing. The simple mean difference often works well for capturing concept directions in neural networks.

### Step 2: Vector Validation (Wasserstein Distance under Emperical Distribution)

The testing script validates vectors using:

1. **Wasserstein Distance**: Measures how far apart the two distributions are (better than Cohen's d for non-normal distributions)
2. **Direction Check**: Verifies that concept1 projects to the right (positive) and concept2 to the left (negative)
3. **Statistical Significance**: Bootstrap p-values to ensure the separation is not by chance
4. **FDR Correction**: Benjamini-Hochberg correction for testing multiple layers

#### Key Metrics

- **Wasserstein D**: Distance between distributions (higher = better separation)
- **p-value**: Probability the difference is due to chance (lower = better)
- **Direction**: ✅ if concept1 > concept2, ❌ if reversed
- **Layer Status**: 
  - `PASS`: Good separation AND correct direction
  - `FAIL(DIR)`: Good separation but wrong direction
  - `FAIL(SEP)`: Correct direction but poor separation
  - `FAIL(BOTH)`: Both metrics fail

#### Reading the Results

The test report shows:
```
层      W-D         p         p_adj    -log10(p)   方向   综合判定
15    0.845    1.0e-05    1.5e-04      5.00      ✅      PASS
18    0.923    2.0e-06    3.0e-05      5.70      ✅      PASS
22    0.654    8.0e-04    8.0e-03      3.10      ❌    FAIL(DIR)
```

- Layer 15 & 18: Good vectors (pass both tests)
- Layer 22: Wrong direction (data might be swapped or vector is inverted)

## Monitoring Modes

- **light**: 3 layers (fast, good for testing)
- **comprehensive**: 6 layers across the model depth (recommended)
- **dense**: ~10 layers (more detailed)
- **full**: all layers (thorough but slow)

With `--enhanced_monitoring`, you also capture:
- Self-attention outputs
- Attention projection layers
- MLP layers
- MLP down-projection
- Residual stream


