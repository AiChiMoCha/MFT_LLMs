# Tracing Moral Foundations in LLMs

![Intro figure](assets/Intro2.png)

## Abstract
Large language models (LLMs) often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial "moral mimicry." Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed within two instruction-tuned LLMs: Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that both models represent and distinguish moral foundations in a structured, layer-dependent way that aligns with human judgments. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone.

## What This Does

This repository contains code for analyzing how LLMs represent moral foundations internally. Our approach consists of three main components:
1. Feeding the model scenarios related to different moral foundations (care, fairness, loyalty, authority, sanctity)
2. Constructing Relative Moral Representations: Extract concept vectors from LLMs by comparing activations between different moral foundations using the Persona Vector approach
3. Topological Alignment with Human-Labeled Distributions: Validate that extracted vectors align with human moral perceptions using Wasserstein distance metrics
4. Causal Intervention Through Steering: Demonstrate causal relationships between internal representations and moral outputs via activation steering

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
├── moral_steer_all_layers_robust.py    # Step 3: Causal intervention via steering
├── Data/
│   ├── MFV130Gen.json                  # Moral scenarios for training (generating vectors)
│   ├── MFV130.json                     # Moral scenarios dataset
│   ├── MFQ2.json                       # Moral Foundations Questionnaire for steering evaluation
│   └── MFRC_by_foundation/             # Test scenarios by foundation
│       ├── care.json
│       ├── fairness.json
│       ├── loyalty.json
│       ├── authority.json
│       ├── sanctity.json
│       └── nonmoral_2000.json
├── environment.yaml                    # Conda environment
├── requirements.txt                    # pip requirements
└── README.md
```

## Available Options

### Supported Models

- `llama-3.1-8b-instruct` (path: `model_cache/Meta-Llama-3.1-8B-Instruct`)
- `qwen-2.5-7b-instruct` (path: `model_cache/Qwen2.5-7B-Instruct`)

### Available Moral Foundations

Choose from: `care`, `fairness`, `loyalty`, `authority`, `sanctity`, `social_norms`

## Three-Step Workflow

### Step 1: Constructing Relative Moral Representations

Use `universal_moral_vector.py` to extract concept vectors from the model by comparing two moral foundations.

```bash
CUDA_VISIBLE_DEVICES=0 python universal_moral_vector.py \
  --model_name llama-3.1-8b-instruct \
  --model_path model_cache/Meta-Llama-3.1-8B-Instruct \
  --target_foundation sanctity \
  --control_foundation social_norms \
  --enhanced_monitoring \
  --monitoring_mode full \
  --temperature 0.01 \
  --max_new_tokens 10 \
  --n_samples 200
```

This will create a directory like:
```
MFV130/llama-3.1-8b-instruct_sanctity_vs_social_norms_enhanced_concept_vector/
└── concept_vectors/
    └── vectors_npy/  # ← Use this path in Step 2
```

### Step 2: Topological Alignment with Human-Labeled Distributions

Use `projection_test_cli_wasserstein.py` to test how well the extracted vectors separate different concepts using independent test data.

#### Option A: Between Foundation and Social Norm

Test whether the vector can distinguish a moral foundation from non-moral content:

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path model_cache/Meta-Llama-3.1-8B-Instruct \
  --vector_path MFV130/llama-3.1-8b-instruct_sanctity_vs_social_norms_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 Data/MFRC_by_foundation/sanctity.json \
  --test_file2 Data/MFRC_by_foundation/nonmoral_2000.json \
  --wass_d_min 0.3 \
  --wass_p_max 0.05
```

#### Option B: Between Foundations

Test whether the vector can distinguish between two different moral foundations:

```bash
CUDA_VISIBLE_DEVICES=0 python projection_test_cli_wasserstein.py \
  --model_path model_cache/Meta-Llama-3.1-8B-Instruct \
  --vector_path MFV130/llama-3.1-8b-instruct_care_vs_sanctity_enhanced_concept_vector/concept_vectors/vectors_npy \
  --test_file1 Data/MFRC_by_foundation/care.json \
  --test_file2 Data/twitter_by_foundation/sanctity.json \
  --wass_d_min 0.3 \
  --wass_p_max 0.05
```

**Important:** The order of test files matters!
- `test_file1`: Scenarios that should project in the **positive direction** (target concept)
- `test_file2`: Scenarios that should project in the **negative direction** (control concept)

### Step 3: Causal Intervention Through Steering

Use `moral_steer_all_layers_robust.py` to test whether the extracted vectors can causally influence model behavior on moral reasoning tasks.

```bash
CUDA_VISIBLE_DEVICES=0 python moral_steer_all_layers_robust.py \
  --model_path model_cache/Meta-Llama-3.1-8B-Instruct \
  --vector_dir MFV130 \
  --concept_pair care_vs_authority \
  --mfq_path Data/MFQ2.json \
  --output_path results/RB_care_vs_authority_all_layers.json \
  --n_rollouts 5
```

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
| `--data_file` | str | `Data/MFV130Gen.json` | JSON file with moral scenarios |

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

### Step 3: Steering Test (`moral_steer_all_layers_robust.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | (required) | Path to the model |
| `--vector_dir` | str | (required) | Directory containing extracted concept vectors |
| `--concept_pair` | str | (required) | Concept pair to test (e.g., `care_vs_authority`) |
| `--mfq_path` | str | (required) | Path to Moral Foundations Questionnaire JSON (e.g., `Data/MFQ2.json`) |
| `--output_path` | str | (required) | Path to save steering results |
| `--n_rollouts` | int | 5 | Number of rollouts for robust evaluation |

