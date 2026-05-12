# Mantyx Surgical AI — Azure Training Pipeline

End-to-end pipeline from raw surgical data on the NAS to a registered model in Azure ML, with full traceability at every step.

## Architecture

```
NAS  →  Bronze  →  Silver  →  Gold  →  Azure ML  →  Model Registry
         (raw)   (queryable)  (training-ready)
```

| Tier | Who runs it | Script |
|---|---|---|
| NAS → Bronze | Data Engineer | manual upload / rsync |
| Bronze → Silver | Data Engineer | `pipeline/bronze_to_silver/run.sh` |
| Silver → Gold | Data Engineer or AI team | `pipeline/silver_to_gold/build_gold_dataset.py` |
| Gold → Training | AI team | `run_training.py` |
| Training → Registry | AI team | `register_model.py` |

---

## Quick Start

```bash
git clone https://github.com/Mantyx/Azure_TestCase.git
cd Azure_TestCase
uv sync
az login
```

### Step 1 — Bronze → Silver (Data Engineer, per procedure)

Flatten one procedure's raw exports into queryable Parquet files:

```bash
bash pipeline/bronze_to_silver/run.sh \
    RAPN_AZORG_Toumai_001 \
    RAPN \
    Toumai \
    AZORG
```

Outputs three Parquet files to Silver storage:
- `silver_phases/…/phases.parquet`
- `silver_masks/…/masks.parquet`
- `silver_clinicals/…/clinicals.parquet`

### Step 2 — Silver → Gold (Data Engineer or AI team, per dataset)

Build a named training dataset by running a query across Silver, extracting video frames, and rendering panoptic masks:

```bash
uv run pipeline/silver_to_gold/build_gold_dataset.py \
    --dataset-name suction_colon_mobilization_v1 \
    --procedure RAPN_AZORG_Toumai_001 \
    --procedure-type RAPN \
    --platform Toumai \
    --hospital AZORG \
    --video-key RAPN/Toumai/AZORG/RAPN_AZORG_Toumai_001/VIDEO/RAPN_AZORG_Toumai_001.mp4
```

Outputs to Gold storage `datasets/suction_colon_mobilization_v1/`:
- `frames/*.png` — extracted video frames
- `masks/*.png` — rendered panoptic masks (one per frame)
- `manifest.json` — per-frame lineage back to Bronze
- `dataset_card.json` — dataset description and selection criteria

### Step 3 — Train (AI team)

Launch an Azure ML training job on a Gold dataset:

```bash
uv run run_training.py --dataset-name suction_colon_mobilization_v1
```

Optional overrides:
```bash
uv run run_training.py \
    --dataset-name suction_colon_mobilization_v2 \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.0005
```

### Step 4 — Register model (AI team)

After a successful training job, register the model artifact in Azure ML Model Registry:

```bash
uv run register_model.py
```

---

## Project Structure

```
.
├── pipeline/
│   ├── bronze_to_silver/
│   │   ├── flatten_phases.py       # PHASES_RAW JSON → Silver parquet
│   │   ├── flatten_masks.py        # MASKS_RAW JSON  → Silver parquet
│   │   ├── flatten_clinicals.py    # clinicals.json  → Silver parquet
│   │   └── run.sh                  # Runs all three for one procedure
│   └── silver_to_gold/
│       └── build_gold_dataset.py   # Query + frame extraction + mask rendering
├── src/
│   └── train.py                    # PyTorch training script (runs inside Azure ML)
├── run_training.py                 # Launches an Azure ML training job
├── register_model.py               # Registers a trained model to Azure ML Model Registry
├── docs/
│   ├── LINEAGE.md                  # Full traceability chain — start here for audits
│   ├── DATA.md                     # Bronze / Silver / Gold data formats
│   └── AZURE_ML.md                 # Azure ML setup and concepts
└── pyproject.toml
```

---

## Traceability

Every artifact in this pipeline is traceable back to its origin. See **[docs/LINEAGE.md](docs/LINEAGE.md)** for the full chain and how to answer audit questions such as:

- Which annotator labeled a specific frame?
- What raw Bronze file produced a Silver row?
- What exact query selected the training frames?
- What code, data, and hardware produced a given model version?

---

## Documentation

- **[docs/LINEAGE.md](docs/LINEAGE.md)** — Full traceability chain (Bronze → model)
- **[docs/DATA.md](docs/DATA.md)** — Data formats and storage structure for all three tiers
- **[docs/AZURE_ML.md](docs/AZURE_ML.md)** — Azure ML setup, concepts, and training workflow

---

## Prerequisites

- Azure subscription with Azure ML workspace
- Azure CLI installed and configured (`az login`)
- [uv](https://github.com/astral-sh/uv) for Python package management
- `ffmpeg` installed locally (for the Silver → Gold step)
