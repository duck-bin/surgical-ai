# Surgical CVS AI — Automated Critical View of Safety Assessment

Automated assessment of the **Critical View of Safety (CVS)** in laparoscopic
cholecystectomy: pixel-level anatomical segmentation → Strasberg 3-criteria
classification → a frame-level CVS achievement score (0–3).

> **Research prototype — not for clinical use.**

This work replicates and extends the DeepCVS / LG-CVS / Endoscapes2023
benchmark line of research.

---

## 1. Motivation

Laparoscopic cholecystectomy is one of the most common abdominal operations.
**Bile duct injury (BDI)** is its most feared complication; the majority of
BDIs stem from misidentification of anatomy. The **Critical View of Safety**,
introduced by Strasberg, is the surgical safety standard intended to prevent
this misidentification.

- BDI incidence in laparoscopic cholecystectomy: `[CITATION NEEDED]`
- Proportion of BDIs attributable to misidentification: `[CITATION NEEDED]`

Automating CVS assessment offers a path to intraoperative decision support and
objective surgical-quality measurement.

## 2. Method

```mermaid
flowchart LR
    A[Laparoscopic frame] --> B[Anatomical segmentation]
    B -->|SAM2 + LoRA / U-Net baseline| C[6-class mask:<br/>background, liver, gallbladder,<br/>cystic_duct, cystic_artery, tool]
    A --> D
    C --> D[CVS classifier<br/>ViT-Small, 9-channel input]
    D --> E[3 binary Strasberg criteria]
    E --> F[CVS score 0–3]
```

**Segmentation.** Primary model: `facebook/sam2-hiera-base-plus`, loaded via
`transformers.Sam2Model`. LoRA adapters (rank 8) are applied to the Hiera image
encoder's attention; the small mask decoder is fully fine-tuned and repurposed
to emit a prompt-free dense 6-class logit map (`num_multimask_outputs = 6`, no
point/box prompts). Baseline: EfficientNet-B4 U-Net.

**Temporal variant** (`model=sam2_temporal`). Adds a lightweight ConvGRU head
over a window of `T = 3` consecutive frames' native mask-decoder outputs and a
zero-init residual: at initialisation it reproduces the frame-level model
exactly, and training only learns the temporal correction. Output stays
`(B, num_classes, H, W)` for the target (last) frame — a drop-in for the shared
`SegmentationModule`. Aimed at thin, flickering structures (`cystic_duct`); see
`src/models/temporal.py` for the rationale on why the fusion is at decoder
outputs rather than encoder embeddings.

**CVS classification.** ViT-Small backbone consuming a 9-channel input
(6-channel segmentation mask + RGB frame), with three independent binary heads
for Strasberg's criteria.

Training details: PyTorch Lightning + Hydra, mixed precision (bf16, with
automatic fp16 fallback on non-Ampere GPUs), AdamW with separate
encoder/decoder learning rates, cosine schedule with 5-epoch warmup, full seed
control and deterministic algorithms. See `configs/` for exact hyperparameters.

## 3. Results

Numbers are produced by actual experiments — do **not** edit cells to
non-`TBD` values until the corresponding run has completed.

### CholecSeg8k segmentation (test split, 1834 frames)

| Method | mIoU | Cystic Duct Dice | CVS mAP |
|---|---|---|---|
| U-Net (ours) | TBD | TBD | TBD |
| SAM2 + LoRA (ours) | TBD | TBD | TBD |
| **SAM2 + LoRA + temporal (ours)** | **0.703 (0.696–0.709)** | **0.000 (0.000–0.000)** | TBD |
| SAM2 zero-shot | TBD | TBD | TBD |
| DeepCVS (Mascagni 2022, reported) | — | — | 71.9 |
| LG-CVS (Murali 2023, reported) | — | — | 80.6 |

Per-class Dice (temporal model, only model trained so far): background 0.945,
liver 0.849, gallbladder 0.606, **cystic_duct 0.000**, tool 0.805. The
`cystic_artery` class has no CholecSeg8k labels; it will be learned from
Endoscapes2023.

**Why `cystic_duct = 0`.** Selection metric was `val_cystic_duct_dice`
(monitor, mode=max, patience=10). On a 0.031%-prevalence class the model needs
30+ epochs before that metric leaves zero, but early stopping interpreted
"10 epochs at 0" as no improvement and cut training at epoch 11/99. Frame-level
visualization (notebook 07) shows the trained model already learned the large
classes (liver/gallbladder/tool match GT well); the duct simply did not get
enough time. Open issue, to be addressed in the next training run.

Qualitative examples: see `notebooks/07_results_visualization.ipynb` (loads
the trained checkpoints from HuggingFace and renders
`[input | GT | <each model>]` side by side).

### Trained checkpoints (HuggingFace)

The first full run's checkpoints, benchmark table, train log and run notes are
stored at
**[`duckbin/surgical-sam2-temporal`](https://huggingface.co/duckbin/surgical-sam2-temporal)**
(private). Download with `hf download duckbin/surgical-sam2-temporal
sam2_temporal_results.zip --repo-type=model && unzip
sam2_temporal_results.zip` to restore `outputs/sam2_temporal/best.ckpt` and
`results/benchmark_table.md`.

## 4. Reproducing

**On Google Colab** — open `notebooks/run_pipeline.ipynb` and run it top
to bottom. It is idempotent (safe to re-run after a disconnect) and resumes
interrupted training from the last checkpoint. The manual steps below are the
equivalent.

```bash
# 1. Environment (Python >= 3.11; Colab / RunPod GPU runtime recommended)
pip install -r requirements.txt
# torch is preinstalled on Colab/RunPod; SAM2 loads via transformers (Sam2Model)
# — no separate install.

# 2. Data
bash scripts/download_cholecseg8k.sh
# Endoscapes2023 requires PhysioNet credentialed access — download manually
# to ./data/endoscapes2023/, then:
bash scripts/prepare_endoscapes.sh

# 3. Train: SAM2 + LoRA segmentation, then the CVS classifier.
#    Checkpoints are written to outputs/<model>/best.ckpt; train_cvs and the
#    benchmark runner read them automatically.
python -m src.train.train_segmentation model=sam2_lora   # or model=unet_baseline
# Temporal variant: a ConvGRU head over a window of T=3 consecutive frames,
# targeting cystic_duct recall + frame-to-frame consistency (drop-in output).
python -m src.train.train_segmentation model=sam2_temporal
python -m src.train.train_cvs

# 4. Benchmark + visualization + demo
python -m src.eval.benchmark_runner    # -> results/benchmark_table.md
# Visual comparison of the three trained models (input | GT | U-Net | SAM2 | temporal):
#   open notebooks/07_results_visualization.ipynb  (reads outputs/*/best.ckpt)
python -m app.gradio_demo              # interactive CVS assessment demo
```

### Running on RunPod (A100, recommended for the full run)

The repo defaults are wired for a single 24 GB A100 (or 16 GB T4 with
`low_memory=true`). To reproduce on a RunPod A100 pod end-to-end:

```bash
# 1. Create the pod
#    Template: "PyTorch 2.x" (CUDA 12.x)  GPU: A100 (80 GB or 40 GB both fine)
#    Volume:   60 GB+ (CholecSeg8k ~3 GB + 3 checkpoints ~3-6 GB + scratch)
#    Open the pod's Jupyter / Web Terminal.

# 2. Clone + install (one-off, ~3 min)
git clone https://github.com/duck-bin/surgical-ai.git && cd surgical-ai
pip install -r requirements.txt

# 3. Data (~3 GB, ~20-40 min on first run; cached afterwards)
bash scripts/download_cholecseg8k.sh

# 4. Train the three segmentation models (~6-8 h each on A100;
#    low_memory=false lifts the per-device batch from 1 to 4)
python -m src.train.train_segmentation model=unet_baseline \
       low_memory=false num_workers=4
python -m src.train.train_segmentation model=sam2_lora      \
       low_memory=false num_workers=4
python -m src.train.train_segmentation model=sam2_temporal  \
       low_memory=false num_workers=4

# 5. Comparison table + visualization
python -m src.eval.benchmark_runner    # -> results/benchmark_table.md
#    then open notebooks/07_results_visualization.ipynb in Jupyter

# 6. (Optional) Stream training curves to wandb live (default mode=disabled)
python -m src.train.train_segmentation model=sam2_temporal wandb.mode=online
```

**Tips**

- Pod disconnects are safe — every run picks up the last checkpoint
  automatically. Just re-run the same `train_segmentation` command.
- If a model is too slow, drop to T4-style: `low_memory=true num_workers=2`.
- To visualise on a different machine (e.g. Colab) instead of the pod, just
  copy the `outputs/` folder over — notebook 07 reads from `outputs/<model>/best.ckpt`.

### Live training curves with Weights & Biases (optional)

Training logs `loss / val_miou / val_<class>_dice / val_cystic_duct_dice` every
epoch. By default these go to local files only (`wandb.mode=disabled`). To
stream them to the [wandb](https://wandb.ai) web UI so you can watch the run
from anywhere (phone, another laptop) without staying connected to the pod:

```bash
# Once per machine: install + paste your wandb API key (free signup, ~30 s)
pip install wandb && wandb login

# Add wandb.mode=online to any training command:
python -m src.train.train_segmentation model=sam2_temporal wandb.mode=online
```

This is **highly recommended for long runs on RunPod** — you can see whether
`val_cystic_duct_dice` is actually climbing without paying for the pod just to
watch a terminal. `wandb.project=surgical-cvs-ai` is preset; override it with
`wandb.project=<name>` if you want a different workspace.

The configs default to a 16 GB T4 (`low_memory: true` — per-device batch 1 with
16x gradient accumulation); set `low_memory=false` on a larger GPU. Expected
runtime and cost (RunPod A100, see Step-1 plan for details):

| Stage | Runtime (A100) | Approx. cost |
|---|---|---|
| CholecSeg8k segmentation training | ~6–8 h | ~$10–15 |
| Endoscapes CVS classifier | ~3–4 h | ~$5–8 |
| **Full reproduction** | — | **< $50** |

### Implementation progress (state of the repo)

What is wired and verified end-to-end:

- **Segmentation training** for `unet_baseline`, `sam2_lora`, `sam2_temporal`
  (Lightning + Hydra, bf16 mixed precision, AdamW + cosine + 5-epoch warmup,
  resume from `outputs/<model>/last.ckpt`).
- **Class-balance pipeline** with inverse-sqrt-frequency loss weights and a
  WeightedRandomSampler over the train split. The expensive per-frame pass is
  computed once and cached to
  `<data.cache_dir>/class_stats_seed<seed>.npz`, so re-runs and subsequent
  models start in seconds instead of re-decoding the full train set.
- **Video-level split + sliding windows.** `CholecSeg8kWindowDataset` builds
  contiguous T-frame windows grouped by video, never crossing a video or
  train/val/test boundary; replay-consistent augmentation across the clip via
  `albumentations.ReplayCompose`.
- **Per-class metrics + checkpoint selection.** `val_<class>_dice` is logged
  every epoch (NaN-ignoring aggregation so rare classes don't poison the
  monitor), with `val_cystic_duct_dice` as the default selection metric.
- **wandb logger** is wired through `wandb.mode=online/offline/disabled`; the
  run is named after the model so the three models can be overlaid in one
  workspace.
- **Benchmark + visualization.** `benchmark_runner` evaluates each available
  checkpoint on the CholecSeg8k test split (1834 frames or 1834 windows for the
  temporal model) and writes `results/benchmark_table.md` with 95% bootstrap
  CIs. `notebooks/07_results_visualization.ipynb` pulls checkpoints from
  HuggingFace and renders `[input | GT | <each available model>]` side by side
  — it auto-detects which checkpoints are present, so it works with one model
  today and grows with the others.
- **Smoke tests** (CPU CI + a tiny notebook 06 run) keep the temporal path,
  Lightning module, and window dataset honest end-to-end.

What is **not** done yet (so the results table is what it is):

- Frame-level baselines (`unet_baseline`, `sam2_lora`) at full schedule —
  needed for a fair comparison.
- CVS classifier training (Endoscapes2023 — manual PhysioNet download required).
- Lifting `cystic_duct` Dice off zero. The selection-metric / early-stop
  interaction needs to be addressed first; see Results and Limitations.

## 5. Limitations

- Single segmentation dataset (CholecSeg8k) for pretraining.
- Temporal modeling (`sam2_temporal`) is implemented and trained; the
  frame-level baselines (`unet_baseline`, `sam2_lora`) have not yet been
  trained to completion, so a fair frame-vs-temporal comparison is still
  pending.
- Ground truth is the public dataset annotations; not independently
  surgeon-validated by the author. Frame-level inspection of the trained model
  (notebook 07) suggests `cystic_duct` labels in CholecSeg8k are sparse and
  inconsistent — some frames show a clearly visible duct in the input that is
  not labelled in the ground-truth mask.
- `cystic_artery` is not labeled in CholecSeg8k; it is learned only from
  Endoscapes2023 (see Clinical Note and `src/data/cholecseg8k.py`).
- The first full `sam2_temporal` run early-stopped at epoch 11 because the
  selection metric (`val_cystic_duct_dice`) was 0 for that whole window — see
  Results above. The other anatomical classes were learnt well; the duct
  itself did not get enough training time.

## 6. Clinical Note

The **Critical View of Safety** is a method of target identification in
laparoscopic cholecystectomy. Before any structure is clipped or divided, the
surgeon establishes that the structures entering the gallbladder have been
unambiguously identified. Strasberg's three criteria are:

- **C1 — Two structures.** Two and only two tubular structures are seen
  entering the gallbladder (the cystic duct and the cystic artery).
- **C2 — Hepatocystic triangle cleared.** The hepatocystic triangle is cleared
  of fat and fibrous/connective tissue.
- **C3 — Cystic plate exposed.** The lower one-third of the cystic plate (the
  gallbladder bed on the liver) is exposed.

The CVS score in this project is the sum of the three satisfied criteria
(0–3), matching the Strasberg formulation. Achieving CVS does not require an
intraoperative cholangiogram; it is a visual, anatomy-based safety checkpoint
whose purpose is to prevent the cystic duct / common bile duct
misidentification that underlies most bile duct injuries.

## 7. Citations

<!-- TODO (Step 9): verify every BibTeX entry (authors, venue, year, IDs)
     against the primary source before publication. Entries below are
     placeholders and must not be cited as-is. -->

Datasets and key papers used in this project:

- CholecSeg8k — Hong et al., *CholecSeg8k: A Semantic Segmentation Dataset for
  Laparoscopic Cholecystectomy Based on Cholec80*. `% TODO: verify`
- Endoscapes2023 — Mascagni et al., *Scientific Data*, 2024. `% TODO: verify`
- SAM 2 — Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024.
  `% TODO: verify`
- SurgiSAM2 — Kamtam et al., arXiv:2503.03942, 2025. `% TODO: verify`
- DeepCVS — Mascagni et al., 2022. `% TODO: verify`
- LG-CVS — Murali et al., 2023. `% TODO: verify`
- Strasberg & Brunt — *Rationale and use of the critical view of safety in
  laparoscopic cholecystectomy*. `% TODO: verify`

## License

MIT — see [LICENSE](LICENSE).
