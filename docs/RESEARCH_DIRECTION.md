# Research direction — building the end-to-end CVS pipeline

This note fixes the research direction for the project and the concrete order of
work. The goal is a **single, working end-to-end system**:

```
laparoscopic frame
  → 6-class anatomical segmentation   (background, liver, gallbladder,
                                        cystic_duct, cystic_artery, tool)
  → CVS classifier (9-channel: 6 mask + 3 RGB)
  → 3 binary Strasberg criteria
  → CVS score 0–3
```

The individual stages are implemented and tested (`src/inference/pipeline.py`
chains them; `app/gradio_demo.py` serves them). The blocker to a *usable*
end-to-end system is not any single stage — it is that **no single segmentation
model produces all six classes**, and the CVS classifier needs all six.

## 1. The core problem: complementary, partially-labelled datasets

The two segmentation datasets are complementary and each only annotates a
*subset* of the six shared classes:

| Class          | CholecSeg8k | Endoscapes-Seg |
|----------------|:-----------:|:--------------:|
| background     | ✅ | ✅ |
| liver          | ✅ | ❌ (folded into background) |
| gallbladder    | ✅ | ✅ |
| cystic_duct    | ❌ (≤3 px in the whole train split) | ✅ |
| cystic_artery  | ❌ (no label at all) | ✅ |
| tool           | ✅ | ✅ |

So a model trained on **CholecSeg8k alone** never learns the cystic
duct/artery — and those are the *most* CVS-critical structures (criterion C1 is
literally "two tubular structures = cystic duct + cystic artery"). A model
trained on **Endoscapes-Seg alone** never learns liver, losing the anatomical
context for criterion C3 (cystic plate on the liver bed). Feeding either into
the CVS classifier hands it mask channels that are structurally blank for the
classes that matter.

The naïve fix — concatenate the two datasets and train with ordinary
cross-entropy — is **actively harmful**. The two datasets disagree on what
"background" means: an Endoscapes frame labels liver pixels as background, so
full CE pushes the model to *suppress* liver exactly where CholecSeg8k teaches
it; a CholecSeg8k frame labels any duct/artery pixels as background, undoing
what Endoscapes teaches. They fight over the shared background channel.

## 2. The direction: joint training with partial-label (marginal) supervision

Train **one** 6-class model on the **union** of both datasets, and supervise
each sample only on the classes its source actually annotates. Every class a
source does *not* annotate is merged into that source's background probability,
so a background-labelled pixel is satisfied by predicting background **or** any
unlabelled class. The model is then free to predict liver on an Endoscapes frame
(folded into that frame's background) and the duct on a CholecSeg8k frame,
without penalty.

This is the established *marginal loss* for combining heterogeneously-labelled
medical segmentation datasets (Shi et al., *Marginal loss and exclusion loss for
partially supervised multi-organ segmentation*, Medical Image Analysis 2021; the
same principle underlies DoDNet-style partial supervision). By construction it
reduces **exactly** to the ordinary focal + Dice/Tversky loss when every class
is labelled, so nothing about single-dataset training changes.

**Implemented in this repo:**

- `src/losses/partial_label.py` — `PartialLabelSegLoss` (marginal focal +
  marginal Dice/Tversky), taking a per-sample `labeled` class mask.
- `src/data/joint_seg.py` — `LabeledClassDataset` (tags each item with its
  source's labelled-class mask) and `JointSegDataset` (unions the sources while
  preserving the `load_mask` / `_load_raw` fast paths).
- `configs/data/joint_seg.yaml` + the `joint_seg` branch in
  `src/train/train_segmentation.py`. Train on the union; validate/test on
  Endoscapes-Seg (it labels the CVS-critical duct/artery, so the
  `val_cystic_duct_dice` selection metric stays honest).

Run it:

```bash
bash scripts/download_cholecseg8k.sh
bash scripts/download_endoscapes.sh
python -m src.train.train_segmentation \
       data=joint_seg model=sam2_lora \
       low_memory=false num_workers=4 copy_paste.enabled=true wandb.mode=online
```

The rare-class levers already in the repo stack on top of this: inverse-sqrt
loss weights, the weighted sampler, Focal-Tversky, and rare-class copy-paste
(harvested/pasted only for the classes each source labels, so pasted pixels stay
inside the sample's labelled set).

## 3. Order of work

1. **[done] Joint partially-labelled segmentation.** One model, all six classes.
   Code + CPU tests landed; the confirming GPU run is pending.
2. **Train the joint segmentation model** (`data=joint_seg model=sam2_lora`) and
   confirm `cystic_duct` / `cystic_artery` Dice leave zero on the Endoscapes
   test split while liver/gallbladder/tool stay strong on CholecSeg8k.
3. **Point the CVS classifier at the joint checkpoint.** Set
   `segmentation.checkpoint`/`model_config` in `configs/cvs_classifier.yaml` to
   the joint run's `outputs/sam2_lora/best.ckpt`, then train
   `src/train/train_cvs.py` on Endoscapes2023 CVS labels.
4. **Benchmark end-to-end.** `benchmark_runner` for CholecSeg8k mIoU;
   Endoscapes test for duct/artery Dice; CVS mAP + QWK from the classifier.
5. **Frame-level baselines** (`unet_baseline`, `sam2_lora` single-dataset) at
   full schedule for a fair comparison, plus the temporal variant.

## 4. Open questions / risks

- **Domain gap.** CholecSeg8k and Endoscapes come from different scopes/centres;
  joint training assumes the shared classes transfer. Worth a quick check that
  liver learned on CholecSeg8k is not degraded on Endoscapes frames (and vice
  versa) — the marginal loss prevents active suppression but cannot force
  cross-domain generalisation.
- **Validation coverage.** Val/test run on Endoscapes only, so liver quality is
  not tracked during training; it is checked post-hoc via the CholecSeg8k
  benchmark. A joint val set with marginal-aware metrics would close this.
- **Label-map verification.** The Endoscapes semseg id→class table is from
  CAMMA docs, not the bytes on disk; `semseg_id_histogram` / notebook 09 must
  confirm duct=4 / artery=3 before the run (the CholecSeg8k colour-map lesson).
