# Image–Caption Mismatch as a Propaganda Proxy in Russian-Language Telegram

Code for my MSc Artificial Intelligence dissertation at the University of Edinburgh (2026).

## Research question

Is cross-modal (image–caption) mismatch a viable proxy for propaganda in Russian-language Telegram war media?

## Finding

**No, not at post level in this corpus.** Image–caption mismatch did not work as a reliable post-level signal for propaganda. In these channels, war propaganda appears to work through *emotionally coherent* framing: the image and the caption support each other rather than contradict each other. That is the opposite of out-of-context news misinformation, where M-CLIP-style mismatch signals are effective. The project is best read as a boundary condition on when mismatch-based detection applies.

Throughout, mismatch is treated as a **proxy under investigation**, never as equivalent to propaganda.

## Pipeline overview

| Stage | What it does | Main scripts |
|---|---|---|
| 1. Collection | Scrape posts and media from ~100 public Telegram channels | `script_posts.py`, `script_media.py`, `script_media_new.py`, `rescrape_posts_json.py`, `merge_rescrape.py` |
| 2. Corpus building & cleaning | Build the post/image corpus, audit it, remove junk | `build_corpus_csv.py`, `build_channel_table.py`, `audit_dataset.py`, `clean_source_junk.py`, `collect_corpus_images.py`, `sampling.py` |
| 3. Translation | Russian → English caption translation with NLLB-200, plus quality filtering | `translate_captions_gpu.py`, `translate_captions_cpu.py`, `translation_quality_filter.py`, `apply_translation_quality.py`, `scan_translation_collapse.py`, `diff_translations.py` |
| 4. Mismatch scoring | M-CLIP image–caption embeddings and mismatch scores | `mclip_embed.py`, `mclip_score.py`, `evaluate_mismatch.py`, `diagnose_pairs.py` |
| 5. Image reuse | Perceptual hashing to detect images reused across channels | `image_reuse_hash.py`, `channel_reuse_rate.py` |
| 6. Annotation | Gold-set sampling, annotation tool, inter-annotator agreement | `make_pilot.py`, `build_pilot_bundle.py`, `build_bundle.py`, `build_bundle_p2.py`, `draw_heldout.py`, `draw_overlap.py`, `annotate_server.py`, `compute_kappa.py`, `review_disagreements.py` |
| 7. Analysis | Clustering, statistical tests, probes, LLM baseline | `hdbscan_clustering.py`, `kmeans_analysis.py`, `spearman_analysis.py`, `embedding_probe.py`, `fusion_probe.py`, `probe_predictions.py`, `probe_agreement.py`, `llm_baseline_infer.py`, `consolidate_results.py` |
| 8. Figures | Figures used in the dissertation | `fig_*.py`, `make_probe_figures.py`, `collect_sample_images.py` |

`mclip.slurm` and `translate.slurm` are job scripts for running the GPU steps on a Slurm cluster.

## Methods

- **M-CLIP** (multilingual CLIP) for image–caption similarity / mismatch scoring
- **NLLB-200 (1.3B)** for Russian → English translation
- **Perceptual hashing** for cross-channel image-reuse detection
- **HDBSCAN** and k-means clustering
- **Mann-Whitney U** tests and **Spearman ρ** correlation
- Linear / fusion **probes** on embeddings, and an **LLM baseline**

## Annotation

A gold set of ~400 posts was annotated into four categories (`frontline_reporting`, `aid_support_mobilization`, `propaganda`, and rejection categories), including a ~200-post set double-annotated by a second annotator to measure inter-annotator agreement (Cohen's κ, `compute_kappa.py`). Annotation is part of the measurement setup, not a contribution in its own right.

## Data

**The dataset is not included in this repository**, for size and privacy reasons. It was collected from public Russian-language Telegram channels using the Telegram API via Telethon (see stage 1). Annotations are also not included.

## Setup

```bash
pip install -r requirements.txt
```

The collection scripts need your own Telegram API credentials (from [my.telegram.org](https://my.telegram.org)), set as environment variables:

```powershell
$env:TG_API_ID = "your_api_id"
$env:TG_API_HASH = "your_api_hash"
```

## Author

Kiran Balamurugan, MSc Artificial Intelligence, University of Edinburgh
