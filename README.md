# Image–Caption Mismatch as a Propaganda Proxy in Russian-Language War Telegram Channels

Code for my MSc Artificial Intelligence dissertation at the University of Edinburgh (2026).

## Research question

Can cross-modal semantic inconsistency between an image and its caption, a signal established for out-of-context misinformation, serve as a proxy for propaganda in Russian-language war Telegram channels?

## Findings

- **Post level:** the M-CLIP cosine mismatch score does not distinguish propagandistic from neutral posts (AUC 0.542). The two classes are congruent in their cross-modal alignment.
- **The signal is there, but in the text.** A supervised probe on the same M-CLIP embeddings recovers clear signal (text-only AUC 0.782, p = 0.001 under a 1000-permutation test). Adding the image does not improve on text alone.
- **Independent check:** a zero-shot LLM baseline reading only the translated caption agrees with the gold labels at Cohen's κ = 0.523 (accuracy 0.794), which corroborates that the signal is carried by the text.
- **Channel level:** the mismatch proxy is again uninformative. Cross-channel image reuse appears as a coordination signal, and channels vary along a continuous gradient of scale and activity rather than forming discrete types.

The contribution is a characterisation of where the propaganda signal resides (linguistic at the post level, behavioural at the channel level) and a demonstration that cross-modal alignment was the wrong quantity to measure. Mismatch is treated throughout as a proxy under investigation, never as equivalent to propaganda.

| Feature set | Dimensions | Mean CV AUC |
|---|---|---|
| Cosine mismatch | 1 | 0.542 |
| Image embedding | 512 | 0.640 |
| Concatenation (early fusion) | 1024 | 0.742 |
| Late fusion | 2×512 | 0.770 |
| Text embedding | 512 | **0.782** |

## Data

A corpus of 9,054 image–caption pairs from 92 public Russian-language war Telegram channels, collected with the Telegram API via Telethon.

**The dataset is not included in this repository**, for size and privacy reasons. Gold-set labels are also not included.

## Annotation

The gold set is a 400-post sample, stratified across channels (one to ten posts per channel) with a fixed random seed. Each post, image and caption together, received one of three labels: **propagandistic**, **neutral** or **unclear**, using a holistic, manner-not-topic definition of propaganda. Annotation was done on the English translations. After filtering and removing unclear posts, the evaluable set is 345 posts (115 propagandistic, 230 neutral).

The scheme was refined through piloting with a second annotator, who was used only to measure reliability:

| Round | n | Cohen's κ |
|---|---|---|
| Pilot 1 | 30 | 0.276 |
| Pilot 2 (refined scheme) | 30 | 0.874 |
| Held-out check | 30 | 0.935 |
| Full overlap, three-class | 100 | **0.658** |
| Full overlap, binary | 100 | 0.699 |

## Methods

- **M-CLIP** (multilingual CLIP) for image and caption embeddings and the cosine mismatch score
- **NLLB-200 (1.3B)**, beam search, for Russian → English translation, with a deterministic quality filter applied identically to the gold set and the full corpus
- **Supervised probes** on M-CLIP embeddings (text, image, early and late fusion) with nested stratified five-fold cross-validation and label-permutation tests
- **Mistral-7B-Instruct-v0.3**, zero-shot, 4-bit quantised, greedy decoding, as a text-only LLM baseline
- **Perceptual hashing** for cross-channel image-reuse detection
- **Spearman correlation** (Benjamini–Hochberg FDR correction), **HDBSCAN** and **k-means** for channel-level analysis

## Pipeline overview

| Stage | What it does | Main scripts |
|---|---|---|
| 1. Collection | Scrape posts and media from Telegram channels | `script_posts.py`, `script_media.py`, `script_media_new.py`, `rescrape_posts_json.py`, `merge_rescrape.py` |
| 2. Corpus building & cleaning | Build the image–caption corpus, audit it, remove junk | `build_corpus_csv.py`, `build_channel_table.py`, `audit_dataset.py`, `clean_source_junk.py`, `collect_corpus_images.py`, `sampling.py` |
| 3. Translation | NLLB-200 translation and quality filtering | `translate_captions_gpu.py`, `translate_captions_cpu.py`, `translation_quality_filter.py`, `apply_translation_quality.py`, `scan_translation_collapse.py`, `diff_translations.py` |
| 4. Mismatch scoring | M-CLIP embeddings and mismatch scores | `mclip_embed.py`, `mclip_score.py`, `evaluate_mismatch.py`, `diagnose_pairs.py` |
| 5. Image reuse | Perceptual hashing across channels | `image_reuse_hash.py`, `channel_reuse_rate.py` |
| 6. Annotation | Pilot, held-out and overlap draws, annotation tool, agreement | `make_pilot.py`, `build_pilot_bundle.py`, `build_bundle.py`, `build_bundle_p2.py`, `draw_heldout.py`, `draw_overlap.py`, `annotate_server.py`, `compute_kappa.py`, `review_disagreements.py` |
| 7. Analysis | Probes, LLM baseline, clustering, correlations | `embedding_probe.py`, `fusion_probe.py`, `probe_predictions.py`, `probe_agreement.py`, `llm_baseline_infer.py`, `hdbscan_clustering.py`, `kmeans_analysis.py`, `spearman_analysis.py`, `consolidate_results.py` |
| 8. Figures | Dissertation figures | `fig_*.py`, `make_probe_figures.py`, `collect_sample_images.py` |

`mclip.slurm` and `translate.slurm` are job scripts for running the GPU steps on a Slurm cluster.

## Setup

```bash
pip install -r requirements.txt
```

`multilingual-clip` is incompatible with `transformers` 5.x, so the M-CLIP scripts need `transformers` 4.x.

The collection scripts need your own Telegram API credentials (from [my.telegram.org](https://my.telegram.org)), set as environment variables:

```powershell
$env:TG_API_ID = "your_api_id"
$env:TG_API_HASH = "your_api_hash"
```

## Author

Kiran Balamurugan, MSc Artificial Intelligence, University of Edinburgh
