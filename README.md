# SciFit-Net

**Explainable journal recommendation from a title and abstract, with scope-mismatch estimation and title–abstract consistency checking.**

SciFit-Net takes a manuscript's title and abstract and returns:

1. a **title–abstract consistency** score,
2. a ranked list of **recommended journals** with compatibility scores,
3. a **scope-compatibility / scope-mismatch risk** assessment for the best venue,
4. **explanations**: key concepts and similar papers with DOIs.

---

## Example output (real model output, test paper)

```
╔══════════════════════════════════════════════════╗
║                SCIFIT-NET RESULT                 ║
╠══════════════════════════════════════════════════╣
║ TITLE–ABSTRACT CONSISTENCY                       ║
║                                                  ║
║ Score: 84.8%                                     ║
║ Assessment: Consistent                           ║
╠══════════════════════════════════════════════════╣
║ VENUE RECOMMENDATION                             ║
║                                                  ║
║ 1. Journal of Biopesticides           52.1%      ║
║ 2. Indian Journal of Entomology       36.2%      ║
║ 3. Journal of Entomological Research  30.0%      ║
║ 4. International Journal of Tropi...  25.5%      ║
║ 5. Phytoparasitica                    25.3%      ║
╠══════════════════════════════════════════════════╣
║ BEST VENUE                                       ║
║                                                  ║
║ Journal of Biopesticides                         ║
║                                                  ║
║ Scope Compatibility: HIGH                        ║
║ Scope-Mismatch Risk: LOW  (p=0.01, dist=0.60)    ║
╚══════════════════════════════════════════════════╝

EXPLAINABILITY
  Key concepts : ants, potential, feeding, invasion, repeated, agent
  Similar papers in best venue:
    - Biocontrol of home invading rubber litter beetle, Luprops tristis with weaver ants
    - Bioefficacy of coccinellid predators on major tea pests
    - Population and predatory potency of spiders in brinjal and snakegourd
```

The true journal of this paper is *Journal of Biopesticides*, ranked first. Venue percentages are calibrated compatibility scores, so they do **not** sum to 100%.

---

## Architecture

```
Title + Abstract
      │
      ├── Stream A: Title encoder → Abstract encoder → Cross-attention   (SciBERT, fine-tuned)
      └── Stream B: SPECTER-2 scientific embedding → Projection          (frozen, cached)
                         │
                   Gated fusion  (semantic + title–abstract alignment)
                         │
                Paper representation h_p
        ┌────────────────┼──────────────────┐
  Venue recommender   Scope-mismatch    Consistency
  (multi-prototype    estimator         head
   journal profiles)
        └────────────────┼──────────────────┘
                  Explainability
        (key concepts · similar papers · DOI evidence)
```

| Component | Description |
|---|---|
| Stream A | Shared SciBERT encodes title and abstract; bidirectional title↔abstract cross-attention; attention pooling |
| Stream B | SPECTER-2 embedding of `title [SEP] abstract`, computed once and cached |
| Gated fusion | Learned gate mixes both streams plus title–abstract alignment features |
| Venue recommender | 3 learnable prototypes per journal; soft-max over prototype cosine similarities |
| Scope-mismatch estimator | Paper–venue pair → mismatch probability and scope distance |
| Consistency head | Predicts whether the title and abstract belong together |

The model has 121.1M parameters. Training uses multi-task loss (venue cross-entropy with label smoothing, consistency BCE, scope BCE, prototype-diversity regularizer) with mixed precision.

---

## Dataset

Scopus export of article metadata (`Scopus_unique_data.csv`): title, abstract, source title (journal), DOI and keywords.

Preprocessing:
- Duplicate titles and abstracts removed.
- Publisher copyright lines (for example "© 2024 Publisher") stripped from abstracts, because they leak the journal.
- Journals with fewer than 10 papers dropped (closed-set recommendation).
- Stratified 80/10/10 split per journal.

| | Count |
|---|---|
| Papers | 10,565 |
| Venues | 213 |
| Train / Val / Test | 8,457 / 1,054 / 1,054 |

The dataset is not redistributed in this repository.

---

## Installation

```bash
git clone https://github.com/<your-username>/SciFit-Net.git
cd SciFit-Net
pip install -r requirements.txt
```

Requirements: Python 3.9+, PyTorch, Transformers, scikit-learn, SciPy, pandas, NumPy, tqdm, Streamlit. A GPU is recommended for training (developed and tested on a Kaggle T4, about 170 s per epoch).

---

## Usage

### Train and evaluate

```bash
python scifit_net.py --mode all --csv /path/to/Scopus_unique_data.csv
```

| Option | Meaning |
|---|---|
| `--epochs 8` | Number of epochs |
| `--batch 16` | Batch size (use `8` or `--grad_ckpt` if out of memory) |
| `--min_papers 10` | Minimum papers per journal |
| `--n_proto 3` | Prototypes per journal |
| `--debug_n 1500 --epochs 1` | Quick smoke test |

Outputs are written to `scifit_out/`: `best.pt`, `meta.json`, `index.npz`, `index_meta.json`, `venue_geometry.npz`, `test_results.json`.

### Evaluate a saved model

```bash
python scifit_net.py --mode eval --csv /path/to/Scopus_unique_data.csv
```

### Predict for one paper

```bash
python scifit_net.py --mode predict --title "Your title" --abstract "Your abstract"
```

### Web app

```bash
streamlit run app.py
```

Set the model folder in the sidebar, or use the `SCIFIT_OUT` environment variable. The first prediction downloads SPECTER-2.

---

## Results (test split, 1,054 papers)

### 1. Venue recommendation

| Metric | SciFit-Net | SPECTER-2 centroid | Frequency prior |
|---|---|---|---|
| Hit@1 | **0.2970** | 0.1423 | 0.0569 |
| Hit@3 | **0.5000** | 0.2865 | 0.1338 |
| Hit@5 | **0.6015** | 0.3776 | 0.1954 |
| Hit@10 | **0.7362** | 0.5342 | 0.3245 |
| MRR | **0.4388** | 0.2652 | 0.1409 |
| NDCG@5 | **0.4569** | 0.2618 | 0.1261 |
| Macro-F1@1 | **0.1400** | 0.1198 | 0.0005 |
| Median rank | **3.5** | 9.0 | 23.0 |

SciFit-Net roughly doubles Hit@1 over the SPECTER-2 centroid baseline and is about 5× better than the frequency prior, across 213 fine-grained venues.

Performance by venue size (training papers per journal):

| Group | n (test) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|---|
| Head (≥ 50) | 691 | 0.3792 | 0.7004 | 0.5216 |
| Mid (20–49) | 187 | 0.1872 | 0.5080 | 0.3428 |
| Tail (< 20) | 176 | 0.0909 | 0.3125 | 0.2154 |

### 2. Scope-mismatch estimator (proxy labels)

| Metric | Value |
|---|---|
| AUROC / AUPRC | 0.9635 / 0.9184 |
| F1 / Precision / Recall (threshold 0.26) | 0.8605 / 0.8129 / 0.9141 |
| Accuracy | 0.9001 |
| Brier / ECE | 0.1010 / 0.1187 |
| Spearman vs. soft target | 0.8155 |
| AUROC, true vs. far venue | 0.9935 |
| Pairwise order, true < far | 0.9886 |
| Pairwise order, true < near | 0.6784 |
| Mean mismatch: true / near / far venue | 0.048 / 0.063 / 0.565 |
| Top-1 error detection AUROC | 0.5698 |

### 3. Title–abstract consistency (synthetic negatives)

| Metric | Value |
|---|---|
| AUROC / AUPRC | 0.9745 / 0.9743 |
| EER | 0.0840 |
| Brier / ECE | 0.0705 / 0.0495 |
| Accuracy (threshold 0.5 / tuned 0.83) | 0.9023 / 0.9170 |
| F1 / Precision / Recall (tuned) | 0.9159 / 0.9279 / 0.9042 |
| AUROC vs. easy negatives (random abstract) | 0.9949 |
| AUROC vs. hard negatives (similar-topic abstract) | 0.9522 |
| Mean score: true / easy neg. / hard neg. | 0.931 / 0.033 / 0.290 |

---

## Limitations and honest notes

- **Venue recommendation is hard.** Many journals overlap heavily in topic (for example *BioControl* and *Entomologia Experimentalis et Applicata*), so the true journal often appears in the top 5 but not at rank 1. Rare journals are much weaker than frequent ones (Hit@1 0.09 for tail venues).
- **Overfitting after epoch 5.** Training venue loss kept falling while validation Hit@1/MRR plateaued; the best checkpoint is from epoch 5 and early stopping ended training at epoch 8.
- **Proxy labels for two heads.** The dataset has no ground-truth labels for scope mismatch or consistency. Consistency positives are real title–abstract pairs; negatives pair a title with another paper's abstract (half random, half from similar-topic papers). Scope mismatch uses soft targets from venue distances between SPECTER-2 centroids of training data. Only the venue metrics use real labels.
- **The scope estimator separates far-off venues, not near ones.** It ranks the true venue below far venues almost perfectly (0.99) but below near, related venues much less reliably (0.68), and its mismatch score barely distinguishes correct from wrong top-1 recommendations (AUROC 0.57). Treat "Scope-Mismatch Risk: LOW" as "topic is in the right area," not as a guarantee of fit.
- **Scope distance** is a raw cosine-based distance in the learned space and varies little between papers; prefer the mismatch probability.
- **Closed set.** Only journals present in the training data can be recommended.
- **Dataset bias.** The data is dominated by entomology, biological control and ecology; the model will be weaker outside these areas.
- **Not a guarantee.** Always check a journal's aims and scope before submitting.

---

## Project structure

```
SciFit-Net/
├── scifit_net.py      # data prep, model, training, evaluation, inference
├── app.py             # Streamlit app
├── requirements.txt
├── README.md
└── scifit_out/        # trained model files (not committed; see below)
```

## Model weights

Weights are not stored in this repository because of their size. Download them from: `<Kaggle / Hugging Face / GitHub Release link>` and place the files in `scifit_out/`.

---

## Citation

```bibtex
@misc{scifitnet2026,
  title  = {SciFit-Net: Explainable Journal Recommendation with Scope-Mismatch Estimation},
  author = {Rakib},
  year   = {2026},
  url    = {https://github.com/<your-username>/SciFit-Net}
}
```

## Acknowledgements

SciBERT and SPECTER-2 (Allen Institute for AI), PyTorch, Hugging Face Transformers, Streamlit.

## License

Add a license (for example MIT) before publishing.

## Author

Rakib, Department of CSE, IUBAT
