# SciFit-Net

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python" alt="Python 3.9+" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Transformers-4.x-FFD21E?style=for-the-badge&logo=huggingface" alt="Transformers" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit" />
</p>

<p align="center">
  <strong>Explainable journal recommendation from a title and abstract, with scope-mismatch estimation and title–abstract consistency checking.</strong>
</p>

SciFit-Net evaluates a manuscript's title and abstract to deliver:

- a title–abstract consistency score,
- a ranked list of recommended journals with compatibility scores,
- a scope-compatibility and scope-mismatch risk assessment for the best venue,
- explainability outputs such as key concepts and similar papers with DOI references.

---

## Why SciFit-Net?

Choosing the right journal is often a mix of semantic fit, topical relevance, and editorial scope. SciFit-Net models this process end-to-end using a dual-stream architecture that combines:

- scientific language understanding from SciBERT,
- citation-aware scientific embeddings from SPECTER-2,
- multi-prototype venue modeling,
- explainability for real-world decision support.

This makes it suitable for authors, editorial teams, and research support workflows where transparency matters as much as accuracy.

---

## Example output

Below is a real model output on a test paper:

```text
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
  Key concepts: ants, potential, feeding, invasion, repeated, agent
  Similar papers in best venue:
    - Biocontrol of home invading rubber litter beetle, Luprops tristis with weaver ants
    - Bioefficacy of coccinellid predators on major tea pests
    - Population and predatory potency of spiders in brinjal and snakegourd
```

The true journal for this paper is *Journal of Biopesticides*, which appears ranked first. The venue percentages are calibrated compatibility scores and do not sum to 100%.

---

## Model architecture

```text
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
| Stream A | Shared SciBERT encodes the title and abstract; bidirectional title↔abstract cross-attention is applied, followed by attention pooling. |
| Stream B | SPECTER-2 embedding of `title [SEP] abstract`, computed once and cached. |
| Gated fusion | A learned gate mixes both streams and title–abstract alignment features. |
| Venue recommender | Three learnable prototypes per journal; softmax over prototype cosine similarities. |
| Scope-mismatch estimator | Maps a paper–venue pair to mismatch probability and scope distance. |
| Consistency head | Predicts whether the title and abstract belong together. |

The model contains approximately 121.1M parameters. Training uses a multi-task objective combining venue cross-entropy (with label smoothing), consistency BCE, scope BCE, and a prototype-diversity regularizer, using mixed precision.

---

## Dataset

The training data is a Scopus export of article metadata stored in `Scopus_unique_data.csv`, including:

- title,
- abstract,
- source title (journal),
- DOI,
- keywords.

### Preprocessing

- Duplicate titles and abstracts were removed.
- Publisher copyright lines such as `© 2024 Publisher` were stripped from abstracts, because they leak the journal identity.
- Journals with fewer than 10 papers were excluded from recommendation.
- A stratified 80/10/10 split was applied per journal.

| Item | Count |
|---|---:|
| Papers | 10,565 |
| Venues | 213 |
| Train / Val / Test | 8,457 / 1,054 / 1,054 |

> The dataset itself is not redistributed in this repository.

---

## Installation

```bash
git clone https://github.com/<your-username>/SciFit-Net.git
cd SciFit-Net
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- PyTorch
- Transformers
- scikit-learn
- SciPy
- pandas
- NumPy
- tqdm
- Streamlit

A GPU is strongly recommended for training. The project was developed and tested on a Kaggle T4, with approximately 170 seconds per epoch.

---

## Usage

### Train and evaluate

```bash
python scifit_net.py --mode all --csv /path/to/Scopus_unique_data.csv
```

| Option | Meaning |
|---|---|
| `--epochs 8` | Number of training epochs |
| `--batch 16` | Batch size (use `8` or `--grad_ckpt` if memory is limited) |
| `--min_papers 10` | Minimum papers required per journal |
| `--n_proto 3` | Prototypes generated per journal |
| `--debug_n 1500 --epochs 1` | Quick smoke test |

Outputs are saved in `scifit_out/`:

- `best.pt`
- `meta.json`
- `index.npz`
- `index_meta.json`
- `venue_geometry.npz`
- `test_results.json`

### Evaluate a saved model

```bash
python scifit_net.py --mode eval --csv /path/to/Scopus_unique_data.csv
```

### Predict for one paper

```bash
python scifit_net.py --mode predict --title "Your title" --abstract "Your abstract"
```

### Launch the web app

```bash
streamlit run app.py
```

Set the model folder in the sidebar, or define the `SCIFIT_OUT` environment variable. On the first prediction, the app will download SPECTER-2.

---

## Results on the test split (1,054 papers)

### 1) Venue recommendation

| Metric | SciFit-Net | SPECTER-2 centroid | Frequency prior |
|---|---:|---:|---:|
| Hit@1 | **0.2970** | 0.1423 | 0.0569 |
| Hit@3 | **0.5000** | 0.2865 | 0.1338 |
| Hit@5 | **0.6015** | 0.3776 | 0.1954 |
| Hit@10 | **0.7362** | 0.5342 | 0.3245 |
| MRR | **0.4388** | 0.2652 | 0.1409 |
| NDCG@5 | **0.4569** | 0.2618 | 0.1261 |
| Macro-F1@1 | **0.1400** | 0.1198 | 0.0005 |
| Median rank | **3.5** | 9.0 | 23.0 |

SciFit-Net roughly doubles Hit@1 relative to the SPECTER-2 centroid baseline and is approximately 5× better than the frequency prior across 213 fine-grained venues.

#### Performance by venue size

| Group | n (test) | Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Head (≥ 50) | 691 | 0.3792 | 0.7004 | 0.5216 |
| Mid (20–49) | 187 | 0.1872 | 0.5080 | 0.3428 |
| Tail (< 20) | 176 | 0.0909 | 0.3125 | 0.2154 |

### 2) Scope-mismatch estimator (proxy labels)

| Metric | Value |
|---|---:|
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

### 3) Title–abstract consistency (synthetic negatives)

| Metric | Value |
|---|---:|
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

- Venue recommendation remains difficult. Many journals overlap heavily in topic, such as *BioControl* and *Entomologia Experimentalis et Applicata*, so the correct venue frequently appears in the top 5 but not always in the top 1.
- Overfitting can begin after epoch 5. Validation performance plateaued while training loss continued to drop; the strongest checkpoint was selected from epoch 5.
- For the two auxiliary heads, proxy labels are used because no direct ground-truth labels exist for scope mismatch or consistency. Positives are real title–abstract pairs; negatives are constructed by pairing titles with mismatched or similar abstracts.
- The scope estimator is strongest at separating far-off venues from the true venue; it is less reliable for near-related venues.
- Scope distance is a raw cosine-based distance in the learned representation space and varies little across papers. The mismatch probability is generally the more informative signal.
- The system is a closed-set recommender: only journals seen in the training data can be recommended.
- The dataset is biased toward entomology, biological control, and ecology; performance may be weaker outside these domains.
- This is not a guarantee of fit or acceptance. Always verify a journal's aims and scope before submission.

---

## Project structure

```text
SciFit-Net/
├── scifit_net.py      # data prep, model, training, evaluation, inference
├── app.py             # Streamlit app
├── requirements.txt
├── README.md
└── scifit_out/       # trained model files (not committed; see below)
```

---

## Model weights

Model weights are not included in this repository because of their size. Download them from a release, Kaggle asset, or Hugging Face location and place them in `scifit_out/` before running predictions or evaluation.

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

---

## Acknowledgements

SciBERT and SPECTER-2 (Allen Institute for AI), PyTorch, Hugging Face Transformers, and Streamlit.

---

## License

Add a license (for example MIT) before publishing.

---

## Author

Rakib  
Department of CSE, IUBAT

---

<p align="center">
  <sub>Built for explainable, data-driven journal matching.</sub>
</p>
