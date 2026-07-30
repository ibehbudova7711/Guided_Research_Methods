# Measuring Gender, Age, and Nationality Bias in BERT and RoBERTa

## Overview

This project investigates demographic bias in two transformer-based masked language models, **BERT** and **RoBERTa**, using the **CrowS-Pairs** benchmark dataset.

The evaluation is based on **Pseudo Log-Likelihood (PLL)** scoring, where each model is asked to compare stereotypical and less stereotypical sentence pairs. The study focuses on three demographic categories:

- Gender
- Age
- Nationality

The objective is to compare how frequently each model prefers stereotypical sentences and to identify differences in bias across categories.

---

## Features

- CrowS-Pairs dataset filtering
- Pseudo Log-Likelihood (PLL) implementation
- BERT evaluation
- RoBERTa evaluation
- Statistical analysis
- Automatic visualization
- Model comparison
- Reproducible evaluation pipeline

---

## Repository Structure

```
Bert-Roberta_BiasAnalysis/

├── data/
│
├── results/
│   ├── csv/
│   ├── bert_results/
│   ├── roberta_results/
│   └── comparison/
│
├── src/
│   ├── bert/
│   ├── roberta/
│   ├── comparison/
│   ├── experiments/
│   ├── load_data.py
│   ├── filter_data.py
│   └── explore_data.py
│
├── docs/
├── requirements.txt
└── README.md
```

---

## Dataset

**Dataset:** CrowS-Pairs

The original CrowS-Pairs dataset was filtered to include only the demographic categories used in this study.

| Category | Examples |
|-----------|---------:|
| Gender | 262 |
| Age | 87 |
| Nationality | 159 |
| **Total** | **508** |

---

## Models

The following pretrained Hugging Face models were evaluated:

- BERT (`bert-base-uncased`)
- RoBERTa (`roberta-base`)

Both models were evaluated using the same dataset and identical PLL-based scoring methodology.

---

## Methodology

The evaluation pipeline consists of the following steps:

1. Load the CrowS-Pairs dataset.
2. Filter the target demographic categories.
3. Compute PLL scores for both sentences in each pair.
4. Determine whether the model prefers the stereotypical or less stereotypical sentence.
5. Calculate bias statistics.
6. Generate visualizations.
7. Compare BERT and RoBERTa.

---

## Results

### Overall stereotypical preference

| Model | Preference Rate |
|--------|----------------:|
| BERT | 54.53% |
| RoBERTa | 58.07% |

### Category comparison

| Category | BERT | RoBERTa |
|-----------|------:|---------:|
| Gender | 57.25% | 57.63% |
| Age | 60.92% | 63.22% |
| Nationality | 46.54% | 55.97% |

The comparison indicates that RoBERTa showed a higher stereotypical preference than BERT in the evaluated dataset, with the largest difference observed in the nationality category.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Pandas
- Matplotlib
- tqdm

---

## How to Run

Filter the dataset:

```bash
python src/filter_data.py
```

Run the BERT evaluation:

```bash
python src/bert/evaluate_bert_dataset.py
python src/bert/analyze_bert_results.py
python src/bert/visualize_bert_results.py
```

Run the RoBERTa evaluation:

```bash
python src/roberta/evaluate_roberta_dataset.py
python src/roberta/analyze_roberta_results.py
python src/roberta/visualize_roberta_results.py
```

Compare both models:

```bash
python src/comparison/compare_models.py
```

---

## Future Work

Possible extensions of this project include:

- Evaluation on additional demographic categories.
- Comparison with larger language models.
- Investigation of bias mitigation techniques.
- Statistical significance testing.
- Evaluation using additional benchmark datasets.

---

## Author

**Ilaha Behbudova**

ADA University & George Washington University