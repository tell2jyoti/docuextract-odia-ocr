# Odia Dataset Download Scripts

This directory contains Python scripts to download various Odia language datasets for OCR and NLP tasks.

## Overview

| Dataset | Script | Size | Type | Status |
|---------|--------|------|------|--------|
| 1. Mozhi/IndicNLP Corpus | `download_mozhi.py` | 2.18 GB | Text corpus | ✅ Working |
| 2. IIIT-IndicSTR12 | `download_indicstr.py` | ~1.3 GB | Scene text images | ✅ Working |
| 3. IITBBS-OCR | Manual download | N/A | Handwritten chars | ⚠️ Requires IEEE account |
| 4. AI4Bharat IndicDLP | `download_indicdlp.py` | Variable | Document layout/OCR | ✅ Working |
| 5. Odia Wikipedia | `download_wikipedia.py` | ~200 MB | Text for synthetic data | ✅ Working |

## Installation

Install required dependencies:

```bash
pip install datasets requests beautifulsoup4
```

## Usage

All scripts have two modes:

### 1. SAMPLE MODE (Default)
Test with a small sample to verify the script works:

```bash
python3 download_mozhi.py        # Shows 10 samples
python3 download_indicstr.py     # Downloads real dataset only (~500MB)
python3 download_indicdlp.py     # Shows 10 Odia document samples
python3 download_wikipedia.py    # Extracts 100 articles
```

### 2. FULL MODE
Download complete datasets:

Edit the script and change `SAMPLE_MODE = True` to `SAMPLE_MODE = False`, then run:

```bash
python3 download_mozhi.py        # Downloads all 2.18 GB Odia text
python3 download_indicstr.py     # Downloads real + synthetic datasets
python3 download_indicdlp.py     # Downloads all Odia documents
python3 download_wikipedia.py    # Downloads full Wikipedia dump
```

## Dataset Details

### 1. Mozhi / IndicNLP Corpus
**Script**: `download_mozhi.py`

- **Source**: HuggingFace ai4bharat/IndicCorpV2
- **Size**: 2.18 GB of Odia text
- **Content**: 1.2M words from web crawls, news, articles
- **Output**: `mozhi_data/mozhi_odia_full.jsonl`
- **Use case**: Language modeling, NLP tasks

### 2. IIIT-IndicSTR12
**Script**: `download_indicstr.py`

- **Source**: CVIT IIIT Hyderabad
- **Paper**: https://arxiv.org/abs/2403.08007
- **Size**: ~1.3 GB (real), ~3 GB (synthetic)
- **Content**:
  - Real dataset: 1000+ Odia scene text images
  - Synthetic dataset: Additional generated images
- **Output**: `indicstr_data/real_extracted/` and `indicstr_data/synthetic_extracted/`
- **Use case**: Scene text recognition, OCR training

### 3. IITBBS-OCR Dataset
**Manual Download Required**

- **Source**: IEEE DataPort
- **URL**: https://ieee-dataport.org/documents/iitbbs-ocr-dataset
- **DOI**: 10.21227/rxm0-vz55
- **Content**: Handwritten Odia characters and numerals
- **Access**: Requires IEEE DataPort account (free)
- **Contact**: Dr. Niladri B. Puhan (nbpuhan@iitbbs.ac.in)
- **Use case**: Handwritten character recognition

**To download:**
1. Visit https://ieee-dataport.org/documents/iitbbs-ocr-dataset
2. Create free IEEE DataPort account
3. Download the dataset
4. Extract to `iitbbs_data/` directory

### 4. AI4Bharat IndicDLP
**Script**: `download_indicdlp.py`

- **Source**: HuggingFace ai4bharat/indicdlp
- **Paper**: ICDAR 2025 (Best Student Paper Runner-Up)
- **Content**: Document layout parsing dataset
  - 11 Indic languages + English
  - 12 document types
  - 42 layout classes
  - Odia: 8.2% of dataset (~9,800 images)
- **Output**: `indicdlp_data/indicdlp_odia_full.jsonl`
- **Use case**: Document layout analysis, OCR, document understanding

### 5. Odia Wikipedia Dumps
**Script**: `download_wikipedia.py`

- **Source**: Wikimedia dumps (https://dumps.wikimedia.org/orwiki/)
- **Size**: ~200 MB compressed
- **Content**: All Odia Wikipedia articles
- **Output**: `wikipedia_data/odia_wikipedia_full.jsonl`
- **Use case**: Synthetic training data generation, language modeling

## Output Format

### Text datasets (Mozhi, Wikipedia)
JSONL format:
```json
{"id": 0, "text": "ଓଡ଼ିଆ ଲେଖା..."}
{"id": 1, "text": "ଆଉ ଏକ ନମୁନା..."}
```

### Image datasets (IndicSTR)
Directory structure:
```
indicstr_data/
├── real_extracted/
│   └── odia/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── synthetic_extracted/
    └── odia/
        ├── image_001.png
        └── ...
```

### Document datasets (IndicDLP)
JSONL format with metadata:
```json
{"id": 0, "language": "odia", "document_type": "novel", "annotations": [...]}
```

## Troubleshooting

### Issue: Download fails
- **Solution**: Check internet connection, try again later

### Issue: HuggingFace access denied
- **Solution**: Some datasets may require HuggingFace login:
  ```bash
  huggingface-cli login
  ```

### Issue: Out of disk space
- **Solution**: Run in SAMPLE_MODE first, ensure you have enough space:
  - Mozhi: ~2.5 GB
  - IndicSTR: ~4.5 GB (both datasets)
  - Wikipedia: ~500 MB
  - IndicDLP: ~1 GB

### Issue: Dependencies missing
- **Solution**: Install all dependencies:
  ```bash
  pip install datasets requests beautifulsoup4 Pillow
  ```

## License Information

- **Mozhi/IndicNLP**: CC BY-NC-SA 4.0
- **IndicSTR12**: Check CVIT website for license
- **IITBBS-OCR**: Check IEEE DataPort for license
- **IndicDLP**: Check AI4Bharat for license
- **Wikipedia**: CC BY-SA 3.0

## Citation

If you use these datasets, please cite the original papers:

### Mozhi/IndicNLP:
```bibtex
@article{kunchukuttan2020ai4bharat,
  title={AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
  author={Kunchukuttan, Anoop and others},
  journal={arXiv preprint arXiv:2005.00085},
  year={2020}
}
```

### IndicSTR12:
```bibtex
@inproceedings{lunia2023indicstr12,
  title={IndicSTR12: A Dataset for Indic Scene Text Recognition},
  author={Lunia, Harsh and others},
  booktitle={ICDAR},
  year={2023}
}
```

### IndicDLP:
```bibtex
@inproceedings{karthik2025indicdlp,
  title={IndicDLP: A Large-Scale Document Layout Parsing Dataset for Indic Languages},
  author={Karthik, N V and others},
  booktitle={ICDAR},
  year={2025}
}
```

## Support

For issues with specific datasets, contact:
- **Mozhi/IndicNLP**: AI4Bharat team
- **IndicSTR**: CVIT IIIT Hyderabad
- **IITBBS-OCR**: nbpuhan@iitbbs.ac.in
- **IndicDLP**: AI4Bharat team
- **Wikipedia**: Wikimedia Foundation

## Next Steps

After downloading datasets:
1. Verify data quality with sample mode
2. Check for Odia script correctness
3. Process for your specific use case (OCR training, NLP, etc.)
4. Combine datasets as needed for your project

Happy dataset collecting! 🎉
