# DocuExtract Dataset Validation Report
**Generated**: January 04, 2026
**Project**: Odia Handwritten OCR System
**Target**: >90% accuracy on handwritten Odia documents

---

## 📊 Current Dataset Inventory

| Dataset | Size | Images | Type | Relevance |
|---------|------|--------|------|-----------|
| **Odia_handwritten** | 1.1 GB | 253,627 | Handwritten characters | ✅ **CRITICAL** |
| **Odia_Number** | 11 MB | 120 | Handwritten digits | ✅ **HIGH** |
| **IndicSTR (Odia only)** | 17 GB total | ~333 Odia | Scene text | ✅ **MEDIUM** |
| **Mozhi/IndicNLP** | 5.7 GB | 5.58M text | Text corpus | ✅ **MEDIUM** |
| **Wikipedia** | 248 MB | Text | Text corpus | ⚠️ **LOW** |
| **IndicDLP** | 4 KB | 0 | Document layout | ❌ **EMPTY** |

---

## ✅ RELEVANT Datasets for OCR Training

### 1. **Odia_handwritten (HIGHEST PRIORITY)** ⭐⭐⭐
- **Size**: 253,627 handwritten character images
- **Structure**:
  - `Odia_Handwritten_Simple_Character/` - Basic Odia characters
  - `Odia_Compound_Character/` - Compound characters (matras, conjuncts)
- **Relevance**: **CRITICAL** - This is your PRIMARY training data
- **Use Case**: Train character-level OCR models (DeepSeek-OCR, olmOCR, Qwen2.5-VL)
- **Project Plan Match**: Aligns with OHCS v1.0 requirement (handwritten characters)

**Validation Required**:
- [ ] Check annotation format (filenames, labels)
- [ ] Verify all 57 OHCS characters are present
- [ ] Analyze image quality (resolution, noise)
- [ ] Create train/val/test splits (80/10/10)

---

### 2. **Odia_Number (HIGH PRIORITY)** ⭐⭐
- **Size**: 120 handwritten digit images
- **Relevance**: **HIGH** - Supplement for digit recognition in documents
- **Use Case**: Augment training data for numerical text recognition
- **Note**: Smaller dataset, may need augmentation

**Validation Required**:
- [ ] Check digit coverage (0-9 in Odia script)
- [ ] Verify annotation format
- [ ] Consider data augmentation

---

### 3. **IIIT-IndicSTR (MEDIUM PRIORITY)** ⭐⭐
- **Total**: 50,644 images (ALL languages)
- **Odia Only**: ~333 images (verified)
- **Type**: Scene text + document text
- **Relevance**: **MEDIUM** - Good for word/sentence-level OCR
- **Use Case**: Fine-tuning for natural scene text recognition

**Validation Required**:
- [ ] Extract only Odia images from real_extracted folder
- [ ] Check if annotations exist (text labels)
- [ ] Decide: Use for training or just evaluation?

**Action**: Create filtered dataset with Odia images only

---

### 4. **Mozhi/IndicNLP Text Corpus (MEDIUM PRIORITY)** ⭐⭐
- **Size**: 5.58 Million Odia text samples
- **Format**: JSONL text data
- **Relevance**: **MEDIUM** - For synthetic data generation
- **Use Case**: Generate 15K synthetic images using text rendering

**Per Project Plan (Phase 1, Day 5-6)**:
> "Generate 15K synthetic images using Mozhi corpus"

**Validation Required**:
- [ ] Select diverse sentences (short/long, simple/complex)
- [ ] Use for synthetic image generation with Odia fonts
- [ ] Verify text quality (no encoding issues)

---

## ❌ LESS RELEVANT / NOT USABLE

### Wikipedia Text Data ⚠️
- **Size**: 248 MB text
- **Relevance**: **LOW** - Backup text corpus
- **Use**: Optional synthetic data generation if Mozhi is insufficient

### IndicDLP ❌
- **Status**: Empty (0 images)
- **Reason**: Gated dataset, cannot filter Odia-only documents
- **Action**: **Skip** - Not practical for this project

---

## 📁 Recommended Folder Structure

```
docuextratc/
├── data/
│   ├── raw/                          # Original downloaded datasets
│   │   ├── handwritten_characters/   # 253,627 images (Odia_handwritten)
│   │   ├── handwritten_numbers/      # 120 images (Odia_Number)
│   │   ├── scene_text/                # 333 Odia images (IndicSTR filtered)
│   │   └── text_corpus/               # 5.58M text samples (Mozhi)
│   │
│   ├── processed/                     # Standardized format
│   │   ├── train/                     # 80% split
│   │   ├── val/                       # 10% split
│   │   └── test/                      # 10% split (OHCS holdout)
│   │
│   └── synthetic/                     # Generated images
│       └── mozhi_synthetic_15k/       # 15K synthetic images
│
├── annotations/                       # Label files
│   ├── train_labels.json
│   ├── val_labels.json
│   └── test_labels.json
│
└── experiments/                       # Training outputs
    └── ...
```

---

## 🎯 Next Actions (Phase 1 - Week 1)

### Day 1-2: Dataset Inventory ✅ COMPLETE
- [x] Download all datasets
- [x] Create dataset inventory
- [x] Document dataset statistics

### Day 3-4: Data Validation & Preprocessing (NEXT)
1. **Validate Odia_handwritten**:
   ```bash
   python validate_handwritten.py --check-labels --count-per-class
   ```

2. **Extract Odia images from IndicSTR**:
   ```bash
   python filter_odia_indicstr.py --input indicstr_data/real_extracted --output data/raw/scene_text
   ```

3. **Check character completeness**:
   - Verify all 57 OHCS Odia characters are present
   - Identify missing characters
   - Plan synthetic generation for missing classes

4. **Create annotation files**:
   - Convert folder structure to standardized JSON format
   - Format: `{"image_path": "...", "text": "...", "label_id": ...}`

### Day 5-6: Dataset Splitting & Synthetic Generation
- Create stratified train/val/test splits
- Generate 15K synthetic images using Mozhi corpus
- Implement augmentation pipeline (rotation, noise, blur)

---

## 📊 Expected Final Dataset Statistics

| Split | Source | Images | Purpose |
|-------|--------|--------|---------|
| Train | Handwritten chars | ~203,000 | Primary training |
| Train | Synthetic (Mozhi) | 15,000 | Augmentation |
| Val | Handwritten chars | ~25,000 | Validation |
| Test | Handwritten chars | ~25,000 | Final evaluation |
| Scene | IndicSTR Odia | 333 | Scene text eval |
| **TOTAL** | | **~268,333** | **Full dataset** |

---

## ⚠️ Critical Validation Checks

Before proceeding to Phase 2 (Model Training):

- [ ] **Character Coverage**: All 57 OHCS characters present?
- [ ] **Annotation Format**: Consistent across all images?
- [ ] **Image Quality**: Resolution ≥224×224 (or 384×384)?
- [ ] **Data Balance**: Each character has minimum samples?
- [ ] **Split Integrity**: No data leakage between train/val/test?
- [ ] **File Integrity**: All images can be loaded without errors?

---

## 📝 Notes

1. **OHCS v1.0 Dataset**: Not found in current downloads. The `Odia_handwritten` dataset appears to be similar (character-level handwritten data). Need to verify if this is OHCS or a different source.

2. **IndicSTR Odia Filtering**: Only ~333 out of 50,644 images are Odia. Need to extract these carefully.

3. **Synthetic Data Strategy**: Use Mozhi corpus to generate word/sentence-level images to complement character-level training.

4. **Missing Datasets**:
   - OHCS v1.0 (17,100 images) - May need to request from NIT Rourkela
   - Kaggle Odia OCR - Not downloaded (backup)

---

**Status**: Ready for preprocessing pipeline development
**Next Milestone**: Complete Day 3-4 validation by end of week
