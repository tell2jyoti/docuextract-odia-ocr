# DocuExtract Project Plan
**Odia Handwritten OCR System**
**Duration**: 6 weeks
**Target**: >90% accuracy on handwritten Odia documents

---

## 📋 Project Overview

### Objective
Build a state-of-the-art OCR system for handwritten Odia (ଓଡ଼ିଆ) text using progressive fine-tuning of vision-language models.

### Approach
Progressive fine-tuning strategy:
1. **DeepSeek-OCR** → Character-level baseline
2. **olmOCR** → Improved accuracy
3. **Qwen2.5-VL** → Production model

---

## 🗓️ Timeline

### **Phase 1: Dataset Preparation (Week 1-2)** ✅ COMPLETE

**Week 1: Data Collection & Organization**
- [x] Download Odia handwritten character datasets
- [x] Download scene text datasets (IndicSTR)
- [x] Download text corpus (Mozhi/IndicNLP)
- [x] Organize datasets into relevant/non-relevant
- [x] Create dataset inventory and validation report

**Week 2: Annotation & Preprocessing**
- [x] Create annotation generation pipeline
- [x] Map image filenames to character labels
- [x] Identify class imbalance and missing classes
- [x] Balance dataset (downsample + augment)
- [x] Generate synthetic data for missing classes
- [x] Create train/val/test splits (80/10/10)

**Deliverables**:
- ✅ 182,152 annotated images
- ✅ All 47 OHCS characters (3K-10K samples each)
- ✅ Balanced, stratified dataset splits
- ✅ Comprehensive documentation

---

### **Phase 2: Model Training (Week 3-6)** 🔄 IN PROGRESS

#### **Week 3-4: DeepSeek-OCR Baseline**

**Objectives**:
- Establish character-level OCR baseline
- Understand model behavior
- Optimize training hyperparameters

**Tasks**:
- [ ] Set up training environment (PyTorch, Transformers)
- [ ] Implement data loaders with augmentation
- [ ] Configure DeepSeek-OCR for Odia characters
- [ ] Train initial model (50 epochs)
- [ ] Evaluate on validation set
- [ ] Analyze errors and misclassifications

**Target Metrics**:
- Training Accuracy: >85%
- Validation Accuracy: >80%
- Character Error Rate (CER): <15%

**Deliverables**:
- Trained DeepSeek-OCR checkpoint
- Training logs and metrics
- Error analysis report

---

#### **Week 5: olmOCR Fine-tuning**

**Objectives**:
- Improve accuracy through transfer learning
- Reduce character error rate
- Handle difficult character classes

**Tasks**:
- [ ] Load DeepSeek-OCR checkpoint
- [ ] Configure olmOCR architecture
- [ ] Fine-tune on balanced dataset
- [ ] Apply advanced augmentation
- [ ] Hyperparameter optimization
- [ ] Evaluate on test set

**Target Metrics**:
- Validation Accuracy: >90%
- Character Error Rate (CER): <10%
- Per-class accuracy: >85% for all classes

**Deliverables**:
- olmOCR checkpoint
- Improved metrics
- Per-class performance analysis

---

#### **Week 6: Qwen2.5-VL Final Model**

**Objectives**:
- Achieve production-level accuracy
- Optimize for inference speed
- Prepare for deployment

**Tasks**:
- [ ] Configure Qwen2.5-VL for OCR
- [ ] Progressive fine-tuning strategy
- [ ] Model optimization (quantization, pruning)
- [ ] Final evaluation on test set
- [ ] Real-world document testing
- [ ] Benchmark inference speed

**Target Metrics**:
- Character Error Rate (CER): <5%
- Word Error Rate (WER): <10%
- Inference speed: <100ms per image

**Deliverables**:
- Production-ready Qwen2.5-VL model
- Comprehensive evaluation report
- Deployment documentation

---

### **Phase 3: Evaluation & Deployment (Week 7-8)** 📅 PLANNED

#### **Week 7: Comprehensive Evaluation**

**Tasks**:
- [ ] Benchmark on multiple test sets
- [ ] Scene text evaluation (IndicSTR)
- [ ] Real-world document testing
- [ ] Error analysis and failure cases
- [ ] Compare with existing Odia OCR systems
- [ ] Generate performance visualizations

**Metrics to Report**:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Per-class precision/recall/F1
- Confusion matrix
- Processing speed (images/second)

---

#### **Week 8: Deployment & Documentation**

**Tasks**:
- [ ] Model optimization for production
- [ ] Create inference API (FastAPI/Flask)
- [ ] Build web interface
- [ ] Write deployment guide
- [ ] Create user documentation
- [ ] Publish model on Hugging Face
- [ ] Write technical paper/blog post

**Deliverables**:
- Production API
- Web interface
- Complete documentation
- Published model
- Technical write-up

---

## 📊 Dataset Details

### Sources

1. **Odia Handwritten Character Dataset**
   - Source: Public dataset
   - Size: 253,627 images
   - Type: Character-level handwritten samples
   - Usage: Primary training data

2. **IndicSTR Scene Text**
   - Source: AI4Bharat
   - Size: 333 Odia images
   - Type: Natural scene text
   - Usage: Real-world evaluation

3. **Mozhi/IndicNLP Corpus**
   - Source: AI4Bharat
   - Size: 5.58M Odia sentences
   - Type: Text corpus
   - Usage: Synthetic data generation

### Preprocessing Pipeline

```
Raw Images (253K)
  ↓
Annotation Creation
  ↓
Class Balancing (131K)
  ├─ Downsample overrepresented
  └─ Augment underrepresented
  ↓
Synthetic Generation (51K)
  ↓
Complete Dataset (182K)
  ↓
Train/Val/Test Split
  ├─ Train: 145,717 (80%)
  ├─ Val: 18,211 (10%)
  └─ Test: 18,224 (10%)
```

---

## 🎯 Performance Targets

### Minimum Acceptable Performance (MVP)

| Metric | MVP Target | Stretch Goal |
|--------|-----------|--------------|
| Character Error Rate (CER) | <10% | <5% |
| Word Error Rate (WER) | <20% | <10% |
| Training Accuracy | >90% | >95% |
| Validation Accuracy | >85% | >90% |
| Inference Speed | <500ms | <100ms |

### Per-Class Requirements

- Minimum accuracy per class: >80%
- No class with accuracy <70%
- Balance precision and recall

---

## 🛠️ Technical Stack

### Frameworks & Libraries

- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Pre-trained models
- **OpenCV**: Image processing
- **Pillow**: Image manipulation
- **NumPy/Pandas**: Data processing
- **Matplotlib**: Visualization
- **Weights & Biases**: Experiment tracking (optional)

### Models

1. **DeepSeek-OCR**
   - Purpose: Baseline character recognition
   - Architecture: Vision Transformer + Text Decoder
   - Training: From scratch on Odia dataset

2. **olmOCR**
   - Purpose: Improved accuracy
   - Architecture: Advanced OCR model
   - Training: Fine-tune from DeepSeek checkpoint

3. **Qwen2.5-VL**
   - Purpose: Production model
   - Architecture: Vision-Language Model
   - Training: Progressive fine-tuning

---

## 📈 Success Criteria

### Phase 1 (Dataset) ✅
- [x] All 47 OHCS characters present
- [x] Minimum 3,000 samples per class
- [x] Balanced class distribution
- [x] Clean train/val/test splits
- [x] Comprehensive documentation

### Phase 2 (Training) 🔄
- [ ] CER <5% on test set
- [ ] WER <10% on word-level tasks
- [ ] >90% validation accuracy
- [ ] All classes >80% accuracy
- [ ] Reproducible training pipeline

### Phase 3 (Deployment) 📅
- [ ] Production-ready API
- [ ] <100ms inference latency
- [ ] Published model on Hugging Face
- [ ] Complete user documentation
- [ ] Real-world deployment

---

## 🚧 Risks & Mitigation

### Data Quality Risks

**Risk**: Synthetic data may not represent real handwriting
- **Mitigation**: Applied realistic handwriting effects, will validate on real test set

**Risk**: Missing character classes may have poor accuracy
- **Mitigation**: Generated 3,000 samples per class, will monitor closely

### Training Risks

**Risk**: Model overfitting on synthetic data
- **Mitigation**: Heavy validation, regularization, dropout

**Risk**: Insufficient computational resources
- **Mitigation**: Use cloud GPUs if needed, optimize batch sizes

### Deployment Risks

**Risk**: Inference too slow for production
- **Mitigation**: Model quantization, TensorRT optimization

**Risk**: Poor generalization to real documents
- **Mitigation**: Extensive real-world testing before deployment

---

## 📝 Deliverables Checklist

### Code
- [x] Data preparation scripts
- [x] Dataset balancing pipeline
- [x] Synthetic data generation
- [ ] Training scripts
- [ ] Evaluation scripts
- [ ] Inference API
- [ ] Web interface

### Documentation
- [x] Project plan
- [x] Dataset reports
- [x] README
- [ ] Training guide
- [ ] API documentation
- [ ] User manual
- [ ] Technical paper

### Models
- [ ] DeepSeek-OCR checkpoint
- [ ] olmOCR checkpoint
- [ ] Qwen2.5-VL checkpoint
- [ ] Published on Hugging Face

### Data
- [x] Annotated dataset
- [x] Train/val/test splits
- [x] Character mapping
- [ ] Evaluation benchmarks

---

## 🔄 Current Status

**Phase**: Phase 2 - Model Training (Week 3)
**Progress**: 33% Complete (2/6 weeks)

### Completed
- ✅ Phase 1: Dataset preparation (100%)
  - 182,152 images prepared
  - All 47 characters covered
  - Balanced and split

### In Progress
- 🔄 Phase 2: Model training
  - Setting up training environment
  - Implementing data loaders
  - Preparing baseline model

### Upcoming
- 📅 DeepSeek-OCR training (Week 3-4)
- 📅 olmOCR fine-tuning (Week 5)
- 📅 Qwen2.5-VL optimization (Week 6)

---

## 📞 Stakeholders & Resources

### Team
- **Project Lead**: [Your Name]
- **Data Engineer**: [Your Name]
- **ML Engineer**: [Your Name]

### Resources
- **Compute**: Local GPU / Cloud (TBD)
- **Storage**: 100GB local + Cloud backup
- **Budget**: Open-source tools (minimal cost)

---

## 📚 References

### Datasets
- Odia Handwritten Character Dataset
- IndicSTR (AI4Bharat)
- Mozhi/IndicNLP Corpus

### Papers
- DeepSeek-OCR Architecture
- olmOCR: Optimized Language Models for OCR
- Qwen2.5-VL: Vision-Language Models

### Tools
- Hugging Face Transformers
- PyTorch Documentation
- OpenCV Tutorials

---

**Last Updated**: January 2026
**Version**: 1.0
**Status**: Phase 1 Complete, Phase 2 In Progress
