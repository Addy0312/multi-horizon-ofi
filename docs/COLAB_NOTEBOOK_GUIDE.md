# Google Colab Notebook Guide - All 10 Models with Parallelization

## Overview

The updated notebook **`notebooks/03_train_all_models.ipynb`** now includes:

- **All 10 models** (not just 6)
- **Parallelized training** for faster execution and better resource utilization
- **Optimized memory management** for GPU/CPU
- **Complete model weight persistence** for all models

## 🎯 All 10 Models Included

### Linear Models (3)
1. **OLS** - Ordinary Least Squares (Cont et al. 2014)
2. **Ridge Regression** - L2 regularization (Xu et al. 2019)
3. **Lasso Regression** - L1 regularization (Xu et al. 2019)

### Deep Learning Models (7)
4. **MLP** - Feedforward neural network (Kolm et al. 2023)
5. **LSTM** - Long Short-Term Memory (Kolm et al. 2023)
6. **CNN** - Convolutional neural network (Kolm et al. 2023)
7. **DeepLOB** - Inception blocks + LSTM (Zhang et al. 2019)
8. **Seq2Seq + Attention** - Encoder-decoder with Bahdanau attention (Zhang & Zohren 2021)
9. **Transformer** - Multi-head self-attention (Wallbridge 2020)
10. **Seq2Seq Temporal Attention** - Custom model with multi-scale encoder + horizon-aware attention

## ⚡ Parallelization Strategy

### Linear Models (Threads)
- **Execution**: Parallel using `ThreadPoolExecutor` (max 3 workers)
- **Why threads**: I/O bound operations (sklearn fitting is fast, statsmodels uses threading)
- **Time savings**: ~3x speedup (OLS, Ridge, Lasso simultaneously)

### Deep Learning Models (Sequential with GPU Optimization)
- **Execution**: Sequential training with GPU memory management
- **Memory optimization**:
  - Aggressive `torch.cuda.empty_cache()` every 5 batches
  - Larger batch sizes (512-1024) to maximize GPU utilization
  - Gradient checkpointing for transformer
  - Reduced validation frequency
- **Why sequential**: 
  - Prevents GPU OOM errors (models trained one at a time)
  - GPU memory scales linearly with number of simultaneous models
  - Colab GPU memory: ~15GB, vs ~2GB per large model
- **Data loading**: `num_workers=0` (avoids multiprocessing overhead in notebooks)

### Feature Computation
- Pre-computed once and reused across all models
- Features normalized once (fit on train, transform on val/test)
- No redundant computation

## 📊 Expected Performance on Colab

### Time Estimates (with optimized settings)
- **Linear models**: ~1-2 minutes (parallel)
- **MLP**: ~3-4 minutes
- **LSTM/CNN/DeepLOB**: ~4-6 minutes each
- **Seq2Seq**: ~6-8 minutes
- **Transformer**: ~7-10 minutes
- **Temporal Attention**: ~8-12 minutes
- **Total**: ~35-50 minutes (with GPU)
  
Without GPU: ~3-5x slower

### GPU Utilization Improvements
| Setting | Improvement |
|---------|-------------|
| Batch size 256 → 512 | 2x throughput |
| Memory cache clearing | Prevents OOM |
| Early stopping | Prevents wasted epochs |
| 30 epochs → 25 epochs | 17% faster |

## 📁 Output Structure

```
model_weights/
├── ols_models.pkl              # Linear models (3 models, one per horizon)
├── ridge_models.pkl
├── lasso_models.pkl
├── mlp_weights.pt              # Deep models (PyTorch state dicts)
├── lstm_weights.pt
├── cnn_weights.pt
├── deeplob_weights.pt
├── seq2seq_weights.pt
├── transformer_weights.pt
└── temporal_attention_weights.pt

results/
├── results_all_10_models_YYYYMMDD_HHMMSS.json  # Detailed metrics
└── performance_all_10_models.png                 # 4-subplot visualization
```

## 🚀 Using in Google Colab

### Step 1: Upload Data
```
Upload LOBSTER CSV files to:
- data/raw/AMZN_2012-06-21_xxxxx_message_10.csv
- data/raw/AMZN_2012-06-21_xxxxx_orderbook_10.csv
```

### Step 2: Run Cells in Order
1. **Cell 1-3**: Setup dependencies, check GPU
2. **Cell 4-5**: Mount Drive, create directories
3. **Cell 6**: Load and preprocess data (~30s)
4. **Cell 7**: Feature engineering (~60s)
5. **Cell 8**: Train/val/test split
6. **Cell 9**: Define all 10 model architectures
7. **Cell 10**: Utility functions (metrics, etc)
8. **Cell 11**: Train linear models (parallel, ~2 min)
9. **Cell 12**: Train deep models (sequential, ~40-45 min)
10. **Cell 13**: Results summary table
11. **Cell 14**: Visualizations (4 subplots)
12. **Cell 15**: Model loading examples
13. **Cell 16**: Save metadata

### Step 3: Download Results
- Right-click `model_weights/` → Download
- Right-click `results/` → Download

## 📈 Key Improvements

| Feature | Before | After |
|---------|--------|-------|
| Models | 6 | **10** |
| Linear parallelization | ❌ | **✅ (3x speedup)** |
| GPU batch size | 256 | **512** |
| Memory management | Basic | **Aggressive caching** |
| Epochs | 30 | **25** (early stopping active) |
| Total time (Colab GPU) | 60-80 min | **35-50 min** |

## 🔧 Customization

### Increase Speed (Reduce Quality)
```python
# In model_configs:
'epochs': 15,           # was 25
'batch_size': 1024,     # was 512
max_patience = 3        # was 5 (earlier stopping)
```

### Decrease Speed (Improve Quality)
```python
# In model_configs:
'epochs': 50,           # was 25
'batch_size': 256,      # was 512
max_patience = 10       # was 5 (more patience)
```

### Add GPU Parallelism (Advanced)
Instead of sequential training, use `torch.nn.DataParallel`:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Multi-GPU
```

## 📝 Architecture Details

### All Model Outputs
- **Shape**: (batch_size, 4 horizons, 3 classes)
- **Classes**: 0=DOWN, 1=STATIONARY, 2=UP
- **Evaluation**: Multi-horizon accuracy, F1-macro, precision, recall

### Feature Engineering
- **Total features**: 38
  - OFI single-level (5)
  - OFI cumulative (5)
  - Microstructure (8): mid_price, spread, volume_imbalance, etc.
  - Raw LOB (20): 5 levels × 4 columns (ask_price, ask_size, bid_price, bid_size)
- **Horizons**: k=[10, 20, 50, 100] events

### Training Strategy
- **Split**: 60% train, 20% val, 20% test (temporal, no shuffle)
- **Normalization**: Z-score fitted on train only
- **Loss**: CrossEntropyLoss with per-horizon heads
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping**: patience=5

## ✅ Validation Checklist

After running the notebook, verify:

- [ ] All 10 models trained (no errors in output)
- [ ] `model_weights/` has 10 files (3 pkl + 7 pt)
- [ ] `results/` has JSON and PNG files
- [ ] Total time < 60 minutes on Colab GPU
- [ ] Accuracy h10 > 35% (deep models)
- [ ] R² not all negative (linear models)
- [ ] Visualizations show 4 subplots with smooth curves

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size in model_configs:
'batch_size': 256,  # from 512

# Or reduce num features (subsample horizons during dev)
```

### GPU Not Available
```python
# Check:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# If CPU, training will take 3-5x longer
```

### Slow Training
```python
# Check GPU utilization:
!nvidia-smi

# If low (<50%), increase batch_size
# If high (>95%), you're good

# Alternatively, reduce epochs to 15-20 for quick test
```

### Model Not Converging (Accuracy ~33%)
```python
# This is baseline (random 3-class). Check:
# 1. Features computed correctly (check shapes)
# 2. Labels not all one class
# 3. Learning rate reasonable (1e-3 is default)
# 4. Sufficient training data
```

## 📚 References

1. **Cont et al. 2014** - Order Flow Imbalance
2. **Xu et al. 2019** - Multi-level OFI features  
3. **Kolm et al. 2023** - Deep LOB forecasting
4. **Zhang et al. 2019** - DeepLOB architecture
5. **Zhang & Zohren 2021** - Temporal attention for time series
6. **Wallbridge 2020** - Transformers for LOB

## 📞 Support

If models fail:
1. Check data is loaded correctly (shapes printed)
2. Verify feature computation (no NaNs)
3. Check device is set correctly
4. Look at error traceback for specific issue
5. Try reducing batch_size or epochs first

---

**Last Updated**: 2026-02-09  
**Notebook Location**: `notebooks/03_train_all_models.ipynb`
