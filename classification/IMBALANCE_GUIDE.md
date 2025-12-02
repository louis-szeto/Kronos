# Class Imbalance Handling Guide

When you have imbalanced data (e.g., 1138 samples of class 0 vs. 131 samples of class 1), the model can become biased toward the majority class. Here are **four strategies** to handle this, each with pros and cons:

---

## Strategy 1: Class Weights (Recommended for most cases)

**What it does:** Assigns higher loss penalties to minority class predictions during training.

**How to use:**
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance class_weights \
    --num_classes 2 \
    ...
```

**Pros:**
- Uses all your data (no samples discarded)
- No artificial duplication
- Mathematically sound approach
- Works well with gradient descent
- Fast training (same number of samples)

**Cons:**
- May not work as well if imbalance is extreme (>100:1)
- Doesn't increase minority class exposure during training

**When to use:** 
- Most imbalanced scenarios (your case: ~8.7:1 ratio)
- When you want to preserve all training data
- When training time is a concern

**How it works:**
```python
# For your data: Class 0: 1138, Class 1: 131
# Weights calculated as: total / (num_classes * class_count)
# Class 0 weight: 1269 / (2 * 1138) ≈ 0.56
# Class 1 weight: 1269 / (2 * 131) ≈ 4.84
# The model is penalized 4.84/0.56 = 8.6x more for misclassifying class 1
```

---

## Strategy 2: Oversampling

**What it does:** Duplicates minority class samples (with replacement) until balanced.

**How to use:**
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance oversample \
    --oversample_ratio 1.0 \
    --num_classes 2 \
    ...
```

**Parameters:**
- `--oversample_ratio 1.0`: Make minority class equal to majority (full balance)
- `--oversample_ratio 0.5`: Make minority class 50% of majority (partial balance)
- `--oversample_ratio 0.3`: Make minority class 30% of majority

**Pros:**
- Increases minority class exposure during training
- Model sees minority patterns more often
- Often improves recall for minority class
- Simple and intuitive

**Cons:**
- Creates duplicate samples (overfitting risk)
- Longer training time (~8x more iterations in your case)
- May learn to memorize minority samples
- Larger effective dataset size

**When to use:**
- When minority class has diverse patterns you want to emphasize
- When you have enough compute for longer training
- When recall on minority class is critical

**Example for your data:**
```
Original: Class 0: 1138, Class 1: 131
After oversample (ratio=1.0): Class 0: 1138, Class 1: 1138
Training samples: 2276 (vs original 1269)
```

---

## Strategy 3: Undersampling

**What it does:** Randomly removes majority class samples until balanced.

**How to use:**
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance undersample \
    --num_classes 2 \
    ...
```

**Pros:**
- Creates perfectly balanced dataset
- Fast training (fewer samples)
- No risk of overfitting to duplicates
- Simple approach

**Cons:**
- **Discards 1007 samples of class 0** (88% of majority class!)
- Loses valuable information
- May not learn majority class patterns well
- Only use if you have abundance of data

**When to use:**
- When you have massive amounts of majority class data
- When training time is critical
- When majority class is very homogeneous

**Example for your data:**
```
Original: Class 0: 1138, Class 1: 131
After undersample: Class 0: 131, Class 1: 131
Training samples: 262 (vs original 1269)
⚠️ WARNING: You lose 1007 samples!
```

**⚠️ NOT RECOMMENDED for your case** - you'd lose too much data.

---

## Strategy 4: No Balancing

**What it does:** Trains on data as-is without any balancing.

**How to use:**
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance none \
    --num_classes 2 \
    ...
```

**Pros:**
- Uses all data naturally
- Fastest training
- No artificial manipulation

**Cons:**
- Model will be biased toward majority class
- May predict class 0 for almost everything
- Poor performance on class 1
- Accuracy can be misleading (high accuracy by just predicting majority)

**When to use:**
- When classes are naturally balanced
- When you want a baseline to compare against
- When majority class is truly more important

**Not recommended for your imbalanced data.**

---

## Comparison Table

| Strategy | Training Samples | Training Time | Data Loss | Risk | Best For |
|----------|-----------------|---------------|-----------|------|----------|
| **Class Weights** | 1269 (100%) | 1x | None | Low | ⭐ Most cases |
| **Oversample** | 2276 (179%) | ~1.8x | None | Medium (overfitting) | When minority recall is critical |
| **Undersample** | 262 (21%) | ~0.2x | High (88%!) | Medium (underfitting) | ⚠️ Rarely recommended |
| **None** | 1269 (100%) | 1x | None | High (bias) | ⚠️ Only for balanced data |

---

## Recommendation for Your Data

Given your distribution (Class 0: 1138, Class 1: 131, ratio ~8.7:1), here's what I recommend:

### Primary Recommendation: **Class Weights**
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance class_weights \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --fp16
```

**Why:** Preserves all data, mathematically sound, works well for your imbalance ratio.

### Alternative: **Oversample** (if class weights don't work well)
```bash
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance oversample \
    --oversample_ratio 0.5 \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --fp16
```

**Why:** `oversample_ratio 0.5` means class 1 will have ~569 samples (50% of class 0), giving you 1707 total samples instead of 2276. This is a middle ground.

---

## Monitoring & Evaluation

After training, check these metrics to see if your balancing strategy worked:

1. **Confusion Matrix**: Should show reasonable predictions for both classes
2. **Precision/Recall for Class 1**: Should be meaningful (not near 0)
3. **F1 Score**: Balances precision and recall
4. **Per-Class Accuracy**: Check accuracy for each class separately

The training script automatically reports all these metrics during validation!

---

## Example Workflow

```bash
# 1. Start with class weights (recommended)
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance class_weights \
    --num_classes 2 \
    --output_dir ./checkpoints_weighted \
    --fp16

# 2. If results aren't good, try oversampling
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance oversample \
    --oversample_ratio 0.5 \
    --num_classes 2 \
    --output_dir ./checkpoints_oversampled \
    --fp16

# 3. Compare validation metrics and choose the best model
python kronos_inference.py analyze ./checkpoints_weighted/best_model
python kronos_inference.py analyze ./checkpoints_oversampled/best_model
```

---

## Advanced: Combining Strategies

You can also combine approaches in your training pipeline:

```bash
# Use oversampling with partial ratio during pretraining
python kronos_pretrain.py \
    --data_dir ./data \
    --class_balance oversample \
    --oversample_ratio 0.5 \
    --num_classes 2 \
    --output_dir ./pretrain_checkpoints

# Then use class weights during fine-tuning for refinement
python kronos_finetune.py \
    --data_dir ./data \
    --pretrained_checkpoint ./pretrain_checkpoints/best_model \
    --class_balance class_weights \
    --num_classes 2 \
    --output_dir ./finetune_checkpoints
```

---

## Quick Decision Guide

**Answer these questions:**

1. **Is your minority class diverse with many distinct patterns?**
   - YES → Try `oversample` 
   - NO → Use `class_weights`

2. **Do you care more about catching minority class (high recall)?**
   - YES → Try `oversample` with ratio 0.7-1.0
   - NO → Use `class_weights`

3. **Is training time critical?**
   - YES → Use `class_weights` (fastest)
   - NO → Can try `oversample`

4. **Is your imbalance extreme (>50:1)?**
   - YES → Combine `oversample` + `class_weights`
   - NO → `class_weights` should work fine

**For most cases including yours: Start with `class_weights`!**