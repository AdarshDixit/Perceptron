# Demo 04 — Normalisation Demo

**How rescaling inputs makes learning 256× faster**

*Pure Python · No libraries · 2 files*

---

## 🎯 What does this demo do?

It trains the **same perceptron twice on the same data**, with one difference: the second run rescales all input values to fit between 0 and 1 before training. The result is a dramatic speedup — the normalised version converges in **3 epochs** while the raw version takes **768 epochs**, a 256× difference.

| | Without Normalisation | With Normalisation |
|---|---|---|
| Input range | 0 – 10,000 | 0 – 1 |
| Max weight update per step | ±1,000 | ±0.1 |
| Epochs to converge | 768 | 3 |
| Test accuracy | 100% | 100% |

> **Analogy:** Imagine tuning a radio dial. If one small turn moves the frequency by 10,000 MHz you'll overshoot constantly. If one turn moves it by 1 MHz you can land precisely on the right station. Normalisation turns the "coarse" dial into a "fine" one.

---

## 📄 perceptron.py — The Brain

### Step 1 — The update rule: why input magnitude matters

```python
self.weight += self.learning_rate * error * x
```

This single line is the reason normalisation matters so much. The weight update size is **learning_rate × error × x**. Let's see what this means for each run:

| Run | Calculation | Update size |
|---|---|---|
| Without normalisation | 0.1 × 1 × 10,000 | **±1,000 per step** |
| With normalisation | 0.1 × 1 × 1.0 | **±0.1 per step** |

A weight update of 1,000 on one example can be completely cancelled by the next example, causing the perceptron to oscillate back and forth for hundreds of epochs before it happens to land in the right place. An update of 0.1 takes small, stable steps toward the right answer.

---

## 📄 main.py — The Experiment

### Step 1 — Generate deliberately large-range data

```python
random.seed(42)
scores = [random.uniform(0, 10000) for _ in range(200)]
labels = [1 if s >= 5000 else 0 for s in scores]
```

Creates 200 exam scores from 0 to 10,000 — a deliberately exaggerated range. Scores ≥ 5,000 are labelled "pass". The large range is chosen to make the problem with raw training visible. In real life, large feature ranges arise naturally (income in dollars, distances in metres, prices in various currencies).

> **Why 0–10,000?** A range of 0–100 (like Demo 02) converges in 22 epochs even without normalisation, so the difference isn't dramatic. The 100× larger range here makes the oscillation problem obvious and the speedup striking.

---

### Step 2 — Run 1: train on raw values

```python
random.seed(99)
p_raw = Perceptron(learning_rate=0.1)
ep_raw = p_raw.train(train_x, train_y, epochs=1000)
```

Trains directly on the raw scores (0–10,000). The epoch limit is 1,000 (not 100 like other demos) because we expect this run to take much longer. The re-seed ensures a fair comparison with Run 2.

> **What's happening internally:** Each large input (e.g. a score of 9,800) causes a weight update of up to ±980. The weight swings wildly between large positive and negative values. The perceptron corrects one mistake and immediately overcorrects in the process, causing new mistakes on the other side. This continues for hundreds of epochs before the updates accidentally stabilise.

---

### Step 3 — Compute normalisation bounds from training data only

```python
x_min = min(train_x)
x_max = max(train_x)
```

Finds the smallest and largest score in the **training set only**. These two numbers define the rescaling range.

> **🔑 Why training data only — not test data?**
>
> This is called avoiding **"data leakage"**. In the real world, you train a model *before* you deploy it. At training time, you don't have access to future data. If you compute statistics (like min/max) from the test set, you're cheating — you're using information that wouldn't be available in a real scenario. The correct approach: fit the scale on training data, then apply it to everything else.

---

### Step 4 — Define and apply the normalisation function

```python
def normalise(x):
    return (x - x_min) / (x_max - x_min)

norm_train_x = [normalise(x) for x in train_x]
norm_test_x  = [normalise(x) for x in test_x]
```

This is the **min-max normalisation** formula. It squashes any value into the range [0, 1]:

```
normalised = (value − minimum) ÷ (maximum − minimum)
```

Tracing through two examples with x_min ≈ 6 and x_max ≈ 9,975:

- Score = 6 (minimum): (6 − 6) ÷ (9,975 − 6) = **0.0**
- Score = 9,975 (maximum): (9,975 − 6) ÷ (9,975 − 6) = **1.0**
- Score = 5,000 (threshold): (5,000 − 6) ÷ (9,975 − 6) ≈ **0.501**

The same formula (with the same x_min and x_max from training) is applied to the test data. This keeps the two datasets on the same scale.

---

### Step 5 — Run 2: train on normalised values

```python
random.seed(99)
p_norm = Perceptron(learning_rate=0.1)
ep_norm = p_norm.train(norm_train_x, train_y, epochs=1000)
```

Same structure as Run 1 — same seed (fair comparison), same learning rate. The only difference is the input values are now in [0, 1] instead of [0, 10,000]. Maximum weight update per step is now 0.1 × 1 × 1 = **0.1** instead of up to 1,000. The perceptron takes tiny, stable steps and converges in 3 epochs.

---

### Step 6 — Print the side-by-side comparison

```python
if ep_raw and ep_norm:
    print(f"Result: normalisation reduced convergence from epoch {ep_raw} to {ep_norm} "
          f"({ep_raw - ep_norm} fewer epochs, {ep_raw / ep_norm:.1f}x faster).")
```

Prints a summary comparing epochs to convergence. Handles all four possible outcomes (both converge, only raw converges, only normalised converges, neither converges) to avoid a divide-by-zero crash.

---

## ▶ What you'll see when you run it

Run with: `python3 main.py`

```
=== Normalisation Demo ===
Dataset            : exam scores 0–10000, pass if >= 5000
Training samples   : 160
Test samples       : 40
Raw value range    : [5.7, 9975.4]

--- Without Normalisation ---
Converged at epoch : 768
Final weight       : 0.6376
Final bias         : -3191.3998
Test accuracy      : 100.0%

--- With Min-Max Normalisation (scaled to [0, 1]) ---
Scale applied      : (x - 6) / (9975 - 6)
Converged at epoch : 3
Final weight       : 0.4102
Final bias         : -0.1998
Test accuracy      : 100.0%

Result: normalisation reduced convergence from epoch 768 to 3 (765 fewer epochs, 256.0x faster).
```

**768 vs 3 epochs — the same problem, solved 256× faster** — Both perceptrons reach 100% accuracy in the end. The data is the same, the learning rule is the same, the learning rate is the same. The *only* difference is the scale of the inputs. This is why normalisation is a standard first step in almost every machine learning pipeline.

**The weights look very different — but they encode the same boundary:**

- Raw: weight = 0.6376, bias = −3,191.4 → boundary at −(−3,191.4) ÷ 0.6376 ≈ **5,005** ✓
- Normalised: weight = 0.4102, bias = −0.1998 → boundary at ≈ **0.487** in [0,1] space (maps back to ≈ 4,862 in original scale) ✓

Both have found a boundary near 5,000. The numbers look different because one is in the original scale and the other is in the normalised scale — but the decision is the same.
