# Demo 03 — Bias Demonstration

**What happens when you remove the perceptron's ability to shift its decision point**

*Pure Python · No libraries · 2 files*

---

## 🎯 What does this demo do?

It runs the **same dataset through two perceptrons** — one with a bias and one without. The goal is to show that when the correct decision boundary is not at zero, removing the bias makes it impossible to classify data correctly.

| | With Bias | Without Bias |
|---|---|---|
| Boundary | Slides to wherever the data requires (x ≈ 50) | Permanently locked at x = 0 |
| Result | Converges quickly, 100% accuracy | Never converges, ~50% accuracy |

> **Analogy:** Imagine a sliding door that is welded shut at the centre. You can push harder (change the weight) but it will never open past the midpoint. Bias is the mechanism that lets the door slide to the right position.

---

## 📄 perceptron.py — The Brain

### Step 1 — The use_bias flag: the key feature of this file

```python
def __init__(self, learning_rate=0.1, use_bias=True):
    self.weight = random.uniform(-1, 1)
    self.bias   = random.uniform(-1, 1) if use_bias else 0.0
    self.use_bias = use_bias
```

The `use_bias` parameter is what makes this demo possible. When `use_bias=True` (default), the bias is initialised randomly and updated during training. When `use_bias=False`, the bias is permanently set to **0.0** and never changes — it is as if it doesn't exist.

---

### Step 2 — Why a fixed bias of 0 is a problem

```
Decision boundary = −bias ÷ weight = −0 ÷ weight = 0
```

When bias is 0, the decision boundary formula always gives **zero**, regardless of what the weight is. The boundary is stuck at x = 0 forever.

This means the perceptron can only ask: *"is the input positive or negative?"* — it cannot ask *"is the input above 50?"*

> **The trap:** Since all exam scores (0–100) are positive numbers, `weight × score` always has the same sign as `weight` — regardless of the actual score. The perceptron can only predict the same class for every input. It is fundamentally unable to distinguish a score of 30 from a score of 70.

---

### Step 3 — The bias update is guarded

```python
if error != 0:
    self.weight += self.learning_rate * error * x
    if self.use_bias:           # ← only update bias when it's enabled
        self.bias += self.learning_rate * error
```

Every time a mistake is made, the weight is updated regardless. The bias is only updated when `use_bias=True`. When `use_bias=False`, the bias stays at 0.0 — the guard here makes sure of that.

---

## 📄 main.py — The Experiment

### Step 1 — Generate the exam score dataset

```python
random.seed(42)
scores = [random.uniform(0, 100) for _ in range(200)]
labels = [1 if s >= 50 else 0 for s in scores]
```

Creates 200 exam scores between 0 and 100, labelled 1 for pass (≥ 50) and 0 for fail. All values are positive — this is the critical property that makes the no-bias case fail. Then splits 80% for training (160 samples) and 20% for testing (40 samples).

---

### Step 2 — Re-seed before each perceptron for a fair comparison

```python
# Run 1
random.seed(99)
p_with = Perceptron(learning_rate=0.1, use_bias=True)

# Run 2
random.seed(99)
p_no = Perceptron(learning_rate=0.1, use_bias=False)
```

Both perceptrons are created with `random.seed(99)` reset before each. This ensures they both start with the **same initial weight value**. Without this, one perceptron might do better just because it got a luckier random starting point — not because of bias.

> **Why does this matter?** Scientific experiments need a control: the only thing that should differ between the two runs is the single variable being tested (bias on vs off). Identical starting weights make the comparison fair.

---

### Step 3 — Run 1: with bias (use_bias=True)

```python
p_with = Perceptron(learning_rate=0.1, use_bias=True)
ep_with = p_with.train(train_x, train_y, epochs=100)
acc_with = p_with.accuracy(test_x, test_y)
boundary_with = -p_with.bias / p_with.weight
```

The perceptron is free to adjust *both* weight and bias on every mistake. Over a few epochs it discovers a weight and bias combination that places the decision boundary at approximately x = 50. This perfectly separates the two classes.

---

### Step 4 — Run 2: without bias (use_bias=False)

```python
p_no = Perceptron(learning_rate=0.1, use_bias=False)
ep_no = p_no.train(train_x, train_y, epochs=100)
acc_no = p_no.accuracy(test_x, test_y)
```

Only the weight can change. But when all inputs are positive, `weight × x` always has the same sign as `weight`. If weight is positive, every prediction is 1 (pass). If negative, every prediction is 0 (fail). The perceptron oscillates between these two extremes, never finding a valid solution. After 100 epochs it still gets roughly half wrong.

> **🔑 The mathematical reason it fails**
>
> Without bias, `predict(x) = 1 if weight × x > 0 else 0`. For positive x, this simplifies to: `predict(x) = 1 if weight > 0 else 0`. The value of *x itself has no effect on the decision*. A score of 10 and a score of 90 get the same label. The perceptron is blind to the actual value.

---

## ▶ What you'll see when you run it

Run with: `python3 main.py`

```
=== Bias Demonstration ===
Dataset            : exam scores 0–100, pass if >= 50
Training samples   : 160
Test samples       : 40

--- With Bias ---
Converged at epoch : 3
Final weight       : 0.1159
Final bias         : -5.7998
Decision boundary  : x = 50.0578  (should be ~50)
Test accuracy      : 100.0%

--- Without Bias ---
Converged at epoch : did not converge within 100 epochs
Final weight       : 1.0816
Bias               : 0.0  (fixed — cannot move)
Decision boundary  : x = 0.0  (locked at origin)
Test accuracy      : 42.5%

Explanation:
  All inputs are positive (0–100). The correct boundary is at x=50.
  Without bias, the boundary is always x=0 (i.e. weight * x > 0 only
  depends on the sign of weight, not the value of x).
  The perceptron ends up classifying everything as one class — ~50% accuracy.
```

**With bias: converged in 3 epochs, boundary at 50.06, 100% accuracy** — Bias gave the perceptron the freedom to slide the boundary to wherever the data required. It found the threshold in just 3 passes.

**Without bias: never converged, boundary at 0, only 42.5% accuracy** — Without bias the boundary cannot move from zero. Since all scores are positive, the perceptron ends up classifying everything as "pass" (since weight becomes positive from all the failed corrections), which is only correct for the ~50% of scores that actually are ≥ 50.
