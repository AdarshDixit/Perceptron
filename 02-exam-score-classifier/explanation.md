# Demo 02 — Exam-Score Classifier

**Teaching a perceptron to decide pass or fail from a score**

*Pure Python · No libraries · 2 files*

---

## 🎯 What does this demo do?

It trains a perceptron to classify exam scores as "pass" (≥ 50) or "fail" (< 50). This is a step up from Demo 01 because **both classes are positive numbers** — all scores range from 0 to 100. The perceptron can no longer use the sign of the number as a shortcut; it must learn an internal "50-point threshold" entirely from examples.

| Class | Score range | Label |
|---|---|---|
| ✅ Pass | 50 to 100 (e.g. 73.2, 91.0, 50.4) | 1 |
| ❌ Fail | 0 to 50 (e.g. 12.7, 38.9, 3.1) | 0 |

> **Contrast with Demo 01:** In Demo 01 the perceptron only needed to detect the *sign* (positive vs negative). Here, both classes are positive — the only difference is whether the value crosses 50. The perceptron must learn a shifted boundary, which requires the bias parameter.

---

## 📄 perceptron.py — The Brain

This is the same Perceptron class used in all demos (each folder has its own copy). It contains three methods: initialise, predict, train, and accuracy. The explanations below are the same as in Demo 01 — skip ahead to **main.py** if you've already read it.

---

### Step 1 — Initialise with weight, bias, and learning rate

```python
self.weight = random.uniform(-1, 1)
self.bias   = random.uniform(-1, 1) if use_bias else 0.0
self.learning_rate = learning_rate
```

Sets random starting values. **Weight** scales how much the input matters. **Bias** is a free offset that lets the boundary move away from zero — essential for this demo since the true boundary is at 50, not 0.

---

### Step 2 — Predict: step function

```python
return 1 if (self.weight * x + self.bias) > 0 else 0
```

```
output = 1  if  (weight × score + bias) > 0  else  0
```

After training, the weight and bias will be tuned so this formula outputs 1 for scores ≥ 50 and 0 for scores < 50. The decision boundary is at the score where the formula equals zero: `score = −bias ÷ weight`.

---

### Step 3 — Train: correct mistakes epoch by epoch

```python
error = y - pred
if error != 0:
    self.weight += self.learning_rate * error * x
    self.bias   += self.learning_rate * error
```

Each mistake triggers an update to both weight and bias. The key insight here is that **bias is updated independently of the input** — it just slides the boundary left or right without changing the slope. This is how the perceptron can move its decision point from 0 to 50.

---

## 📄 main.py — The Experiment

### Step 1 — Fix the seed and import

```python
import random
from perceptron import Perceptron

random.seed(42)
```

Same pattern as Demo 01. Fixing the seed to 42 ensures every run of this script produces the same data and the same results — making the experiment reproducible.

---

### Step 2 — Generate 200 exam scores and assign pass/fail labels

```python
scores = [random.uniform(0, 100) for _ in range(200)]
labels = [1 if s >= 50 else 0 for s in scores]
```

Generates 200 random decimal scores between 0 and 100, then automatically labels each one:

- Score ≥ 50 → label `1` (pass)
- Score < 50 → label `0` (fail)

The perceptron is *not given this rule*. It must discover from the labelled examples that 50 is the threshold.

> **Key difference from Demo 01:** All 200 scores are positive numbers (0–100). The perceptron cannot rely on sign (positive vs negative) to decide — it must learn the actual value of the threshold.

---

### Step 3 — Split 80% training, 20% test

```python
split = int(0.8 * len(scores))   # 160 training, 40 test
train_x, test_x = scores[:split], scores[split:]
train_y, test_y = labels[:split], labels[split:]
```

The first 160 scores are used for training. The last 40 are held back for testing. No shuffling is needed here because `random.uniform` already produces scores in random order — there's no grouping to break up.

---

### Step 4 — Create, train, and evaluate

```python
p = Perceptron(learning_rate=0.1)
converged_at = p.train(train_x, train_y, epochs=100)
acc = p.accuracy(test_x, test_y)
```

Creates a fresh perceptron with random weight and bias, trains it on the 160 scores, and then checks its accuracy on the 40 test scores.

---

### Step 5 — Print the learned decision boundary

```python
print(f"Decision boundary  : x = {-p.bias / p.weight:.4f}  (should be ~50)")
```

The formula `−bias ÷ weight` calculates the exact score at which the perceptron is "on the fence". If training succeeded, this number should be close to 50. It won't be exactly 50 — small deviations are normal because the boundary just needs to sit in the gap between passing and failing scores, not on the exact threshold value.

> **🔑 Why isn't the boundary exactly 50?**
>
> The perceptron learns the *minimum boundary that separates all training examples*, not the midpoint. As long as no training score is exactly 50 (which is essentially impossible with `random.uniform`), any boundary between ~49.9 and ~50.1 would work equally well.

---

## ▶ What you'll see when you run it

Run with: `python3 main.py`

```
=== Exam-Score Classifier ===
Training samples   : 160
Test samples       : 40
Passing threshold  : 50
Converged at epoch : 22
Final weight       : 1.0020
Final bias         : -49.0473
Decision boundary  : x = 48.9512  (should be ~50)
Test accuracy      : 100.0%
```

**22 epochs — harder than Demo 01** — Demo 01 converged in 2 epochs; this one takes 22. That's because the boundary (50) is in the middle of the data range, not at a natural edge. Both weight and bias need to be tuned together, which takes more corrections.

**Boundary learned at x ≈ 48.95** — Very close to the true threshold of 50. The small gap is fine — no training example lies exactly at 50, so this boundary correctly separates all examples.

**100% test accuracy** — All 40 unseen test scores were classified correctly. The perceptron successfully generalised the rule "score ≥ 50 = pass".
