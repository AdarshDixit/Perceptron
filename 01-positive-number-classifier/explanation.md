# Demo 01 — Positive-Number Classifier

**Teaching a perceptron to tell positive numbers from negative ones**

*Pure Python · No libraries · 2 files*

---

## 🎯 What does this demo do?

It builds a tiny machine learning model from scratch that learns — just from examples — that numbers above zero are "positive" and numbers below zero are "negative". No rules are written by hand. The model figures it out on its own by looking at data and correcting its mistakes.

> **Real-world analogy:** Imagine teaching a child by showing them cards labelled "hot" or "cold". After seeing enough examples, they learn the rule themselves. The perceptron does the same thing with numbers.

---

## 📄 perceptron.py — The Brain

This file defines the **Perceptron class** — the core learning machine. Think of it as a blueprint for creating a brain cell that can be trained to make yes/no decisions. Every demo folder has its own copy of this file so each demo is completely self-contained.

---

### Step 1 — Import the random library

```python
import random
```

Brings in Python's built-in `random` module. We only need it in this file to generate random starting values for the weight and bias (explained next).

> **Why random?** The perceptron must start somewhere. If it always started at zero it might get stuck. A random start gives it a neutral, unbiased beginning — like rolling dice before a game to decide who goes first.

---

### Step 2 — Initialise the perceptron

```python
def __init__(self, learning_rate=0.1, use_bias=True):
    self.weight = random.uniform(-1, 1)
    self.bias   = random.uniform(-1, 1) if use_bias else 0.0
    self.use_bias = use_bias
    self.learning_rate = learning_rate
```

This runs automatically when you create a new perceptron (e.g. `p = Perceptron()`). It sets up three key things:

- **weight** — A random number between -1 and 1. The weight decides how important the input number is. During training, this value is adjusted until the perceptron gets everything right.
- **bias** — Another random starting number. The bias is like a dial that shifts the decision point left or right. Without it, the dividing line is permanently stuck at zero.
- **learning_rate** — How big a step to take when correcting a mistake. `0.1` means "make small corrections". A very large rate (e.g. 10) would overshoot and oscillate wildly.

> **🔑 Key concept: weight and bias**
>
> The perceptron's decision formula is: **weight × input + bias**. If this result is greater than zero, it says "yes" (1). Otherwise it says "no" (0). Training means finding the right weight and bias values so this formula always gives the correct answer.

---

### Step 3 — Make a prediction

```python
def predict(self, x):
    return 1 if (self.weight * x + self.bias) > 0 else 0
```

Given an input number `x`, this computes **weight × x + bias** and asks: is the result positive?

- If yes → returns **1** (the perceptron says "this is positive")
- If no  → returns **0** (the perceptron says "this is negative")

```
output = 1  if  (weight × x + bias) > 0  else  0
```

> **Why this formula?** The point where `weight × x + bias = 0` is the *decision boundary* — the exact value of x that the perceptron is "on the fence" about. You can find it by solving: `x = −bias ÷ weight`.

---

### Step 4 — Train: learn from mistakes

```python
def train(self, data, labels, epochs=100):
    for epoch in range(epochs):
        errors = 0
        for x, y in zip(data, labels):
            pred  = self.predict(x)
            error = y - pred
            if error != 0:
                self.weight += self.learning_rate * error * x
                if self.use_bias:
                    self.bias += self.learning_rate * error
                errors += 1
        if errors == 0:
            return epoch + 1
    return None
```

This is the heart of the perceptron. Here's what happens, step by step:

1. **An epoch is one full pass through all training examples.** The outer loop repeats this up to 100 times.
2. **For each example**, the perceptron makes a prediction and checks if it was right.
3. **error = label − prediction.** This gives three possible values:
   - `error = 0` → correct, do nothing
   - `error = +1` → predicted 0, should be 1 (missed a positive)
   - `error = −1` → predicted 1, should be 0 (false alarm)
4. **Update rule:** `weight += learning_rate × error × x`. This nudges the weight slightly in the direction that would have given the right answer. The bias gets a similar nudge.
5. **Early stopping:** If an entire epoch passes with zero mistakes, the perceptron has fully learned the data. There's no point continuing — return the epoch number.
6. **Returns None** if the limit is reached without a perfect pass.

> **Analogy:** Imagine trying to balance a seesaw. Each time it tips the wrong way, you add a small weight on the other side. Eventually it balances. The training loop is that process of repeated small corrections.

---

### Step 5 — Measure accuracy

```python
def accuracy(self, data, labels):
    correct = sum(1 for x, y in zip(data, labels) if self.predict(x) == y)
    return correct / len(data)
```

Counts how many predictions were correct and divides by the total. Returns a number between 0 and 1 — multiply by 100 to get a percentage. Called on the *test set* (data the perceptron has never seen) to get an honest measure of performance.

---

## 📄 main.py — The Experiment

### Step 1 — Set up imports

```python
import random
from perceptron import Perceptron
```

Loads two things: Python's `random` module (to generate data) and the `Perceptron` class from the file in the same folder.

---

### Step 2 — Fix the random seed

```python
random.seed(42)
```

Sets a fixed starting point for Python's random number generator. The number 42 is arbitrary — any number works. The effect is that *every time you run the script you get identical data and results*.

> **Why?** Without this, each run generates different random data, making it hard to compare or debug. A fixed seed makes experiments reproducible — a fundamental practice in machine learning.

---

### Step 3 — Generate the dataset

```python
positives = [random.uniform(0, 50)   for _ in range(100)]
negatives = [random.uniform(-500, 0) for _ in range(100)]
values = positives + negatives
labels = [1] * 100 + [0] * 100
```

Creates 200 numbers in total:

- 100 **positive numbers** randomly picked from the range 0 to 50. Each is labelled `1` (True / pass).
- 100 **negative numbers** randomly picked from −500 to 0. Each is labelled `0` (False / fail).

The two classes are on completely opposite sides of zero, making them easy to separate — this is called *linearly separable* data.

---

### Step 4 — Shuffle the data

```python
combined = list(zip(values, labels))
random.shuffle(combined)
values, labels = [x for x, _ in combined], [y for _, y in combined]
```

Pairs each number with its label, shuffles them randomly, then unpacks them back into separate lists. This mixes positive and negative examples together so the training loop doesn't see 100 positives first and then 100 negatives.

> **Why shuffle?** If training sees all of one class first, the weight updates are heavily biased in one direction before the other class is seen. Mixed order gives a more balanced learning signal each epoch.

---

### Step 5 — Split into training and test sets

```python
split = int(0.1 * len(values))   # 10% of 200 = 20 samples
train_x, test_x = values[:split], values[split:]
train_y, test_y = labels[:split], labels[split:]
```

Divides the data into two groups:

- **Training set** — first 20 samples (10%). The perceptron learns from these.
- **Test set** — remaining 180 samples (90%). These are kept hidden during training and only used at the end to check how well the perceptron generalises.

> **Note:** This is currently using a 10/90 split. A common convention is 80/20 (train/test). With only 20 training samples the perceptron has less data to learn from, but since this dataset is very simple (clear separation at x=0) it still learns quickly.

---

### Step 6 — Create and train the perceptron

```python
p = Perceptron(learning_rate=0.1)
converged_at = p.train(train_x, train_y, epochs=100)
```

Creates a new perceptron with a learning rate of 0.1, then runs training for up to 100 passes over the 20 training samples. `converged_at` will be the epoch number when the perceptron made zero mistakes, or `None` if it never reached that point.

---

### Step 7 — Evaluate on unseen test data

```python
acc = p.accuracy(test_x, test_y)
```

Runs the trained perceptron on the 180 test samples it has never seen. Returns the fraction of correct answers.

> **Why test on unseen data?** A student who memorises answers does well on practice questions but fails the real exam. Testing on new data checks whether the perceptron learned the *rule* (positive vs negative) rather than just memorising the training examples.

---

### Step 8 — Print results

```python
print(f"Decision boundary  : x = {-p.bias / p.weight:.4f}")
```

The decision boundary is the exact input value where the perceptron is perfectly "on the fence" — anything above it gets labelled 1, anything below gets labelled 0. It is derived by solving `weight × x + bias = 0`, giving `x = −bias ÷ weight`. For this dataset the learned boundary should be very close to **0**, which is the true dividing line between positive and negative numbers.

---

## ▶ What you'll see when you run it

Run with: `python3 main.py`

```
=== Positive-Number Classifier ===
Training samples   : 20
Test samples       : 180
Converged at epoch : 2
Final weight       : 6.0947
Final bias         : 0.0931
Decision boundary  : x = -0.0153
Test accuracy      : 100.0%
```

**Converged at epoch 2** — The perceptron made zero mistakes after only 2 full passes through the 20 training examples. This dataset is very easy — the two classes are far apart with a clear gap at zero.

**Decision boundary ≈ 0** — The learned boundary (x = −0.0153) is almost exactly at zero — which is the true separating point between positive and negative numbers. The perceptron discovered the rule by itself.

**100% test accuracy** — Every single one of the 180 test samples was classified correctly. Since positive numbers and negative numbers don't overlap, once the perceptron finds the boundary it gets everything right.
