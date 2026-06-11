import random
from perceptron import Perceptron

# Fix the random seed so every run produces identical data and initial weights.
random.seed(42)

# --- Data generation ---
# Two clearly separated classes:
#   Positive numbers (0–50)  → label 1 (True)
#   Negative numbers (-500–0) → label 0 (False)
# These are linearly separable at x=0, so a single perceptron can learn this perfectly.
positives = [random.uniform(0, 50) for _ in range(100)]
negatives = [random.uniform(-500, 0) for _ in range(100)]
values = positives + negatives
labels = [1] * 100 + [0] * 100

# Shuffle so the training loop doesn't see all positives before all negatives.
# Presenting mixed classes helps the weight update in a balanced direction each epoch.
combined = list(zip(values, labels))
random.shuffle(combined)
values, labels = [x for x, _ in combined], [y for _, y in combined]

# --- Train / test split (80 / 20) ---
# The model learns from the training set; the test set is held out to measure
# how well it generalises to data it has never seen.
split = int(0.1 * len(values))
train_x, test_x = values[:split], values[split:]
train_y, test_y = labels[:split], labels[split:]

# --- Train ---
p = Perceptron(learning_rate=0.1)
converged_at = p.train(train_x, train_y, epochs=100)

# --- Evaluate ---
acc = p.accuracy(test_x, test_y)

print("=== Positive-Number Classifier ===")
print(f"Training samples   : {len(train_x)}")
print(f"Test samples       : {len(test_x)}")
print(f"Converged at epoch : {converged_at if converged_at else 'did not converge within 100 epochs'}")
print(f"Final weight       : {p.weight:.4f}")
print(f"Final bias         : {p.bias:.4f}")
if p.weight != 0:
    # Decision boundary: the input value x at which the perceptron is exactly on
    # the fence (weighted sum = 0). Derived by solving weight * x + bias = 0.
    print(f"Decision boundary  : x = {-p.bias / p.weight:.4f}")
print(f"Test accuracy      : {acc * 100:.1f}%")
