import random
from perceptron import Perceptron

# Fix the random seed so every run produces identical data and initial weights.
random.seed(42)

# --- Data generation ---
# All inputs are scores between 0 and 100.
# Unlike demo 01, BOTH classes are positive numbers — the only difference is
# whether the score is above or below the threshold of 50.
# This means the decision boundary must sit at x=50, not at x=0.
# The perceptron must learn both a weight (slope) AND a bias (offset) to solve this.
scores = [random.uniform(0, 100) for _ in range(200)]
labels = [1 if s >= 50 else 0 for s in scores]

# --- Train / test split (80 / 20) ---
split = int(0.8 * len(scores))
train_x, test_x = scores[:split], scores[split:]
train_y, test_y = labels[:split], labels[split:]

# --- Train ---
p = Perceptron(learning_rate=0.1)
converged_at = p.train(train_x, train_y, epochs=100)

# --- Evaluate ---
acc = p.accuracy(test_x, test_y)

print("=== Exam-Score Classifier ===")
print(f"Training samples   : {len(train_x)}")
print(f"Test samples       : {len(test_x)}")
print(f"Passing threshold  : 50")
print(f"Converged at epoch : {converged_at if converged_at else 'did not converge within 100 epochs'}")
print(f"Final weight       : {p.weight:.4f}")
print(f"Final bias         : {p.bias:.4f}")
if p.weight != 0:
    # The learned boundary should be close to 50 — the true threshold.
    # Small deviations are normal; the boundary just needs to lie in the
    # gap between the two classes (no scores exist exactly at 50).
    print(f"Decision boundary  : x = {-p.bias / p.weight:.4f}  (should be ~50)")
print(f"Test accuracy      : {acc * 100:.1f}%")
