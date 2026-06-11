import random
from perceptron import Perceptron

# Fix the random seed for reproducible data generation.
random.seed(42)

# --- Data generation ---
# Exam scores 0–100, pass if >= 50.
# Critical property: ALL inputs are positive. The correct decision boundary
# is at x=50, which is NOT at the origin. This is the ideal dataset for
# showing why bias matters — without it the boundary cannot move off x=0.
scores = [random.uniform(0, 100) for _ in range(200)]
labels = [1 if s >= 50 else 0 for s in scores]

split = int(0.8 * len(scores))
train_x, test_x = scores[:split], scores[split:]
train_y, test_y = labels[:split], labels[split:]

print("=== Bias Demonstration ===")
print(f"Dataset            : exam scores 0–100, pass if >= 50")
print(f"Training samples   : {len(train_x)}")
print(f"Test samples       : {len(test_x)}")
print()

# --- Run 1: with bias ---
# Re-seed before creating each perceptron so both start from the same
# initial weight, making the comparison fair.
random.seed(99)
p_with = Perceptron(learning_rate=0.1, use_bias=True)
ep_with = p_with.train(train_x, train_y, epochs=100)
acc_with = p_with.accuracy(test_x, test_y)
boundary_with = -p_with.bias / p_with.weight if p_with.weight != 0 else float("inf")

print("--- With Bias ---")
print(f"Converged at epoch : {ep_with if ep_with else 'did not converge within 100 epochs'}")
print(f"Final weight       : {p_with.weight:.4f}")
print(f"Final bias         : {p_with.bias:.4f}")
# Bias is free to adjust, so the boundary can travel from x=0 to wherever
# the data requires (x≈50 here).
print(f"Decision boundary  : x = {boundary_with:.4f}  (should be ~50)")
print(f"Test accuracy      : {acc_with * 100:.1f}%")
print()

# --- Run 2: without bias ---
random.seed(99)
p_no = Perceptron(learning_rate=0.1, use_bias=False)
ep_no = p_no.train(train_x, train_y, epochs=100)
acc_no = p_no.accuracy(test_x, test_y)

print("--- Without Bias ---")
print(f"Converged at epoch : {ep_no if ep_no else 'did not converge within 100 epochs'}")
print(f"Final weight       : {p_no.weight:.4f}")
print(f"Bias               : 0.0  (fixed — cannot move)")
# Without bias: predict(x) = 1 if weight * x > 0, else 0.
# Because all x values are positive, the sign of weight * x is always the
# sign of weight alone — the actual value of x has no effect on the decision.
# The perceptron is stuck predicting the same class for every input.
print(f"Decision boundary  : x = 0.0  (locked at origin)")
print(f"Test accuracy      : {acc_no * 100:.1f}%")
print()

print("Explanation:")
print("  All inputs are positive (0–100). The correct boundary is at x=50.")
print("  Without bias, the boundary is always x=0 (i.e. weight * x > 0 only")
print("  depends on the sign of weight, not the value of x).")
print("  The perceptron ends up classifying everything as one class — ~50% accuracy.")
