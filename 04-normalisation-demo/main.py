import random
from perceptron import Perceptron

# Fix the random seed for reproducible data and initial weights.
random.seed(42)

# --- Data generation ---
# Exam scores on a 0–10000 scale, passing threshold at 5000.
# The deliberately large range (0–10000) is the key to this demo:
# the weight update rule is  weight += lr * error * x.
# With lr=0.1 and x up to 10000, a single update can shift the weight by 1000.
# These huge oscillating jumps slow convergence dramatically.
# After normalising to [0, 1], the maximum shift per step is just 0.1,
# giving the perceptron stable, incremental steps toward the solution.
scores = [random.uniform(0, 10000) for _ in range(200)]
labels = [1 if s >= 5000 else 0 for s in scores]

split = int(0.8 * len(scores))
train_x, test_x = scores[:split], scores[split:]
train_y, test_y = labels[:split], labels[split:]

print("=== Normalisation Demo ===")
print(f"Dataset            : exam scores 0–10000, pass if >= 5000")
print(f"Training samples   : {len(train_x)}")
print(f"Test samples       : {len(test_x)}")
print(f"Raw value range    : [{min(train_x):.1f}, {max(train_x):.1f}]")
print()

# --- Run 1: without normalisation ---
# Re-seed so both perceptrons start from the same initial weight, for a fair comparison.
random.seed(99)
p_raw = Perceptron(learning_rate=0.1)
ep_raw = p_raw.train(train_x, train_y, epochs=1000)
acc_raw = p_raw.accuracy(test_x, test_y)

print("--- Without Normalisation ---")
if ep_raw:
    print(f"Converged at epoch : {ep_raw}")
else:
    print(f"Did not converge within 1000 epochs")
print(f"Final weight       : {p_raw.weight:.4f}")
print(f"Final bias         : {p_raw.bias:.4f}")
print(f"Test accuracy      : {acc_raw * 100:.1f}%")
print()

# --- Run 2: with min-max normalisation ---
# Compute the scale from the TRAINING set only.
# Using the test set to compute min/max would be "data leakage" — in a real
# scenario you wouldn't have test data available at training time.
x_min = min(train_x)
x_max = max(train_x)

def normalise(x):
    # Squashes any value into the [0, 1] range using the training-set bounds.
    # Values at x_min map to 0; values at x_max map to 1; everything else scales linearly.
    return (x - x_min) / (x_max - x_min)

norm_train_x = [normalise(x) for x in train_x]
# Apply the SAME scale to test data — do not recompute min/max from test set.
norm_test_x = [normalise(x) for x in test_x]

random.seed(99)
p_norm = Perceptron(learning_rate=0.1)
ep_norm = p_norm.train(norm_train_x, train_y, epochs=1000)
acc_norm = p_norm.accuracy(norm_test_x, test_y)

print("--- With Min-Max Normalisation (scaled to [0, 1]) ---")
print(f"Scale applied      : (x - {x_min:.0f}) / ({x_max:.0f} - {x_min:.0f})")
if ep_norm:
    print(f"Converged at epoch : {ep_norm}")
else:
    print(f"Did not converge within 1000 epochs")
print(f"Final weight       : {p_norm.weight:.4f}")
print(f"Final bias         : {p_norm.bias:.4f}")
print(f"Test accuracy      : {acc_norm * 100:.1f}%")
print()

# --- Summary ---
if ep_raw and ep_norm:
    print(f"Result: normalisation reduced convergence from epoch {ep_raw} to {ep_norm} "
          f"({ep_raw - ep_norm} fewer epochs, {ep_raw / ep_norm:.1f}x faster).")
elif not ep_raw and ep_norm:
    print(f"Result: raw data failed to converge in 1000 epochs; "
          f"normalised data converged at epoch {ep_norm}.")
elif ep_raw and not ep_norm:
    print(f"Result: unexpected — raw converged at {ep_raw} but normalised did not.")
else:
    print("Result: neither run converged within 1000 epochs.")
