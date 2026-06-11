import random


class Perceptron:
    def __init__(self, learning_rate=0.1, use_bias=True):
        # The weight controls how strongly the input influences the output.
        # Starting with a small random value avoids accidental bias toward one class.
        self.weight = random.uniform(-1, 1)

        # The bias shifts the decision boundary away from x=0, letting the
        # perceptron separate classes that don't straddle the origin.
        # When use_bias=False the boundary is permanently locked at x=0.
        self.bias = random.uniform(-1, 1) if use_bias else 0.0

        self.use_bias = use_bias
        self.learning_rate = learning_rate

    def predict(self, x):
        # Core decision function: compute the weighted sum plus bias, then
        # apply a step function — output 1 if the result is positive, 0 otherwise.
        # The decision boundary is the value of x where (weight * x + bias) == 0,
        # i.e. x = -bias / weight.
        return 1 if (self.weight * x + self.bias) > 0 else 0

    def train(self, data, labels, epochs=100):
        # One epoch = one full pass through every training sample.
        for epoch in range(epochs):
            errors = 0

            for x, y in zip(data, labels):
                pred = self.predict(x)
                # error is +1 (predicted 0, should be 1) or -1 (predicted 1, should be 0).
                # When error is 0 the prediction was correct — no update needed.
                error = y - pred

                if error != 0:
                    # Perceptron learning rule: nudge the weight in the direction
                    # that would have produced the correct output.
                    # - Positive error (missed a 1): push weight toward recognising x.
                    # - Negative error (false positive): push weight away from x.
                    self.weight += self.learning_rate * error * x

                    # Bias gets the same error signal but is independent of x —
                    # it simply slides the boundary left or right.
                    if self.use_bias:
                        self.bias += self.learning_rate * error

                    errors += 1

            # If no mistakes were made in this epoch, the data is fully learned.
            # Return immediately — extra epochs would change nothing.
            if errors == 0:
                return epoch + 1  # 1-indexed epoch number for readability

        # Reached the epoch limit without a perfect pass — did not converge.
        return None

    def accuracy(self, data, labels):
        # Count correct predictions and express as a fraction of total samples.
        correct = sum(1 for x, y in zip(data, labels) if self.predict(x) == y)
        return correct / len(data)
