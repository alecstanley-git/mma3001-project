"""A neural network regressor written by hand with NumPy.

This is the regression multilayer perceptron from Week 5 of the unit
(section 5.4, "Neural Network Regression"), written out instead of calling
``sklearn.neural_network.MLPRegressor``, so that every step is visible:

1. **Scaling.**  :class:`StandardScaler` shifts and scales each input feature
   to zero mean and unit spread, using statistics from training data only.
2. **Forward prediction.**  Each layer computes ``z = a W + b``.  Hidden
   layers then apply the rectified linear unit ``max(0, z)``; the output layer
   leaves ``z`` unchanged, as a regression output should.
3. **Loss.**  Half the mean squared error plus a penalty on large weights:

   $$J = \\frac{1}{2n} \\sum_i \\sum_k s_{ik} (y_{ik} - \\hat y_{ik})^2 + \\frac{\\alpha}{2n} \\lVert W \\rVert_2^2$$

   This is the loss in the unit notes with one addition: a weight
   $s_{ik}$ on each output of each sample, which lets a caller ignore an
   output where it has no meaning (see :mod:`packaging_defects.grid`).
4. **Backpropagation.**  The chain rule gives the slope of the loss with
   respect to every weight, working from the output layer backwards.
5. **Adam.**  The optimiser used in the unit notes; it takes steps down the
   slope using running averages of the slope and of its square.
6. **Early stopping.**  Training stops once a validation score has not
   improved for a set number of passes, and the best weights are restored.

:func:`gradient_check` verifies step 4 against finite differences.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class StandardScaler:
    """Shift and scale features to zero mean and unit standard deviation.

    The statistics are learned once, from training data, and then reused for
    every later input.  Learning them from validation or test data would let
    information leak from data the model is supposed not to have seen.

    Attributes
    ----------
    mean : numpy.ndarray
        Mean of each feature in the training data.
    scale : numpy.ndarray
        Standard deviation of each feature in the training data.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        """Learn the mean and spread of each feature.

        Parameters
        ----------
        values : numpy.ndarray, shape (..., features)
            Training data.  All axes except the last are treated as samples.

        Returns
        -------
        StandardScaler
            This object, so calls can be chained.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        self.mean = flat.mean(axis=0)
        spread = flat.std(axis=0)
        # A feature that never changes would divide by zero; leave it alone.
        self.scale = np.where(spread > 1e-12, spread, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the learned shift and scale.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.mean is None:
            raise RuntimeError("StandardScaler.fit must be called before transform")
        return ((values - self.mean) / self.scale).astype(np.float32)


class MultilayerPerceptron:
    """Fully connected regression network with rectified linear hidden layers.

    Parameters
    ----------
    layer_sizes : sequence of int
        Number of values in each layer, from input to output.  ``(20, 8, 3)``
        means 20 input features, one hidden layer of 8 neurons and 3 outputs.
        ``(20, 3)`` has no hidden layer and is ordinary linear regression.
    seed : int
        Seed for the random starting weights, for repeatable results.
    dtype : numpy dtype
        Single precision is used for training speed; double precision is used
        when checking gradients.

    Attributes
    ----------
    weights : list of numpy.ndarray
        One matrix per pair of adjacent layers, shape (inputs, outputs).
    biases : list of numpy.ndarray
        One vector per layer after the input.
    """

    def __init__(self, layer_sizes, seed: int = 0, dtype=np.float32) -> None:
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        if len(self.layer_sizes) < 2 or min(self.layer_sizes) < 1:
            raise ValueError("layer_sizes needs at least an input and an output size")
        generator = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for inputs, outputs in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            # Starting weights are random with a spread of sqrt(2 / inputs),
            # which keeps signals a similar size from layer to layer when the
            # rectified linear unit sets about half of them to zero.
            spread = np.sqrt(2.0 / inputs)
            self.weights.append(generator.normal(0.0, spread, (inputs, outputs)).astype(dtype))
            self.biases.append(np.zeros(outputs, dtype=dtype))

    # ------------------------------------------------------------------
    # Forward prediction
    # ------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> list[np.ndarray]:
        """Pass inputs through the network, keeping every layer's output.

        Parameters
        ----------
        inputs : numpy.ndarray, shape (samples, input features)

        Returns
        -------
        list of numpy.ndarray
            The input followed by the output of each layer.  The last entry
            is the prediction.  Backpropagation needs the earlier entries.
        """
        activations = [inputs]
        last = len(self.weights) - 1
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            values = activations[-1] @ weights + biases
            if index < last:
                values = np.maximum(values, 0.0)  # rectified linear unit
            activations.append(values)
        return activations

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the network's prediction for each input sample."""
        return self.forward(inputs)[-1]

    # ------------------------------------------------------------------
    # Loss and backpropagation
    # ------------------------------------------------------------------
    def loss_and_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
        penalty: float = 0.0,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """Compute the loss and its slope with respect to every parameter.

        Parameters
        ----------
        inputs : numpy.ndarray, shape (samples, input features)
        targets : numpy.ndarray, shape (samples, outputs)
            The values the network should have predicted.
        sample_weights : numpy.ndarray, shape (samples, outputs), optional
            Importance of each output of each sample (``s`` in the module
            documentation).  ``None`` means every output counts equally.
        penalty : float
            Strength of the penalty on large weights (``alpha`` in the unit
            notes).

        Returns
        -------
        loss : float
        weight_gradients : list of numpy.ndarray
            Same shapes as :attr:`weights`.
        bias_gradients : list of numpy.ndarray
            Same shapes as :attr:`biases`.
        """
        count = inputs.shape[0]
        activations = self.forward(inputs)
        errors = activations[-1] - targets
        if sample_weights is None:
            sample_weights = np.ones_like(errors)

        squared_weights = sum(float((weights**2).sum()) for weights in self.weights)
        loss = (float((sample_weights * errors**2).sum()) + penalty * squared_weights) / (2 * count)

        # Slope of the loss with respect to the output layer's values.
        delta = sample_weights * errors / count
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for index in reversed(range(len(self.weights))):
            weight_gradients[index] = (
                activations[index].T @ delta + (penalty / count) * self.weights[index]
            )
            bias_gradients[index] = delta.sum(axis=0)
            if index > 0:
                # Chain rule: pass the slope back through the weights, then
                # through the rectified linear unit, whose slope is 1 where
                # its output was positive and 0 elsewhere.
                delta = (delta @ self.weights[index].T) * (activations[index] > 0)
        return loss, weight_gradients, bias_gradients

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def parameters(self) -> list[np.ndarray]:
        """Return all weight matrices followed by all bias vectors."""
        return [*self.weights, *self.biases]

    def copy_parameters(self) -> list[np.ndarray]:
        """Return an independent copy of every parameter."""
        return [parameter.copy() for parameter in self.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Replace every parameter, for example to restore the best weights."""
        count = len(self.weights)
        self.weights = [parameter.copy() for parameter in parameters[:count]]
        self.biases = [parameter.copy() for parameter in parameters[count:]]

    def number_of_parameters(self) -> int:
        """Count the numbers the network learns."""
        return sum(parameter.size for parameter in self.parameters())

    def operations_per_prediction(self) -> int:
        """Estimate the floating point operations for one forward prediction.

        Each connection costs one multiplication and one addition, and each
        neuron after the input costs one more addition for its bias.
        """
        connections = sum(weights.size for weights in self.weights)
        neurons = sum(biases.size for biases in self.biases)
        return 2 * connections + neurons


class Adam:
    """The Adam optimiser.

    Plain gradient descent steps every parameter by the same multiple of its
    slope.  Adam instead keeps, for every parameter, a running average of the
    slope (so steps follow the consistent direction and ignore noise) and of
    the squared slope (so parameters with large slopes take smaller steps).

    Parameters
    ----------
    parameters : list of numpy.ndarray
        The arrays to update in place.
    learning_rate : float
        Size of each step.
    first_decay, second_decay : float
        How quickly the two running averages forget old slopes.
    """

    def __init__(
        self,
        parameters: list[np.ndarray],
        learning_rate: float = 0.001,
        first_decay: float = 0.9,
        second_decay: float = 0.999,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first_decay = first_decay
        self.second_decay = second_decay
        self.average_slope = [np.zeros_like(parameter) for parameter in parameters]
        self.average_square = [np.zeros_like(parameter) for parameter in parameters]
        self.steps_taken = 0

    def step(self, gradients: list[np.ndarray]) -> None:
        """Move every parameter one step down its slope.

        Parameters
        ----------
        gradients : list of numpy.ndarray
            Slope of the loss for each parameter, in the same order.
        """
        self.steps_taken += 1
        # The averages start at zero, so early values are too small; dividing
        # by these factors corrects that.
        first_correction = 1.0 - self.first_decay**self.steps_taken
        second_correction = 1.0 - self.second_decay**self.steps_taken
        for parameter, gradient, slope, square in zip(
            self.parameters, gradients, self.average_slope, self.average_square
        ):
            slope *= self.first_decay
            slope += (1.0 - self.first_decay) * gradient
            square *= self.second_decay
            square += (1.0 - self.second_decay) * gradient**2
            parameter -= (
                self.learning_rate
                * (slope / first_correction)
                / (np.sqrt(square / second_correction) + 1e-8)
            )


def train(
    network: MultilayerPerceptron,
    make_batch: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    number_of_samples: int,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    penalty: float = 0.0001,
    validation_score: Callable[[], float] | None = None,
    patience: int = 8,
    seed: int = 0,
    report: Callable[[str], None] | None = None,
) -> dict[str, list[float]]:
    """Fit a network by repeated passes through the training samples.

    Parameters
    ----------
    network : MultilayerPerceptron
        The network to train; its parameters are updated in place.
    make_batch : callable
        Given an array of sample numbers, returns ``(inputs, targets,
        sample_weights)`` for those samples.  Supplying samples through a
        function lets the caller build them only when needed, which saves
        memory.
    number_of_samples : int
        How many training samples exist.
    epochs : int
        Maximum number of passes through the training samples.
    batch_size : int
        Samples used for each update of the weights.
    learning_rate : float
        Step size for :class:`Adam`.
    penalty : float
        Strength of the penalty on large weights.
    validation_score : callable, optional
        Returns a score, higher being better, measured on data kept out of
        training.  Called after every pass.  If given, training stops early
        and the best-scoring weights are restored.
    patience : int
        Passes to wait for an improved validation score before stopping.
    seed : int
        Seed for the random order of samples.
    report : callable, optional
        Receives one line of progress text per pass, for example ``print``.

    Returns
    -------
    dict
        ``"loss"`` holds the average training loss of each pass and
        ``"validation_score"`` the matching validation scores.
    """
    generator = np.random.default_rng(seed)
    optimiser = Adam(network.parameters(), learning_rate)
    history = {"loss": [], "validation_score": []}
    best_score, best_parameters, passes_without_improvement = -np.inf, None, 0

    for epoch in range(1, epochs + 1):
        order = generator.permutation(number_of_samples)
        total_loss, batches = 0.0, 0
        for start in range(0, number_of_samples, batch_size):
            inputs, targets, sample_weights = make_batch(order[start : start + batch_size])
            loss, weight_gradients, bias_gradients = network.loss_and_gradients(
                inputs, targets, sample_weights, penalty
            )
            optimiser.step([*weight_gradients, *bias_gradients])
            total_loss += loss
            batches += 1
        history["loss"].append(total_loss / batches)

        message = f"pass {epoch:3d}  training loss {history['loss'][-1]:.5f}"
        if validation_score is not None:
            score = float(validation_score())
            history["validation_score"].append(score)
            message += f"  validation score {score:.4f}"
            if score > best_score + 1e-5:
                best_score, best_parameters = score, network.copy_parameters()
                passes_without_improvement = 0
            else:
                passes_without_improvement += 1
        if report is not None:
            report(message)
        if validation_score is not None and passes_without_improvement >= patience:
            break

    if best_parameters is not None:
        network.set_parameters(best_parameters)
    return history


def gradient_check(
    network: MultilayerPerceptron,
    inputs: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray | None = None,
    penalty: float = 0.0,
    step: float = 1e-5,
    checks_per_array: int = 20,
    seed: int = 0,
) -> float:
    """Verify backpropagation against central finite differences.

    Backpropagation is easy to get subtly wrong.  This compares the slope it
    reports for a parameter $p$ with the estimate

    $$\\frac{J(p + h) - J(p - h)}{2h}$$

    which needs nothing but the loss itself.  The two should agree to many
    decimal places.  Use a network created with ``dtype=numpy.float64``.

    Parameters
    ----------
    network, inputs, targets, sample_weights, penalty
        As for :meth:`MultilayerPerceptron.loss_and_gradients`.
    step : float
        The small change ``h`` made to each parameter.
    checks_per_array : int
        Parameters tested, chosen at random, in each weight and bias array.
    seed : int
        Seed for choosing which parameters to test.

    Returns
    -------
    float
        Largest relative difference found.  Around ``1e-7`` or smaller means
        backpropagation is correct.
    """
    generator = np.random.default_rng(seed)
    _, weight_gradients, bias_gradients = network.loss_and_gradients(
        inputs, targets, sample_weights, penalty
    )
    worst = 0.0
    for parameter, gradient in zip(network.parameters(), [*weight_gradients, *bias_gradients]):
        flat_indices = generator.choice(
            parameter.size, size=min(checks_per_array, parameter.size), replace=False
        )
        for flat_index in flat_indices:
            position = np.unravel_index(flat_index, parameter.shape)
            original = parameter[position]
            parameter[position] = original + step
            loss_above = network.loss_and_gradients(inputs, targets, sample_weights, penalty)[0]
            parameter[position] = original - step
            loss_below = network.loss_and_gradients(inputs, targets, sample_weights, penalty)[0]
            parameter[position] = original

            estimate = (loss_above - loss_below) / (2 * step)
            size = max(abs(estimate), abs(gradient[position]), 1e-12)
            worst = max(worst, abs(estimate - gradient[position]) / size)
    return worst
