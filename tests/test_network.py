"""Tests that verify the hand-written neural network.

Three independent kinds of evidence are used:

* backpropagation is compared with finite differences;
* predictions are compared with scikit-learn's ``MLPRegressor``, the tool
  used in the unit notes, when both are given the same weights;
* the network is trained on problems with known answers.
"""

import numpy as np
import pytest
from sklearn.neural_network import MLPRegressor

from packaging_defects.network import (
    Adam,
    MultilayerPerceptron,
    StandardScaler,
    gradient_check,
    train,
)


def small_problem(samples=40, inputs=6, outputs=3, seed=0):
    generator = np.random.default_rng(seed)
    return (
        generator.normal(size=(samples, inputs)),
        generator.normal(size=(samples, outputs)),
        generator.uniform(0.0, 2.0, size=(samples, outputs)),
    )


@pytest.mark.parametrize("layer_sizes", [(6, 3), (6, 8, 3), (6, 10, 7, 3)])
def test_backpropagation_matches_finite_differences(layer_sizes):
    inputs, targets, sample_weights = small_problem()
    network = MultilayerPerceptron(layer_sizes, seed=1, dtype=np.float64)
    worst = gradient_check(network, inputs, targets, sample_weights, penalty=0.3)
    assert worst < 1e-6


def test_gradient_check_detects_a_wrong_gradient(monkeypatch):
    inputs, targets, sample_weights = small_problem()
    network = MultilayerPerceptron((6, 8, 3), seed=1, dtype=np.float64)
    correct = network.loss_and_gradients

    def faulty(*arguments):
        loss, weight_gradients, bias_gradients = correct(*arguments)
        return loss, [1.1 * gradient for gradient in weight_gradients], bias_gradients

    monkeypatch.setattr(network, "loss_and_gradients", faulty)
    assert gradient_check(network, inputs, targets, sample_weights) > 1e-2


def test_loss_matches_the_formula_in_the_unit_notes():
    inputs, targets, _ = small_problem()
    network = MultilayerPerceptron((6, 8, 3), seed=2, dtype=np.float64)
    penalty, count = 0.5, len(inputs)
    loss, _, _ = network.loss_and_gradients(inputs, targets, None, penalty)
    errors = network.predict(inputs) - targets
    expected = (errors**2).sum() / (2 * count) + penalty / (2 * count) * sum(
        (weights**2).sum() for weights in network.weights
    )
    assert loss == pytest.approx(expected)


def test_zero_sample_weight_removes_an_output_from_the_loss():
    inputs, targets, _ = small_problem()
    network = MultilayerPerceptron((6, 8, 3), seed=2, dtype=np.float64)
    sample_weights = np.ones_like(targets)
    sample_weights[:, 2] = 0.0
    before = network.loss_and_gradients(inputs, targets, sample_weights)[0]
    targets[:, 2] += 100.0  # a huge error in the ignored output
    after = network.loss_and_gradients(inputs, targets, sample_weights)[0]
    assert before == pytest.approx(after)


# scikit-learn is only fitted briefly to obtain some weights, so its warning
# that it has not finished converging is expected.
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predictions_match_scikit_learn_given_the_same_weights():
    inputs, targets, _ = small_problem(samples=60)
    reference = MLPRegressor(hidden_layer_sizes=(9, 5), activation="relu", max_iter=30, random_state=0)
    reference.fit(inputs, targets)
    network = MultilayerPerceptron((6, 9, 5, 3), dtype=np.float64)
    network.set_parameters([*reference.coefs_, *reference.intercepts_])
    assert np.allclose(network.predict(inputs), reference.predict(inputs), atol=1e-10)


def test_network_without_hidden_layers_recovers_a_linear_relationship():
    generator = np.random.default_rng(3)
    inputs = generator.normal(size=(400, 3)).astype(np.float32)
    true_weights = np.array([[1.5], [-2.0], [0.5]], dtype=np.float32)
    targets = inputs @ true_weights + 0.7
    network = MultilayerPerceptron((3, 1), seed=0)
    train(network, lambda rows: (inputs[rows], targets[rows], None), len(inputs),
          epochs=200, batch_size=50, learning_rate=0.02, penalty=0.0)
    assert np.allclose(network.weights[0], true_weights, atol=0.02)
    assert network.biases[0][0] == pytest.approx(0.7, abs=0.02)


def test_hidden_layers_learn_a_curved_relationship_a_linear_model_cannot():
    generator = np.random.default_rng(4)
    inputs = generator.uniform(-2, 2, size=(600, 2)).astype(np.float32)
    targets = (np.sin(2 * inputs[:, :1]) * inputs[:, 1:]).astype(np.float32)

    def r_squared(layer_sizes):
        network = MultilayerPerceptron(layer_sizes, seed=0)
        train(network, lambda rows: (inputs[rows], targets[rows], None), len(inputs),
              epochs=300, batch_size=64, learning_rate=0.01, penalty=0.0)
        residual = ((network.predict(inputs) - targets) ** 2).sum()
        return 1.0 - residual / ((targets - targets.mean()) ** 2).sum()

    assert r_squared((2, 32, 32, 1)) > 0.95
    assert r_squared((2, 1)) < 0.3


def test_penalty_shrinks_the_weights():
    inputs, targets, _ = small_problem(samples=200)
    inputs, targets = inputs.astype(np.float32), targets.astype(np.float32)
    sizes = []
    for penalty in (0.0, 50.0):
        network = MultilayerPerceptron((6, 16, 3), seed=0)
        train(network, lambda rows: (inputs[rows], targets[rows], None), len(inputs),
              epochs=60, batch_size=50, learning_rate=0.01, penalty=penalty)
        sizes.append(sum(float((weights**2).sum()) for weights in network.weights))
    assert sizes[1] < 0.5 * sizes[0]


def test_early_stopping_restores_the_best_weights():
    inputs, targets, _ = small_problem()
    inputs, targets = inputs.astype(np.float32), targets.astype(np.float32)
    network = MultilayerPerceptron((6, 8, 3), seed=0)
    scores = iter([0.1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    snapshots = []

    def validation_score():
        snapshots.append(network.copy_parameters())
        return next(scores)

    history = train(network, lambda rows: (inputs[rows], targets[rows], None), len(inputs),
                    epochs=50, validation_score=validation_score, patience=3)
    assert len(history["loss"]) == 5  # the best pass was the 2nd, then 3 without improvement
    for kept, best in zip(network.parameters(), snapshots[1]):
        assert np.array_equal(kept, best)


def test_training_is_repeatable_with_the_same_seed():
    inputs, targets, _ = small_problem()
    inputs, targets = inputs.astype(np.float32), targets.astype(np.float32)
    results = []
    for _ in range(2):
        network = MultilayerPerceptron((6, 8, 3), seed=5)
        train(network, lambda rows: (inputs[rows], targets[rows], None), len(inputs), epochs=5, seed=5)
        results.append(network.predict(inputs))
    assert np.array_equal(results[0], results[1])


def test_adam_minimises_a_simple_bowl():
    position = [np.array([5.0, -3.0])]
    optimiser = Adam(position, learning_rate=0.1)
    for _ in range(500):
        optimiser.step([2.0 * position[0]])  # slope of x squared plus y squared
    assert np.allclose(position[0], 0.0, atol=1e-3)


def test_scaler_uses_training_statistics_only():
    training = np.array([[1.0, 10.0], [3.0, 30.0]])
    scaler = StandardScaler().fit(training)
    assert np.allclose(scaler.mean, [2.0, 20.0])
    assert np.allclose(scaler.transform(training), [[-1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(scaler.transform(np.array([[5.0, 50.0]])), [[3.0, 3.0]])


def test_scaler_leaves_constant_features_alone():
    scaler = StandardScaler().fit(np.array([[4.0], [4.0]]))
    assert np.allclose(scaler.transform(np.array([[4.0]])), [[0.0]])


def test_scaler_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.zeros((1, 2)))


def test_parameter_and_operation_counts():
    network = MultilayerPerceptron((4, 5, 2))
    assert network.number_of_parameters() == 4 * 5 + 5 + 5 * 2 + 2
    assert network.operations_per_prediction() == 2 * (4 * 5 + 5 * 2) + (5 + 2)


def test_layer_sizes_are_validated():
    with pytest.raises(ValueError):
        MultilayerPerceptron((4,))
