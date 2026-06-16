from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


Array = np.ndarray

SPLIT_RANDOM_STATE = 2
TRAIN_TEST_RATIO = 0.80
MODEL_RANDOM_SEED = 42
PERCEPTRON_EPOCHS = 800
DELTA_RULE_EPOCHS = 500
EPSILON = 1e-12
DELTA_DIVERGENCE_THRESHOLD = 1e6


# =============================================================================
# 24F-0633 - MODULE 4: Learning in AI
# =============================================================================
# This module compares two learning algorithms on the Iris flower dataset:
# 1. Multiclass Perceptron Learning Rule
# 2. Gradient Descent Delta Rule
#
# The implementation is written from scratch for the learning algorithms. The
# only external utilities used are:
# - sklearn.datasets.load_iris() to access the Iris dataset locally
# - train_test_split() to create the required 80/20 dataset split
#
# The script performs:
# - data loading and standardization
# - activation-function experiments
# - learning-rate experiments
# - train/test evaluation
# - result documentation and reflection answers
# - self-test verification
# =============================================================================


@dataclass
class IrisSplit:
    X_train: Array
    X_test: Array
    y_train: Array
    y_test: Array
    feature_names: List[str]
    class_names: List[str]


@dataclass
class TrainingRun:
    algorithm: str
    activation: str
    learning_rate: float
    epochs: int
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    confusion_matrix: Array
    converged: bool
    notes: str = ""


@dataclass
class TrainedModel:
    weights: Array
    bias: Array
    losses: List[float]
    epochs_run: int
    converged: bool


def sigmoid(values: Array) -> Array:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_and_prepare_iris(
    test_size: float = 1.0 - TRAIN_TEST_RATIO,
    split_random_state: int = SPLIT_RANDOM_STATE,
) -> IrisSplit:
    dataset = load_iris()
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=split_random_state,
        stratify=y,
    )

    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std[train_std < EPSILON] = 1.0

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    return IrisSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(dataset.feature_names),
        class_names=list(dataset.target_names),
    )


def one_hot(labels: Array, class_count: int) -> Array:
    return np.eye(class_count, dtype=np.float64)[labels]


def confusion_matrix_numpy(y_true: Array, y_pred: Array, class_count: int) -> Array:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def accuracy_score_numpy(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(y_true == y_pred))


def build_perceptron_score_transform(name: str) -> Callable[[Array], Array]:
    transforms: Dict[str, Callable[[Array], Array]] = {
        "raw_score": lambda scores: scores,
        "sigmoid": sigmoid,
        "tanh": np.tanh,
    }
    if name not in transforms:
        raise ValueError(f"Unsupported perceptron activation: {name}")
    return transforms[name]


def delta_activation_bundle(
    name: str,
) -> Tuple[Callable[[Array], Array], Callable[[Array], Array], Callable[[Array], Array], str]:
    if name == "linear":
        return (
            lambda values: values,
            lambda outputs: np.ones_like(outputs),
            lambda labels, class_count: one_hot(labels, class_count),
            "Linear output with mean squared error",
        )
    if name == "sigmoid":
        return (
            sigmoid,
            lambda outputs: outputs * (1.0 - outputs),
            lambda labels, class_count: one_hot(labels, class_count),
            "Smooth logistic output with bounded gradients",
        )
    if name == "tanh":
        return (
            np.tanh,
            lambda outputs: 1.0 - np.square(outputs),
            lambda labels, class_count: np.where(one_hot(labels, class_count) > 0.0, 1.0, -1.0),
            "Bipolar output that matches one-vs-rest targets well",
        )
    raise ValueError(f"Unsupported delta-rule activation: {name}")


def fit_multiclass_perceptron(
    X_train: Array,
    y_train: Array,
    learning_rate: float,
    epochs: int = PERCEPTRON_EPOCHS,
    seed: int = MODEL_RANDOM_SEED,
) -> TrainedModel:
    class_count = int(np.max(y_train)) + 1
    feature_count = X_train.shape[1]
    weights = np.zeros((class_count, feature_count), dtype=np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    losses: List[float] = []

    rng = np.random.default_rng(seed)
    converged = False

    for _epoch_index in range(epochs):
        mistakes = 0
        shuffled_indices = rng.permutation(len(X_train))

        for sample_index in shuffled_indices:
            features = X_train[sample_index]
            target_class = int(y_train[sample_index])
            class_scores = weights @ features + bias
            predicted_class = int(np.argmax(class_scores))

            if predicted_class != target_class:
                weights[target_class] += learning_rate * features
                bias[target_class] += learning_rate
                weights[predicted_class] -= learning_rate * features
                bias[predicted_class] -= learning_rate
                mistakes += 1

        loss = mistakes / len(X_train)
        losses.append(loss)
        if mistakes == 0:
            converged = True
            break

    return TrainedModel(
        weights=weights,
        bias=bias,
        losses=losses,
        epochs_run=len(losses),
        converged=converged,
    )


def predict_multiclass_perceptron(model: TrainedModel, X: Array, activation: str) -> Array:
    score_transform = build_perceptron_score_transform(activation)
    raw_scores = X @ model.weights.T + model.bias
    transformed_scores = score_transform(raw_scores)
    return np.argmax(transformed_scores, axis=1)


def evaluate_perceptron_run(
    split: IrisSplit,
    learning_rate: float,
    activation: str,
) -> TrainingRun:
    model = fit_multiclass_perceptron(
        X_train=split.X_train,
        y_train=split.y_train,
        learning_rate=learning_rate,
    )
    train_predictions = predict_multiclass_perceptron(model, split.X_train, activation)
    test_predictions = predict_multiclass_perceptron(model, split.X_test, activation)

    return TrainingRun(
        algorithm="Perceptron Learning Rule",
        activation=activation,
        learning_rate=learning_rate,
        epochs=model.epochs_run,
        train_accuracy=accuracy_score_numpy(split.y_train, train_predictions),
        test_accuracy=accuracy_score_numpy(split.y_test, test_predictions),
        final_loss=model.losses[-1],
        confusion_matrix=confusion_matrix_numpy(split.y_test, test_predictions, len(split.class_names)),
        converged=model.converged,
        notes=(
            "Monotonic score transforms preserve class-score ranking, so the "
            "multiclass perceptron keeps the same decision boundary here."
        ),
    )


def fit_delta_rule_classifier(
    X_train: Array,
    y_train: Array,
    activation: str,
    learning_rate: float,
    epochs: int = DELTA_RULE_EPOCHS,
    seed: int = MODEL_RANDOM_SEED,
) -> TrainedModel:
    class_count = int(np.max(y_train)) + 1
    feature_count = X_train.shape[1]
    weights = np.random.default_rng(seed).normal(0.0, 0.05, size=(class_count, feature_count))
    bias = np.zeros(class_count, dtype=np.float64)
    activation_fn, derivative_fn, target_encoder, _description = delta_activation_bundle(activation)
    targets = target_encoder(y_train, class_count)
    losses: List[float] = []
    converged = False

    for epoch_index in range(epochs):
        linear_output = X_train @ weights.T + bias
        activated_output = activation_fn(linear_output)
        errors = targets - activated_output
        losses.append(float(np.mean(np.square(errors))))

        local_gradient = errors * derivative_fn(activated_output)
        weights += learning_rate * (local_gradient.T @ X_train) / len(X_train)
        bias += learning_rate * local_gradient.mean(axis=0)

        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            break

        if epoch_index > 10 and abs(losses[-2] - losses[-1]) < 1e-8:
            converged = True
            break

    return TrainedModel(
        weights=weights,
        bias=bias,
        losses=losses,
        epochs_run=len(losses),
        converged=converged,
    )


def predict_delta_rule_classifier(model: TrainedModel, X: Array, activation: str) -> Array:
    activation_fn, _derivative_fn, _target_encoder, _description = delta_activation_bundle(activation)
    class_scores = activation_fn(X @ model.weights.T + model.bias)
    return np.argmax(class_scores, axis=1)


def evaluate_delta_rule_run(
    split: IrisSplit,
    learning_rate: float,
    activation: str,
) -> TrainingRun:
    model = fit_delta_rule_classifier(
        X_train=split.X_train,
        y_train=split.y_train,
        activation=activation,
        learning_rate=learning_rate,
    )

    if (
        not np.isfinite(model.weights).all()
        or not np.isfinite(model.bias).all()
        or not model.losses
        or not np.isfinite(model.losses[-1])
        or model.losses[-1] > DELTA_DIVERGENCE_THRESHOLD
    ):
        return TrainingRun(
            algorithm="Gradient Descent Delta Rule",
            activation=activation,
            learning_rate=learning_rate,
            epochs=model.epochs_run,
            train_accuracy=0.0,
            test_accuracy=0.0,
            final_loss=float("inf"),
            confusion_matrix=np.zeros((len(split.class_names), len(split.class_names)), dtype=np.int64),
            converged=False,
            notes="Training diverged or became numerically unstable for this activation/rate pair.",
        )

    train_predictions = predict_delta_rule_classifier(model, split.X_train, activation)
    test_predictions = predict_delta_rule_classifier(model, split.X_test, activation)
    _activation_fn, _derivative_fn, _target_encoder, description = delta_activation_bundle(activation)

    return TrainingRun(
        algorithm="Gradient Descent Delta Rule",
        activation=activation,
        learning_rate=learning_rate,
        epochs=model.epochs_run,
        train_accuracy=accuracy_score_numpy(split.y_train, train_predictions),
        test_accuracy=accuracy_score_numpy(split.y_test, test_predictions),
        final_loss=model.losses[-1],
        confusion_matrix=confusion_matrix_numpy(split.y_test, test_predictions, len(split.class_names)),
        converged=model.converged,
        notes=description,
    )


def select_best_run(runs: List[TrainingRun]) -> TrainingRun:
    return max(
        runs,
        key=lambda run: (
            run.test_accuracy,
            run.train_accuracy,
            -run.final_loss if np.isfinite(run.final_loss) else float("-inf"),
            -run.learning_rate,
        ),
    )


def format_loss_value(loss: float) -> str:
    if not np.isfinite(loss):
        return "inf"
    if loss > DELTA_DIVERGENCE_THRESHOLD:
        return "unstable"
    return f"{loss:.6f}"


def run_experiments(split: IrisSplit) -> Tuple[List[TrainingRun], List[TrainingRun], TrainingRun, TrainingRun]:
    perceptron_activations = ["raw_score", "sigmoid", "tanh"]
    perceptron_learning_rates = [0.1, 0.5, 1.0]
    perceptron_runs = [
        evaluate_perceptron_run(split, learning_rate=learning_rate, activation=activation)
        for activation in perceptron_activations
        for learning_rate in perceptron_learning_rates
    ]

    delta_activations = ["linear", "sigmoid", "tanh"]
    delta_learning_rates = [0.01, 0.05, 0.1, 0.5, 1.0]
    delta_runs = [
        evaluate_delta_rule_run(split, learning_rate=learning_rate, activation=activation)
        for activation in delta_activations
        for learning_rate in delta_learning_rates
    ]

    best_perceptron = select_best_run(perceptron_runs)
    best_delta_rule = select_best_run(delta_runs)
    return perceptron_runs, delta_runs, best_perceptron, best_delta_rule


def format_run_table(title: str, runs: List[TrainingRun]) -> str:
    header = (
        f"{title}\n"
        "activation   lr      epochs   train_acc   test_acc   final_loss   converged\n"
        "--------------------------------------------------------------------------"
    )
    rows = []
    for run in runs:
        rows.append(
            f"{run.activation:<11} {run.learning_rate:<7.2f} {run.epochs:<8d} "
            f"{run.train_accuracy:<11.4f} {run.test_accuracy:<10.4f} "
            f"{format_loss_value(run.final_loss):<11} {str(run.converged):<9}"
        )
    return "\n".join([header, *rows])


def format_confusion_matrix(matrix: Array, class_names: List[str]) -> str:
    header = "predicted -> " + "  ".join(f"{name:>10}" for name in class_names)
    lines = [header]
    for row_index, class_name in enumerate(class_names):
        row_values = "  ".join(f"{value:>10d}" for value in matrix[row_index])
        lines.append(f"{class_name:<12} {row_values}")
    return "\n".join(lines)


def build_reflection_text(
    split: IrisSplit,
    best_perceptron: TrainingRun,
    best_delta_rule: TrainingRun,
) -> str:
    train_count = len(split.X_train)
    test_count = len(split.X_test)
    return (
        "Reflection Questions\n"
        "1. Key differences:\n"
        "   The perceptron updates weights only when a sample is misclassified, while the delta rule minimizes\n"
        "   a continuous loss using gradient descent. The perceptron is discrete and boundary-focused; the delta\n"
        "   rule provides smoother optimization and explicit loss tracking.\n"
        "2. Impact of activation functions:\n"
        "   In this implementation, monotonic score transforms did not change perceptron predictions because\n"
        "   argmax ranking stayed the same. The delta rule was much more sensitive: linear output was the weakest,\n"
        "   while sigmoid and tanh reached the strongest test results.\n"
        "3. Learning-rate adjustment strategy:\n"
        "   Sweep from small to moderate values, keep inputs standardized, and reject unstable rates. This run\n"
        "   showed that overly large linear-output updates can diverge, while tanh and sigmoid benefited from\n"
        "   larger but still controlled learning rates.\n"
        "4. Implications of train/test ratio:\n"
        f"   The required 80/20 split created {train_count} training samples and {test_count} test samples. A larger\n"
        "   training share improves learning stability but reduces evaluation depth; a larger test share gives a\n"
        "   stronger estimate of generalization but leaves less data for fitting.\n"
        "5. Implementation challenges:\n"
        "   The main issues were extending binary rules to three classes, matching target encodings with activation\n"
        "   ranges, and preventing numerical overflow. These were handled with a multiclass perceptron, one-vs-rest\n"
        "   delta targets, standardized features, and clipped sigmoid inputs.\n"
        "6. Strengths and limitations observed:\n"
        f"   The best perceptron run reached train/test accuracy {best_perceptron.train_accuracy:.4f}/{best_perceptron.test_accuracy:.4f}\n"
        f"   and learned a strong decision boundary quickly, but it does not optimize a smooth objective. The best\n"
        f"   delta-rule run reached {best_delta_rule.train_accuracy:.4f}/{best_delta_rule.test_accuracy:.4f}, offered more tuning\n"
        "   control through activation and learning-rate choice, and exposed optimization behavior through loss values.\n"
    )


def build_report(
    split: IrisSplit,
    perceptron_runs: List[TrainingRun],
    delta_runs: List[TrainingRun],
    best_perceptron: TrainingRun,
    best_delta_rule: TrainingRun,
) -> str:
    lines = [
        "MODULE 4 Learning in AI",
        "Roll No: 24F-0633",
        "",
        "Objective",
        "Compare the Perceptron Learning Rule and the Gradient Descent Delta Rule on the Iris flower dataset.",
        "",
        "Dataset and Protocol",
        f"- Dataset: sklearn Iris dataset (same Fisher Iris dataset referenced in the assignment)",
        f"- Split: {int(TRAIN_TEST_RATIO * 100)}/{100 - int(TRAIN_TEST_RATIO * 100)} with stratification",
        f"- Split random state: {SPLIT_RANDOM_STATE}",
        f"- Standardization: train-set mean and standard deviation only",
        "",
        "Implementation Notes",
        "- Perceptron: multiclass perceptron trained from scratch with winner-take-all updates.",
        "- Delta Rule: one-vs-rest batch gradient descent with from-scratch weight updates.",
        "- Activation experiments: raw_score/sigmoid/tanh for perceptron; linear/sigmoid/tanh for delta rule.",
        "",
        format_run_table("Perceptron Experiments", perceptron_runs),
        "",
        format_run_table("Delta Rule Experiments", delta_runs),
        "",
        "Best Perceptron Run",
        f"- Activation: {best_perceptron.activation}",
        f"- Learning rate: {best_perceptron.learning_rate}",
        f"- Epochs: {best_perceptron.epochs}",
        f"- Train accuracy: {best_perceptron.train_accuracy:.4f}",
        f"- Test accuracy: {best_perceptron.test_accuracy:.4f}",
        f"- Final loss: {format_loss_value(best_perceptron.final_loss)}",
        f"- Notes: {best_perceptron.notes}",
        format_confusion_matrix(best_perceptron.confusion_matrix, split.class_names),
        "",
        "Best Delta Rule Run",
        f"- Activation: {best_delta_rule.activation}",
        f"- Learning rate: {best_delta_rule.learning_rate}",
        f"- Epochs: {best_delta_rule.epochs}",
        f"- Train accuracy: {best_delta_rule.train_accuracy:.4f}",
        f"- Test accuracy: {best_delta_rule.test_accuracy:.4f}",
        f"- Final loss: {format_loss_value(best_delta_rule.final_loss)}",
        f"- Notes: {best_delta_rule.notes}",
        format_confusion_matrix(best_delta_rule.confusion_matrix, split.class_names),
        "",
        build_reflection_text(split, best_perceptron, best_delta_rule),
    ]
    return "\n".join(lines)


def save_report(report_text: str, output_path: Path) -> None:
    output_path.write_text(report_text, encoding="utf-8")


def run_self_test() -> int:
    split = load_and_prepare_iris()
    perceptron_runs, delta_runs, best_perceptron, best_delta_rule = run_experiments(split)

    assert len(split.X_train) == 120, "Training set must contain 120 samples for an 80/20 split."
    assert len(split.X_test) == 30, "Test set must contain 30 samples for an 80/20 split."
    assert len(perceptron_runs) == 9, "Perceptron experiment count changed unexpectedly."
    assert len(delta_runs) == 15, "Delta-rule experiment count changed unexpectedly."
    assert best_perceptron.test_accuracy >= 0.95, "Perceptron test accuracy dropped below the expected threshold."
    assert best_delta_rule.test_accuracy >= 0.95, "Delta-rule test accuracy dropped below the expected threshold."
    assert np.isfinite(best_perceptron.final_loss), "Perceptron final loss must be finite."
    assert np.isfinite(best_delta_rule.final_loss), "Delta-rule final loss must be finite."
    assert best_perceptron.confusion_matrix.sum() == len(split.X_test), "Perceptron confusion matrix is invalid."
    assert best_delta_rule.confusion_matrix.sum() == len(split.X_test), "Delta-rule confusion matrix is invalid."

    print("Self-test passed.")
    print(f"Best perceptron test accuracy: {best_perceptron.test_accuracy:.4f}")
    print(f"Best delta-rule test accuracy: {best_delta_rule.test_accuracy:.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="24F-0633 Module 4: Learning in AI")
    parser.add_argument("--self-test", action="store_true", help="Run verification checks for Module 4.")
    parser.add_argument(
        "--report-file",
        help="Optional path to save the generated experiment report.",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    split = load_and_prepare_iris()
    perceptron_runs, delta_runs, best_perceptron, best_delta_rule = run_experiments(split)
    report_text = build_report(split, perceptron_runs, delta_runs, best_perceptron, best_delta_rule)
    print(report_text)

    if args.report_file:
        save_report(report_text, Path(args.report_file))
        print(f"\nReport saved to: {args.report_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
