# Proving LTH

In this folder we empirically validate the **Lottery Ticket Hypothesis (LTH)** on MNIST using a SimpleMLP architecture.

## What is LTH?

The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) states that a randomly initialized dense neural network contains sparse subnetworks (*winning tickets*) that, when trained in isolation from the same initial weights, can match or exceed the accuracy of the full network.

## Experiment

Each notebook in this folder implements the following pipeline for a given dataset/architecture pair:

1. **Baseline**: Train 10 independently initialized full networks (100% active weights) to establish a reference accuracy.
2. **LTH experiments**: For each active weight percentage, repeat 10 times:
   - Initialize a new `net` with random weights and save the initial state.
   - Train the full network to convergence.
   - Create a global magnitude-based pruning mask on the trained weights.
   - Reset the network to its original random initialization and apply the mask.
   - Retrain the pruned network (maintaining the mask) and evaluate its test accuracy.
3. **Visualization**: Generate a boxplot showing accuracy distributions across all sparsity levels with the baseline (100%) included for comparison.

## Key details

- **Training**: Adam optimizer, 10 epochs per training phase.
- **Pruning**: Global unstructured magnitude pruning — the smallest weights across all layers are zeroed out, and the mask is maintained during retraining via post-update masking.
- **Repetitions**: 10 independent experiments per sparsity level, each with a different random initialization.
- **Model and dataset** details vary per notebook (see each file for specifics).

## Utility functions

The shared `utils.py` module provides:

- `train_model()` — Standard training loop.
- `train_model_with_mask()` — Training loop that reapplies the pruning mask after each optimizer step.
- `evaluate_model()` — Test accuracy evaluation.
- `create_prune_mask()` — Global magnitude-based binary mask generation.
- `prune_model()` — Applies the mask to model weights and registers gradient hooks.

## Results

Each notebook generates a boxplot showing that pruned subnetworks found via LTH consistently match or outperform the full network baseline, even at high sparsity levels — confirming the existence of winning tickets.