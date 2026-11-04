# Does a Smaller Gradient Mean a More Robust Model? Defensive Distillation Revisited

Defensive distillation (Papernot et al., 2016) is reported to shrink a network's input gradients by many orders of
magnitude and thereby resist adversarial examples. This project asks whether that small gradient actually means a
more robust model. It also asks whether the *average gradient norm* is a reliable way to measure robustness.

The study runs on CIFAR-10 first (to validate the method against the literature). It then moves to a biomedical
dataset, PathMNIST (histopathology), where a false sense of robustness would matter most.

> **Status:** the CIFAR-10 temperature sweep is complete. PathMNIST is the next step.

## Research question

When distillation at temperature *T* makes the input gradient small, is the model really harder to attack, or has the
gradient just been hidden (gradient masking) so that gradient-based attacks fail?

## Method

**Defensive distillation (Papernot-style).** For each temperature *T* in {1, 5, 10, 20, 50}:

1. A *teacher* is trained on hard labels with a softmax at temperature *T*.
2. A *student* with the same architecture is trained on the teacher's soft labels `softmax(z_teacher / T)`, also at
   temperature *T*. The soft labels are computed on the fly for each augmented batch.
3. At test time, the student is used at *T* = 1.

The baseline without distillation is the ordinary model (hard labels, *T* = 1). All distilled students are compared
against this same baseline.

**Model and data.** The CNN follows Papernot et al. (4 conv + 2 fully connected layers, dropout 0.5). CIFAR-10 is
split into 45,000 training images, a fixed 5,000-image validation set and the 10,000-image test set. Training uses
crop + flip augmentation.

**Training protocol (identical for every model).**

- SGD with learning rate 0.01 and momentum 0.9, both decayed in 5 evenly spaced steps, 100 epochs.
- No loss scaling and no gradient clipping.
- Paired seeds across temperatures.
- Convergence is checked on the validation set only, with criteria fixed before looking at the results. The test set
  is used once, for the final evaluation.

**Robustness evaluation.** The threat model is L-infinity PGD in [0, 1] pixel space (epsilon = 8/255, step 2/255,
20 iterations, random start). Robust accuracy counts a sample as robust only if no misclassified point is found
anywhere in the epsilon-ball, the unperturbed input included. Two attacks are compared:

- **CE-PGD** follows the cross-entropy gradient. Because this gradient passes through the softmax, it vanishes when the
  softmax saturates.
- **Margin-PGD** follows the logit margin `z_y - max_{j != y} z_j`, as in the Carlini-Wagner loss. It does not pass
  through the softmax.

**Gradient analysis.** For every test sample, three quantities are computed in pixel space:

- the cross-entropy input-gradient norm (L2 and L1);
- the logit-margin input-gradient norm (L2 and L1);
- the scale-invariant linearized distance to the decision boundary, `m(x) / ||grad m(x)||_1`.

All statistics are reported separately for correctly and incorrectly classified samples.

## Results (CIFAR-10, 100 epochs, test set)

| Model | Clean accuracy | Robust accuracy, CE-PGD | Robust accuracy, Margin-PGD |
|---|---|---|---|
| No distillation | 84.49% | 0.18% | 0.00% |
| Distilled, T = 1 | 83.03% | 0.00% | 0.00% |
| Distilled, T = 5 | 83.99% | 10.46% | 0.00% |
| Distilled, T = 10 | 83.04% | 24.50% | 0.00% |
| Distilled, T = 20 | 82.72% | 45.74% | 0.00% |
| Distilled, T = 50 | 81.04% | **63.03%** | **0.02%** |

As T increases, robust accuracy under CE-PGD rises steadily from 0% to 63%. Under Margin-PGD it stays at 0% for
every T, with the same attack budget (epsilon = 8/255, 20 steps). The robustness seen under CE-PGD therefore comes
from softmax saturation (gradient masking), not from a model that is genuinely harder to attack. This agrees with
Carlini & Wagner (2016). The cost in clean accuracy is small: at most 3.5 points (T = 50).

All models passed the pre-registered validation check (converged, no late decline). The one exception is the T = 50
student: it is 3.3 points below the baseline on the validation set, which is above the 2-point threshold. Its
validation accuracy had stopped improving, so this gap is a cost of distillation at high T, not undertraining.

**Gradients.** The table gives medians over the correctly classified test samples, in [0, 1] pixel space.

| Model | CE gradient, L2 | Exactly zero CE gradient | Margin gradient, L2 | Linearized distance d_inf |
|---|---|---|---|---|
| No distillation | 5.1e-1 | 0.0% | 41 | 0.0031 |
| Distilled, T = 1 | 1.3e0 | 0.0% | 21 | 0.0040 |
| Distilled, T = 5 | 1.8e-7 | 3.8% | 117 | 0.0047 |
| Distilled, T = 10 | 4.6e-17 | 36.8% | 220 | 0.0053 |
| Distilled, T = 20 | 0 | 67.1% | 400 | 0.0057 |
| Distilled, T = 50 | 0 | 85.7% | 819 | 0.0062 |

The gradients explain the attack results:

- **The cross-entropy gradient vanishes.** Distillation drives it to zero. At T = 50, it is exactly zero in float32 for
  86% of the correctly classified samples, so CE-PGD has no direction to follow.
- **The model itself is not flatter.** The gradient of the logit margin, which does not pass through the softmax,
  grows about 20-fold.
- **The distance to the boundary barely changes.** The scale-invariant linearized distance only doubles, from 0.0031
  to 0.0062. It stays below epsilon = 8/255 ≈ 0.031 for 100% of the correctly classified samples at every T, which is
  why Margin-PGD succeeds everywhere.
- **The masking only affects correct samples.** On misclassified samples the softmax is not saturated, and the CE
  gradient is large at every T.

A small average input-gradient norm is therefore not evidence of robustness. A gradient norm depends on which quantity
it is taken of: the cross-entropy gradient can vanish while the model is as easy to attack as before. The median is
reported here because the mean is dominated by the few misclassified samples, which have large gradients.

## Repository contents

| Path | Content |
|---|---|
| `network-model.ipynb` | The full pipeline: data, training, validation check, PGD attacks, gradient analysis, figures |
| `project/`, `project.pdf` | Report of the original course project (earlier version of the code, results superseded) |

## Running the notebook

The notebook runs on Kaggle (GPU) or locally. On Kaggle, use *Save & Run All (Commit)* with a GPU; the full sweep takes
about 3 hours. All settings are in the configuration cell (`TEMPERATURES`, `EPOCHS`, `LR_DECAY`, ...).

Every trained model is saved with its full training configuration (`.meta.json`). When the outputs of an earlier run
are attached as an input, matching checkpoints are reloaded instead of retrained, so the evaluation and figures can
be regenerated in minutes. A checkpoint whose stored configuration differs from the current one raises an error
instead of being loaded silently. Figures are exported as vector PDF and 300 dpi PNG.

## Next steps

- Add the CIFAR-10 figures (accuracy, attack success and gradients vs. temperature) to this page.
- Repeat the study on **PathMNIST** (MedMNIST, 9 tissue classes) with the same pipeline, to test whether the
  conclusion carries over to histopathology images.

## References

- N. Papernot, P. McDaniel, X. Wu, S. Jha, A. Swami. *Distillation as a Defense to Adversarial Perturbations against
  Deep Neural Networks.* IEEE S&P 2016.
- N. Carlini, D. Wagner. *Defensive Distillation is Not Robust to Adversarial Examples.* arXiv:1607.04311, 2016.
- A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. *Towards Deep Learning Models Resistant to Adversarial
  Attacks.* ICLR 2018.
- G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop 2014.
- J. Yang et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.*
  Scientific Data 10, 41 (2023).

## Background

This repository started as a course project at the University of Basel ("Study robustness in neural networks in terms
of average gradient norm"). It has since been reworked: per-temperature teachers, a corrected PGD threat model, two
attack losses, scale-invariant gradient diagnostics, and a validation protocol fixed in advance.
