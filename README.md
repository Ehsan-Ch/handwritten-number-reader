# Handwritten Number Reader

Recognize handwritten digits, return prediction sets when the reading is uncertain, and inspect how small image changes affect performance. This project includes a trained neural network, an offline browser drawing demo, a separate calibration stage, and evaluation on the official UCI test writers.

**94.71% digit accuracy on 1,797 official test images.** A cautious policy produces one candidate for 1,638 of those images, with 97.99% accuracy among those suggestions. It requests review for the remaining 159. These are benchmark results, not a guarantee for arbitrary handwriting or documents.

## Try the reader

Download the repository and open [reports/benchmark/report.html](reports/benchmark/report.html) in a browser. No server, API key or GPU is required for the report. Model inference runs locally in JavaScript; drawings are not uploaded.

The report contains:

- Real held-out digit examples and model score bars.
- A drawing pad with blank-input rejection and local preprocessing.
- Standard and cautious prediction-set policies.
- Synthetic four-digit numbers showing the difference between digit and whole-number accuracy.
- An error gallery, confusion matrix, and image-shift stress test.

GitHub's HTML file view shows source. Download the file to use the demo. The report's supporting links work when the whole benchmark directory is downloaded.

## Reproduce the project

Python 3.11 or later is supported. Dependencies are NumPy and scikit-learn. The published benchmark used Python 3.12.14, NumPy 2.5.4 and scikit-learn 1.9.1; versions are recorded in `metrics.json`.

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e .
python -m unittest discover -s tests -v
python -m numberreader.cli download --data data
python -m numberreader.cli run --data data --out reports/my-run
```

Use a fresh data directory and an empty output directory; the commands protect prepared snapshots and existing outputs from replacement. The downloader records archive and source-file SHA-256 hashes; the loader checks those fingerprints before training. Future upstream changes are detectable by comparison with the published metadata, although the download URL is not an immutable versioned endpoint. Retraining on other library versions or platforms may produce numerical differences.

Run saved-model inference on the included 64-pixel example:

```sh
python -m numberreader.cli predict reports/benchmark/sample-input.json --model reports/benchmark/model.json
```

Input is a JSON list of 64 pixel counts in row-major 8 × 8 order, each in the range 0–16, or a list of such rows. Output contains the top digit, candidate set and `SINGLE_CANDIDATE` or `REVIEW` status. This command does not crop text from photographs or segment documents.

## Data and evaluation design

Data: [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), E. Alpaydin and C. Kaynak, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49).

The original preprocessing maps 32 × 32 normalized binary images to 8 × 8 arrays of 4 × 4 block counts. The official training file contains 3,823 samples from 30 writers; the official test file contains 1,797 samples from 13 different writers.

The official test file is preserved intact. The training file is split by label with seed 42:

| Partition | Images | Purpose |
|---|---:|---|
| Fit | 2,293 | Learn candidate models |
| Validation | 765 | Choose the model |
| Calibration | 765 | Set uncertainty cutoffs after selection |
| Official test | 1,797 | Final evaluation on different writers |

Writer IDs are unavailable within the files, so the three internal partitions are image-level splits and can share writers. The official writer separation comes from the dataset documentation. Exact pixel duplicates are checked within and across the two files; none were found. Near duplicates and writer-level clustering are not quantified.

The downloadable split manifest records source filename, zero-based row and partition for every sample. Test labels are never used for fitting, model selection or cutoff calibration. There is no final refit after selection.

## Model selection

Pixel counts are divided by 16. There is no fitted feature scaler, augmentation or per-image contrast normalization. Four candidates use the same fit partition:

| Candidate | Validation accuracy | Validation log loss |
|---|---:|---:|
| Logistic regression, C=1 | 95.42% | 0.1582 |
| Logistic regression, C=10 | 95.56% | 0.1400 |
| Neural network, 32 hidden units | 94.90% | 0.2202 |
| Neural network, 64 hidden units | **95.82%** | 0.1570 |

Selection maximizes validation accuracy, breaking ties by lower log loss. The selected model is a **64 → 64 → 10 multilayer perceptron**, with ReLU hidden activations and softmax output. The neural candidates use Adam, alpha 0.01, seed 42, up to 500 iterations, and early stopping with 20 non-improving iterations. Scikit-learn reserves its default 10% of the fit partition for internal early stopping; the separate model-selection validation partition is not used for gradient fitting or early stopping.

The saved JSON model contains weights, biases, class order, schema version and calibration cutoffs. Python and browser implementations use the same divide-by-16, affine, ReLU and stable-softmax operations. Tests check exported model parity with native scikit-learn predictions; the browser checks six reference score vectors before enabling drawing inference. No executable pickle model is loaded.

## Prediction sets and review

For calibration image i with true class y, the nonconformity score is `1 - p(y | image)`. For error budget alpha and n calibration samples, use the `ceil((n + 1) * (1 - alpha))`-th sorted score. If the rank exceeds n, use the maximal bounded score 1. A digit enters the prediction set when its score is at least `1 - qhat`, including ties.

| Policy | Per-digit alpha | Rank | Minimum included score |
|---|---:|---:|---:|
| Standard digit | 0.05 | 728 | 0.5232482145 |
| Cautious four-digit | 0.0125 | 757 | 0.2047801276 |

Exactly one candidate produces a single-candidate suggestion. An empty or multiple-candidate set requests review. At the standard cutoff, two probabilities cannot both exceed the cutoff, so all 44 review cases in this benchmark have empty sets. The cautious policy admits multiple candidates and requests 159 reviews.

These are split-conformal-style sets. Their nominal marginal coverage requires exchangeability between calibration and future examples, which is not established across writers here. Neither policy guarantees a particular accuracy among single-candidate suggestions or within each digit class. The softmax scores themselves are not independently calibrated probabilities.

## Test results

| Official test metric | Standard | Cautious |
|---|---:|---:|
| Top-digit accuracy | 94.71% | 94.71% |
| True digit included in set | 94.16% | 97.33% |
| Single-candidate images | 1,753 | 1,638 |
| Accuracy among single-candidate images | 96.52% | 97.99% |
| Images needing review | 44 (2.45%) | 159 (8.85%) |

Top-digit accuracy is identical because the policy changes the candidate sets, not the model. Observed coverage is below the nominal 95% and 98.75% targets. This is evidence to report, not a reason to tune on the test set. No writer-cluster confidence intervals are claimed without writer IDs.

The number experiment samples **500 synthetic groups of four isolated test cells**, with replacement and seed 71. It preserves leading zeroes. No detection or segmentation is performed, and groups can reuse images.

| Synthetic four-digit metric | Standard | Cautious |
|---|---:|---:|
| Exact top-prediction number match | 78.60% | 78.60% |
| Single-candidate numbers | 444 / 500 | 329 / 500 |
| Accuracy among those numbers | 87.39% | 95.44% |
| Every true digit included in its set | 77.60% | 91.60% |

The cautious policy allocates alpha = 0.05/4 per digit. A union-bound interpretation would target at least 95% whole-number set coverage if each digit's marginal coverage assumptions held. The observed 91.60% is below that target. These reused, synthetic combinations are not an independent document benchmark, and their metrics should not be given a sample-size interpretation of 500 independent documents.

## Drawing and stress limitations

The drawing pad finds dark pixels, crops their bounding box, scales the longer side to 28 pixels, centers it on a white 32 × 32 canvas, and sums ink density in 4 × 4 blocks. This approximates the dataset representation but does not reproduce the original NIST normalization. Drawing performance is unmeasured. Even a confident-looking result may be wrong. The pad processes one isolated digit and rejects very small or empty marks; it is not an out-of-distribution detector.

A fixed **one-pixel right shift** of the 8 × 8 test grid, dropping the rightmost column, reduces top-digit accuracy from 94.71% to **48.58%**. Under the standard policy, shifted set coverage is only **43.57%**, and single-candidate accuracy is **56.01%**, despite a review rate of 22.20%. The model is sensitive to image alignment; uncertainty filtering alone does not solve distribution shift.

No claim is made about full-page documents, touching digits, postal addresses, handwriting in other scripts, arbitrary photographs or production OCR reliability.

## Tests and structure

**22 local tests pass.** They cover input ranges and source parsing, disjoint reproducible splits, logistic and neural JSON inference parity, calibration-data invariance of model selection, softmax stability, finite-sample quantile ranks, inclusive ties, cautious-set nesting, empty and multiple sets, confidently wrong suggestions, synthetic-number logic and safe report embedding. The GitHub workflow is configured for Python 3.11, 3.12 and 3.13 without downloading data; check Actions for actual run status.

```text
numberreader/data.py       Download, fingerprints, duplicate audit, partitions
numberreader/model.py      Model selection, JSON inference, sets and evaluation
numberreader/pipeline.py   Reproducible benchmark and exports
numberreader/report.py     Offline interactive reader and drawing pad
numberreader/cli.py        Download, run and predict commands
tests/                    Offline behavioral tests
reports/benchmark/        Model, metrics, predictions, splits and report
```

Raw download files and local environments are excluded from Git and the ZIP. The report embeds the official test pixel arrays for offline examples and local inference.

## Attribution and license

Original code is MIT licensed; see [LICENSE](LICENSE). Digit data is **CC BY 4.0** and remains under its original terms, including the test arrays embedded in the report. Attribution: E. Alpaydin and C. Kaynak (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, [DOI 10.24432/C50P49](https://doi.org/10.24432/C50P49). The report renders pixel-count arrays as enlarged grayscale grids and the drawing pad uses new user strokes. The dataset authors do not endorse this project. See [DATA_LICENSE.md](DATA_LICENSE.md).
