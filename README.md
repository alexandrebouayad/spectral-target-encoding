# Spectral Target Encoding

This repository is the official implementation of [Spectral Target
Encoding](https://iclr.cc).

## Prerequisites
Python 3.9

## Set-up

Clone this repository:

```
git clone https://github.com/spectraltargetencoding/iclr-2022.git
```

To install the requirements, cd to the cloned repository, and run the following
command within an empty and isolated virtual environment:

```
pip install -r requirements.txt
```

### Docker

Alternatively, you can use the same Docker image which was used by the authors
to run the evaluations:

```
docker run -it spectraltargetencoding/iclr-2022 bash
```

The image is based upon Docker's official image `python:3.9.7-bullseye`; see the
[Dockerfile](Dockerfile).

## Evaluation

### Statistical benchmarks

Run the following command:

```
python run_benchmark.py stat
```

### ML benchmarks

Run the following commands:

```
python run_benchmark.py ml adult
python run_benchmark.py ml amazon
python run_benchmark.py ml churn
python run_benchmark.py ml churn-cat
python run_benchmark.py ml mortgages
python run_benchmark.py ml toydataset
```

## Results

We ran the benchmarks on a Google Cloud's compute-optimized virtual machine of
type
[c2-standard-4](https://cloud.google.com/compute/docs/compute-optimized-machines).

### Statistical benchmarks

The following table compares the performance of spectral and likelihood
inferences in terms of mean error

|  a / b      | 0.001   | 0.01   | 0.1   | 1     | 10     | 100     |
|-------------|---------|--------|-------|-------|--------|---------|
|   **0.001** | +2%     |        |       |       |        |         |
|   **0.01**  | -3%     | -8%    |       |       |        |         |
|   **0.1**   | +1%     | -20%   | -25%  |       |        |         |
|   **1**     | -22%    | -32%   | -73%  | +3%   |        |         |
|  **10**     | -87%    | -78%   | -94%  | -28%  | -0%    |         |
| **100**     | -99%    | -96%   | -95%  | -92%  | -14%   | +2%     |

The following table gives the multiplicative factor

|  a / b      |   0.001 | 0.01   | 0.1   | 1.0   | 10.0   | 100.0   |
|-------------|---------|--------|-------|-------|--------|---------|
|   **0.001** |    2000 |        |       |       |        |         |
|   **0.01**  |    2000 | 2000   |       |       |        |         |
|   **0.1**   |    2000 | 2000   | 700   |       |        |         |
|   **1**     |    2000 | 2000   | 1000  | 900   |        |         |
|  **10**     |    2000 | 2000   | 1000  | 1000  | 1000   |         |
| **100**     |     800 | 2000   | 2000  | 1000  | 2000   | 2000    |

### ML Benchmarks

- [Encoding times](#encoder)
- [AUC scores - Adult](#pipe-adult)
- [AUC scores - Amazon](#pipe-amazon)
- [AUC scores - Churn](#pipe-churn)
- [AUC scores - Churn (no numerical columns)](#pipe-churn-cat)
- [AUC scores - Mortgages](#pipe-mortgages)
- [AUC scores - ToyDataset](#pipe-toydataset)

Encoder keys:

- SpectralTargetEncoder: Spectral target encoding
- CVSpectralTargetEncoder: Spectral target encoding with cross-validation
- MLE-HBBMEncoder: target encoding with HBBM + likelihood inference
- JamesSteinEncoder:
  [JamesSteinEncoder](https://contrib.scikit-learn.org/category_encoders/jamesstein.html)
- GLMMEncoder:
  [GLMMEncoder](https://contrib.scikit-learn.org/category_encoders/glmm.html)
- TargetEncoder:
  [TargetEncoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
- CVTargetEncoder:
  [TargetEncoder](https://contrib.scikit-learn.org/category_encoders/targetencoder.html) with cross-validation
- TargetRegressorEncoder:
  [TargetRegressorEncoder](https://github.com/scikit-learn/scikit-learn/pull/17323)
- drop: no encoding, categorical columns are dropped

Classifier keys:

- GB:
  [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- LR:
  [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- MLP:
  [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- RF:
  [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

#### <a name="encoder"></a> **Encoding times**

| encoder                |   adult | amazon   |   churn | churn-cat   |   mortgages | toydataset   |
|------------------------|---------|----------|---------|-------------|-------------|--------------|
| SpectralTargetEncoder  |       1 | 1        |       1 | 1           |           1 | 1            |
| MLE-HBBMEncoder        |      18 | 335      |     233 | 657         |          84 | 719          |
| JamesSteinEncoder      |       1 | 1        |       3 | 2           |           1 | 1            |
| GLMMEncoder            |     285 | 77,144   |     372 | 4,692       |         856 | 2,797        |
| TargetEncoder          |       1 | 1        |       3 | 2           |           1 | 1            |
| TargetRegressorEncoder |       2 | 10       |       2 | 4           |           2 | 4            |

#### <a name="pipe-adult"></a> **AUC scores - Adult**

| encoder                 | GB          | LR          | MLP         | RF          |
|-------------------------|-------------|-------------|-------------|-------------|
| SpectralTargetEncoder   | 0.91        | 0.90        | 0.91        | 0.89        |
| CVSpectralTargetEncoder | 0.91 (+0%)  | 0.90 (-0%)  | 0.91 (-0%)  | 0.89 (+0%)  |
| MLE-HBBMEncoder         | 0.91 (+0%)  | 0.90 (-0%)  | 0.91 (-0%)  | 0.89 (+0%)  |
| JamesSteinEncoder       | 0.91 (+0%)  | 0.91 (+0%)  | 0.91 (-0%)  | 0.89 (-0%)  |
| GLMMEncoder             | 0.91 (+0%)  | 0.91 (+0%)  | 0.91 (-0%)  | 0.89 (+0%)  |
| TargetEncoder           | 0.91 (+0%)  | 0.90 (-0%)  | 0.91 (-0%)  | 0.89 (-0%)  |
| CVTargetEncoder         | 0.91 (+0%)  | 0.90 (-0%)  | 0.91 (+0%)  | 0.89 (-0%)  |
| TargetRegressorEncoder  | 0.91 (+0%)  | 0.90 (+0%)  | 0.91 (-0%)  | 0.89 (-0%)  |
| drop                    | 0.53 (-41%) | 0.50 (-45%) | 0.50 (-45%) | 0.60 (-32%) |

#### <a name="pipe-amazon"></a> **AUC scores - Amazon**

| encoder                 | GB          | LR          | MLP         | RF          |
|-------------------------|-------------|-------------|-------------|-------------|
| SpectralTargetEncoder   | 0.84        | 0.85        | 0.83        | 0.81        |
| CVSpectralTargetEncoder | 0.84 (+0%)  | 0.85 (+1%)  | 0.83 (-0%)  | 0.81 (-0%)  |
| MLE-HBBMEncoder         | 0.83 (-1%)  | 0.83 (-1%)  | 0.83 (-0%)  | 0.81 (+0%)  |
| JamesSteinEncoder       | 0.79 (-6%)  | 0.84 (-1%)  | 0.81 (-3%)  | 0.77 (-5%)  |
| GLMMEncoder             | 0.80 (-5%)  | 0.85 (+0%)  | 0.83 (+0%)  | 0.80 (-1%)  |
| TargetEncoder           | 0.80 (-4%)  | 0.84 (-1%)  | 0.83 (-0%)  | 0.80 (-1%)  |
| CVTargetEncoder         | 0.82 (-2%)  | 0.84 (-0%)  | 0.83 (+0%)  | 0.81 (-0%)  |
| TargetRegressorEncoder  | 0.79 (-6%)  | 0.84 (-1%)  | 0.82 (-1%)  | 0.77 (-5%)  |
| drop                    | nan (+nan%) | nan (+nan%) | nan (+nan%) | nan (+nan%) |

#### <a name="pipe-churn"></a> **AUC scores - Churn**

| encoder                 | GB          | LR          | MLP         | RF          |
|-------------------------|-------------|-------------|-------------|-------------|
| SpectralTargetEncoder   | 0.90        | 0.86        | 0.91        | 0.90        |
| CVSpectralTargetEncoder | 0.90 (-0%)  | 0.86 (+0%)  | 0.91 (-0%)  | 0.90 (+0%)  |
| MLE-HBBMEncoder         | 0.90 (-0%)  | 0.86 (+0%)  | 0.91 (+0%)  | 0.90 (-0%)  |
| JamesSteinEncoder       | 0.89 (-1%)  | 0.81 (-5%)  | 0.87 (-4%)  | 0.90 (+0%)  |
| GLMMEncoder             | 0.90 (-0%)  | 0.84 (-2%)  | 0.90 (-1%)  | 0.90 (+0%)  |
| TargetEncoder           | 0.90 (-1%)  | 0.82 (-5%)  | 0.90 (-2%)  | 0.90 (+0%)  |
| CVTargetEncoder         | 0.90 (-0%)  | 0.84 (-2%)  | 0.90 (-1%)  | 0.90 (-0%)  |
| TargetRegressorEncoder  | 0.89 (-1%)  | 0.81 (-5%)  | 0.89 (-3%)  | 0.90 (+0%)  |
| drop                    | 0.73 (-19%) | 0.65 (-24%) | 0.59 (-35%) | 0.72 (-20%) |

#### <a name="pipe-churn-cat"></a> **AUC scores - Churn (no numerical columns)**

| encoder                 | GB          | LR          | MLP         | RF          |
|-------------------------|-------------|-------------|-------------|-------------|
| SpectralTargetEncoder   | 0.68        | 0.74        | 0.66        | 0.65        |
| CVSpectralTargetEncoder | 0.68 (-0%)  | 0.78 (+6%)  | 0.77 (+17%) | 0.65 (+0%)  |
| MLE-HBBMEncoder         | 0.68 (-1%)  | 0.69 (-6%)  | 0.69 (+4%)  | 0.65 (+0%)  |
| JamesSteinEncoder       | 0.67 (-2%)  | 0.61 (-18%) | 0.58 (-13%) | 0.63 (-2%)  |
| GLMMEncoder             | 0.68 (-1%)  | 0.71 (-4%)  | 0.65 (-2%)  | 0.65 (+1%)  |
| TargetEncoder           | 0.65 (-4%)  | 0.63 (-14%) | 0.64 (-4%)  | 0.65 (+0%)  |
| CVTargetEncoder         | 0.65 (-5%)  | 0.65 (-12%) | 0.65 (-2%)  | 0.65 (+0%)  |
| TargetRegressorEncoder  | 0.67 (-2%)  | 0.61 (-17%) | 0.60 (-9%)  | 0.63 (-3%)  |
| drop                    | nan (+nan%) | nan (+nan%) | nan (+nan%) | nan (+nan%) |

#### <a name="pipe-mortgages"></a> **AUC scores - Mortgages**

| encoder                 | GB         | LR          | MLP         | RF         |
|-------------------------|------------|-------------|-------------|------------|
| SpectralTargetEncoder   | 0.68       | 0.65        | 0.68        | 0.66       |
| CVSpectralTargetEncoder | 0.68 (-0%) | 0.65 (-0%)  | 0.68 (+0%)  | 0.66 (-0%) |
| MLE-HBBMEncoder         | 0.68 (-0%) | 0.65 (+0%)  | 0.68 (+0%)  | 0.66 (-0%) |
| JamesSteinEncoder       | 0.66 (-2%) | 0.62 (-5%)  | 0.65 (-4%)  | 0.66 (-1%) |
| GLMMEncoder             | 0.68 (-0%) | 0.65 (-0%)  | 0.68 (-0%)  | 0.66 (-0%) |
| TargetEncoder           | 0.66 (-2%) | 0.63 (-3%)  | 0.67 (-2%)  | 0.66 (-1%) |
| CVTargetEncoder         | 0.67 (-1%) | 0.63 (-3%)  | 0.66 (-2%)  | 0.66 (-1%) |
| TargetRegressorEncoder  | 0.66 (-2%) | 0.63 (-4%)  | 0.66 (-3%)  | 0.66 (-1%) |
| drop                    | 0.67 (-1%) | 0.57 (-12%) | 0.52 (-23%) | 0.65 (-2%) |

#### <a name="pipe-toydataset"></a> **AUC scores - ToyDataset**

| encoder                 | GB          | LR          | MLP         | RF          |
|-------------------------|-------------|-------------|-------------|-------------|
| SpectralTargetEncoder   | 0.58        | 0.62        | 0.59        | 0.58        |
| CVSpectralTargetEncoder | 0.59 (+0%)  | 0.63 (+2%)  | 0.61 (+4%)  | 0.58 (-0%)  |
| MLE-HBBMEncoder         | 0.59 (+0%)  | 0.63 (+2%)  | 0.61 (+3%)  | 0.58 (+0%)  |
| JamesSteinEncoder       | 0.58 (-2%)  | 0.58 (-6%)  | 0.58 (-2%)  | 0.57 (-2%)  |
| GLMMEncoder             | 0.59 (+0%)  | 0.60 (-3%)  | 0.59 (-0%)  | 0.58 (+0%)  |
| TargetEncoder           | 0.57 (-2%)  | 0.58 (-5%)  | 0.59 (-1%)  | 0.57 (-2%)  |
| CVTargetEncoder         | 0.58 (-2%)  | 0.58 (-5%)  | 0.59 (-1%)  | 0.57 (-2%)  |
| TargetRegressorEncoder  | 0.58 (-2%)  | 0.58 (-5%)  | 0.59 (-1%)  | 0.57 (-1%)  |
| drop                    | nan (+nan%) | nan (+nan%) | nan (+nan%) | nan (+nan%) |
