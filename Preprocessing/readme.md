
# Data Preprocessing in Machine Learning

This repository demonstrates common data preprocessing techniques using Python libraries like scikit-learn, pandas, numpy, and skimage. These steps are essential before applying machine learning models.

---
## Table of Contents

1. Introduction
2. Why Preprocessing is Needed
3. StandardScaler
4. MinMaxScaler
5. RobustScaler
6. Normalizer
7. Binarizer
8. Encoding Categorical Features
9. Imputation
10. Polynomial Features
11. Custom Transformer
12. Text Processing
13. Image Processing
14. Summary Table

---

## Introduction

Machine learning models perform better with properly prepared data. Most algorithms expect numeric input, scaled features, and no missing values. This document covers techniques to get data ready.

---

## Why Preprocessing is Needed

- Machine learning algorithms work on numbers, not raw text or images.
- Features with different scales can bias results.
- Missing or incorrect values can cause models to fail.

---

## StandardScaler

Scales data to have a mean of 0 and a standard deviation of 1.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
```
Use Case: Data follows a normal (bell curve) distribution. 
- Useful in algorithms like Logistic Regression or PCA.

Limitation: Not robust to outliers.

When to standardize:
* Dataset features have high variance 
* If data is not **normally distributed**, this is not the best scaler to use.

---

## MinMaxScaler

Scales features to a given range, usually 0 to 1.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
```

Use Case: Neural networks, distance-based models like KNN.

Limitation: A single outlier can stretch the scale.

---

## RobustScaler

Uses median and interquartile range, making it robust to outliers.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(df)
```

Use Case: Data with outliers (e.g., financial data).

Limitation: Not as interpretable as standard scaling.

---

## Normalizer

Scales each data point (row) to unit norm.

```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
normalized_data = scaler.fit_transform(df)
```

Use Case: Text classification, where each row is a word vector.

Limitation: Not useful for normalizing feature columns.

---

## Binarizer

Converts values above a threshold to 1, others to 0.

```python
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0)
binary_data = binarizer.fit_transform(X)
```

Use Case: Binary classification with Bernoulli Naive Bayes.

Limitation: Loss of numeric precision.

---

## Encoding Categorical Features

### Ordinal Encoding

Maps categories with order to integers.

```python
df['Income'] = df['Income'].map({'Low': 1, 'Medium': 2, 'High': 3})
```

Use Case: Categories with a meaningful order like education level.

Limitation: Assumes equal distance between categories.

### One-Hot Encoding

Creates binary columns for each category.

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['Gender']]).toarray()
```

Use Case: Nominal data like gender or country.

Limitation: Increases dataset size.

---

## Imputation

Fills missing values with a strategy like mean or median.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
filled = imputer.fit_transform(df)
```

Use Case: When some values are missing at random.

Limitation: Imputed values are only estimates.

---

## Polynomial Features

Generates additional features by combining features with powers.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
poly_features = poly.fit_transform(df)
```

Use Case: Polynomial regression, non-linear models.

Limitation: Can increase feature count rapidly.

---

## Custom Transformer

Applies a custom function to modify or clean data.

```python
from sklearn.preprocessing import FunctionTransformer

def modify(df):
    df['Age'] += 2
    return df

ft = FunctionTransformer(modify, validate=False)
transformed = ft.transform(df)
```

Use Case: Any custom logic not covered by built-in transformers.

Limitation: Needs to be compatible with sklearn pipelines.

---

## Text Processing

### CountVectorizer

Counts how many times each word appears in documents.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
matrix = cv.fit_transform(corpus).toarray()
```

Use Case: Basic bag-of-words model.

Limitation: Loses word order.

### TfidfVectorizer

Weights words by frequency across all documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus).toarray()
```

Use Case: Document similarity, spam detection.

Limitation: Does not account for word position.

### HashingVectorizer

Uses hashing to reduce memory usage.

```python
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=10)
hashed = hv.fit_transform(corpus).toarray()
```

Use Case: Large-scale or streaming text data.

Limitation: Cannot reverse to original words.

---

## Image Processing

Uses skimage to load and transform images into arrays.

```python
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

img = imread("image.jpg")
gray = rgb2gray(img)
resized = resize(gray, (100, 100))
```

Use Case: Prepare images for machine learning models.

Limitation: Must handle different image sizes and shapes.

---

## Summary Table

| Technique          | Use Case                            | Limitation                        |
|-------------------|--------------------------------------|-----------------------------------|
| StandardScaler     | Normal distribution, PCA, Logistic  | Not robust to outliers            |
| MinMaxScaler       | Neural nets, KNN                    | Sensitive to outliers             |
| RobustScaler       | Financial data, with outliers       | May lose interpretability         |
| Normalizer         | Text or row-normalized data         | Not for feature-wise scaling      |
| Binarizer          | Binary classification               | Loss of numerical detail          |
| One-Hot Encoding   | Gender, country                     | Adds many columns                 |
| Ordinal Encoding   | Education level, ratings            | Assumes linear order              |
| Imputation         | Filling missing values              | Can introduce bias                |
| PolynomialFeatures | Polynomial regression               | May overfit or add too many cols  |
| CountVectorizer    | Text frequency                      | No order or context               |
| TfidfVectorizer    | Document ranking                    | No semantic meaning               |
| HashingVectorizer  | Large datasets                      | Cannot interpret output           |
| skimage            | Image processing                    | Requires manual resizing          |

---

## Modify

- Replace example datasets with your own data.
- Use `Pipeline` and `ColumnTransformer` to combine preprocessing steps efficiently.
- Integrate with classifiers and regression models after preprocessing.
