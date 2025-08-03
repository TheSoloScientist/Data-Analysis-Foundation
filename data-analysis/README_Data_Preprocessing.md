
# Data Preprocessing with Visual Diagnostics

A hands-on guide to understanding and applying common data preprocessing techniques using `scikit-learn`, with visual feedback to help you make better decisions.

> Goal: Intuitively understand preprocessing transformations by showing their effects before and after using visualizations and summary statistics.

---

## Project Structure: Techniques Covered & Best Use Cases

| Technique                  | Best Use Case                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| StandardScaler             | When data is normally distributed – centers around 0, unit variance           |
| MinMaxScaler               | When you need values scaled to [0, 1] – useful for neural networks             |
| RobustScaler               | When data contains outliers – uses median and IQR                              |
| Normalizer                 | When comparing vector directions, e.g. cosine similarity, text clustering      |
| Binarizer                  | When you want binary flags for thresholded data                               |
| Label/One-Hot Encoder      | When handling categorical features – label for ordinal, one-hot for nominal   |
| Imputer                    | When dealing with missing data – median strategy shown                        |
| PolynomialFeatures         | When modeling interactions or non-linear effects                              |
| Custom Transformer         | When needing domain-specific transformations                                  |
| Text Vectorizers           | When processing text – Count, TF-IDF, Hashing shown                           |

---

## Visual Diagnostics

Each scaler and transformer includes:
- Histograms and boxplots before vs. after
- .describe() summaries for mean, std, min, max
- Inline explanations of why each transformation is used

---

## Example Code Snippets

### Standard Scaler (for Normal Distributions)

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_scaled = pd.DataFrame(ss.fit_transform(df), columns=df.columns)
```
Use when you want to center data and scale to unit variance.

---

### Robust Scaler (for Outliers)

```python
from sklearn.preprocessing import RobustScaler
robust = RobustScaler()
df_robust = pd.DataFrame(robust.fit_transform(df), columns=df.columns)
```
Use when your data contains outliers.  
Median and IQR are more robust than mean/std.

---

### Normalizer (for Vectors)

```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
df_norm = pd.DataFrame(normalizer.fit_transform(df), columns=df.columns)
```
Use when each row is a vector — e.g., in text mining or clustering.

---

### Text Preprocessing

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer()
hv = HashingVectorizer(n_features=5)
```

| Vectorizer            | Description                                 |
|------------------------|---------------------------------------------|
| CountVectorizer        | Raw token frequency                         |
| TfidfVectorizer        | Term frequency × inverse document frequency |
| HashingVectorizer      | Space-efficient, fast, non-invertible       |

---

## Lessons

- Always inspect your data with:
  - .describe()
  - .info()
  - Graphs and distributions

- Avoid MinMaxScaler when data has outliers — use RobustScaler instead.

- Use OneHotEncoding when your model can’t interpret numeric labels as categories.

---

## How to Run

To run the complete preprocessing and visual diagnostics:

```bash
python data_preprocessing_diagnostics.py
```

Or launch the notebook in Jupyter for an interactive experience.

---

## References

- Scikit-learn Preprocessing Guide: https://scikit-learn.org/stable/modules/preprocessing.html
- Hands-On Machine Learning by Aurélien Géron
- DataCamp: Preprocessing Tutorial
