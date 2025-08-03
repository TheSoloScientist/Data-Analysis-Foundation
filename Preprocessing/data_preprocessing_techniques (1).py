"""This code demonstrates various data preprocessing techniques using Python libraries such as pandas, numpy, seaborn, matplotlib, and scikit-learn.
"""
# data_preprocessing.py
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, Binarizer,
    LabelEncoder, OneHotEncoder, PolynomialFeatures, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

def plot_distribution(df, title):
    df.hist(bins=30, figsize=(10, 4))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 2. StandardScaler
# ----------------------------
df = pd.DataFrame({
    'x1': np.random.normal(0, 2, 1000),
    'x2': np.random.normal(5, 3, 1000),
    'x3': np.random.normal(-5, 5, 1000)
})
plot_distribution(df, "Before StandardScaler")
ss = StandardScaler()
data_tf = ss.fit_transform(df)
df_scaled = pd.DataFrame(data_tf, columns=df.columns)
plot_distribution(df_scaled, "After StandardScaler")

# ----------------------------
# 3. MinMaxScaler
# ----------------------------
df = pd.DataFrame({
    'x1': np.random.chisquare(8, 1000),
    'x2': np.random.beta(8, 2, 1000) * 40,
    'x3': np.random.normal(50, 3, 1000)
})
plot_distribution(df, "Before MinMaxScaler")
mm = MinMaxScaler()
data_tf = mm.fit_transform(df)
df_scaled = pd.DataFrame(data_tf, columns=df.columns)
plot_distribution(df_scaled, "After MinMaxScaler")

# ----------------------------
# 4. RobustScaler
# ----------------------------
df = pd.DataFrame({
    'x1': np.concatenate([np.random.normal(20, 1, 1000), np.random.normal(1, 1, 25)]),
    'x2': np.concatenate([np.random.normal(30, 1, 1000), np.random.normal(50, 1, 25)]),
})
plot_distribution(df, "Before RobustScaler")
robustscaler = RobustScaler()
data_tf = robustscaler.fit_transform(df)
df_scaled = pd.DataFrame(data_tf, columns=df.columns)
plot_distribution(df_scaled, "After RobustScaler")

# ----------------------------
# 5. Normalizer
# ----------------------------
df = pd.DataFrame({
    'x1': np.random.randint(-100, 100, 1000).astype(float),
    'y1': np.random.randint(-80, 80, 1000).astype(float),
    'z1': np.random.randint(-150, 150, 1000).astype(float),
})
plot_distribution(df, "Before Normalizer")
normalizer = Normalizer()
data_tf = normalizer.fit_transform(df)
df_normalized = pd.DataFrame(data_tf, columns=df.columns)
plot_distribution(df_normalized, "After Normalizer")

# ----------------------------
# 6. Binarization
# ----------------------------
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
binarizer = Binarizer()
data_tf = binarizer.fit_transform(X)
print("\nOriginal:\n", X)
print("Binarized:\n", data_tf)

# ----------------------------
# 7. Encoding Categorical Values
# ----------------------------
df = pd.DataFrame({
    'Age': [33, 44, 22, 44, 55, 22],
    'Income': ['Low', 'Low', 'High', 'Medium', 'Medium', 'High']
})
df['Income_Encoded'] = df.Income.map({'Low': 1, 'Medium': 2, 'High': 3})
sns.barplot(x='Income', y='Income_Encoded', data=df)
plt.title("Ordinal Encoding Visualization")
plt.show()

df = pd.DataFrame({
    'Age': [33, 44, 22, 44, 55, 22],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male']
})
le = LabelEncoder()
df['gender_tf'] = le.fit_transform(df.Gender)
ohe = OneHotEncoder()
one_hot = ohe.fit_transform(df[['gender_tf']]).toarray()
print("One-Hot Encoding Result:\n", one_hot)

# ----------------------------
# 8. Imputation
# ----------------------------
df = pd.DataFrame({
    'A': [1, 2, 3, 4, np.nan, 7],
    'B': [3, 4, 1, np.nan, 4, 5]
})
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputed_data = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed_data, columns=['A', 'B'])
print("Imputed Data:\n", df_imputed)

# ----------------------------
# 9. Polynomial Features
# ----------------------------
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [2, 3, 4, 5, 6]})
pol = PolynomialFeatures(degree=2)
poly_features = pol.fit_transform(df)
print("Polynomial Features:\n", pd.DataFrame(poly_features, columns=pol.get_feature_names_out()))

# ----------------------------
# 10. Custom Transformer
# ----------------------------
def mapping(x):
    x['Age'] = x['Age'] + 2
    x['Counter'] = x['Counter'] * 2
    return x

df = pd.DataFrame({
    'Age': [33, 44, 22, 44, 55, 22],
    'Counter': [3, 4, 2, 4, 5, 2],
})
customtransformer = FunctionTransformer(mapping, validate=False)
df_transformed = customtransformer.transform(df)
print("Custom Transformed Data:\n", df_transformed)

# ----------------------------
# 11. Text Processing (CountVectorizer, TFIDF, Hashing)
# ----------------------------
corpus = [
    'This is the first document awesome food.',
    'This is the second second document.',
    'And the third one the is mission impossible.',
    'Is this the first document?',
]
df = pd.DataFrame({'Text': corpus})

cv = CountVectorizer()
count_matrix = cv.fit_transform(df.Text).toarray()
print("Count Vectorizer Matrix:\n", count_matrix)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df.Text).toarray()
print("TF-IDF Matrix:\n", tfidf_matrix)

hv = HashingVectorizer(n_features=5)
hashed_matrix = hv.fit_transform(df.Text).toarray()
print("Hashing Vectorizer Matrix:\n", hashed_matrix)