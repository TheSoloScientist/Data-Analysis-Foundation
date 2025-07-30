# Mastering Data Analysis Before Machine Learning

> **"Better data beats fancier algorithms." — Peter Norvig**

# Project Vision

This repository is dedicated to building *strong, reproducible* data analysis workflows on **diverse datasets** — before jumping into ML or complex modeling.

This repository is my learning space where I explore how to work with different types of data and practice proper data analysis techniques.

By focusing on **Financial data**, **Bioinformatics data**, and **General-purpose data**, this project aims to demonstrate **why data analysis is not just the first step — it's the foundation**.
And to demonstrate a foundational application of data analysis to modelling (covered in classical-ml-and-data-analysis)[]

---

## Why Focus on Data Analysis First?

 1. Clean Data Makes a Difference
I've learned that messy or misunderstood data leads to weak models. Without cleaning, models can:
- Give inaccurate results
- Mislead decision-making
> Important: Missing values, duplicates, wrong data types, and outliers must be fixed early to avoid misleading results later.

2. Data Tells a Story
Before any prediction, we should ask:
- What trends exist?
- Are there anomalies?
- What features actually matter?
> Residual plots and charts are powerful tools. They help visualize patterns and errors, giving insight into how well a model is capturing relationships.

3. Some Fields Need **Trust**
Fields like bioinformatics and finance
These domains demand:
- **Transparency**
- **Statistical robustness**
- **Reproducibility**

> Reminder: **You can’t trust machine learning unless you trust the data first**.

--- 


## 🧰 Data Domains Covered

Bioinformatics Data:
- Gene expressions, genetic variations, and alignment data
- Importance: Domain-specific cleaning (e.g., normalization, filtering low-expression genes)
I’m learning how to normalize and filter this kind of data

Financial Data:
- Stock prices, volumes, and economic indicators
Macroeconomic indicators
- Importance: Handling missing data, smoothing noise, identifying seasonality
I’m practicing how to clean and analyze trends over time

General Data:
- Surveys, time series, public datasets
- Weather, transport, health indicators
> Importance: Universal EDA patterns, pattern discovery, feature correlation


---

## Core Skills

| Area                    | Tools/Skills                                  |
|-------------------------|-----------------------------------------------|
| Data Cleaning           | `pandas`, `numpy`, missing value handling     |
| Exploratory Data Analysis (EDA) | histograms, correlation heatmaps, group analysis |
| Feature Engineering     | transformations, scaling, one-hot encoding    |
| Domain Knowledge Mapping| Applying context to patterns in data          |
| Visualization           | `matplotlib`, `seaborn`, `plotly`             |

---## ⚙️ Optimizations & Best Practices
1. Cloud Platform Integration:
For scaling and automating data workflows, I'm learning to explore cloud-based tools:
- AWS
- AZURE
- GCP 

2.  Bias-Variance Trade-off:
High Bias means the model is too simple to capture a pattern in data.
High variance indicates a model that is too complex for the data and  results in poor generalization to new, unseen data. 
And High variance as result of [Data leakage](https://airbyte.com/data-engineering-resources/what-is-data-leakage).
- High Bias: Add more features, try complex models
- High Variance: Use regularization, cross-validation, add more data

The **issue** is handling one would increase the other, meaning we need to find a perfect balance to minimize errors

> Residual plots help spot both. Random scatter = good fit. 

