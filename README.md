# Sentiment Analysis Using PySpark

## Overview

This project focuses on performing sentiment analysis on large datasets using PySpark. The goal is to classify text data into sentiments (Positive, Negative, Neutral) and extract valuable insights from it.

## Table of Contents

- [Project Setup](#project-setup)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Results](#results)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Project Setup

1. **Install PySpark**:
    ```bash
    pip install pyspark
    ```

2. **Download Data**:
    Ensure the following data files are available in your working directory:
    - `imdb_reviews_preprocessed.parquet`
    - `sentiments.parquet`
    - `tweets.parquet`

    You can download them from the following links:
    ```bash
    wget https://raw.githubusercontent.com/wewilli1/ist718_data/master/imdb_reviews_preprocessed.parquet
    wget https://raw.githubusercontent.com/wewilli1/ist718_data/master/sentiments.parquet
    wget https://raw.githubusercontent.com/wewilli1/ist718_data/master/tweets.parquet
    ```

3. **Initialize Spark Session**:
    ```python
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("Sentiment Analysis").getOrCreate()
    ```

## Data Cleaning

The data cleaning process involves:
- Removing URLs and punctuation from the text.
- Dropping rows with missing values.
- Handling missing values in specific columns.
- Converting data types for specific columns.

## Feature Engineering

Feature engineering includes:
- Tokenizing text and removing stop words.
- Vectorizing text data using TF-IDF.
- Encoding sentiments to numerical values.
- Aggregating sentiment scores for analysis.

## Model Training

Multiple machine learning models are trained and evaluated:
- Logistic Regression: For predicting sentiments.
- Support Vector Machine (SVM): For predicting sentiments.
- Artificial Neural Network (ANN): For predicting sentiments.

## Results

The performance of the models is evaluated using various metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Visualizations

Several visualizations are created to understand the data and model performance:
- Distribution of sentiment labels.
- ROC Curve for model evaluation.
- Word Cloud of most frequent terms.

## Conclusion

The project demonstrates the use of PySpark for large-scale sentiment analysis, providing insights into the text data and model performance.

