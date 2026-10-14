# Grab App Review Sentiment Analysis (ISP610)

An end-to-end Natural Language Processing (NLP) and text mining pipeline built in **RapidMiner AI Studio (v12.1)** to classify user sentiment across **1,200 Google Play Store reviews** for the **Grab** mobile application.

This project was developed for the **UiTM ISP610 (Data Analytics)** course to benchmark classical machine learning classifiers (**Naive Bayes**, **Support Vector Machine**, and **Random Forest**) on noisy, multilingual user-generated mobile app feedback.

---

## RapidMiner AI Studio Process Workflow

![RapidMiner AI Studio Process Screenshot](./AI%20Studio%20Process%20Screenshot.png)

---

## Project Summary

Mobile super-apps like Grab receive thousands of unstructured daily reviews spanning ride-hailing, food delivery, and digital payment experiences. This project implements a reproducible text preprocessing and supervised classification workflow that transforms raw review text into pruned **TF-IDF** feature vectors (726 stemmed token attributes) and evaluates classifier performance using **10-Fold Cross-Validation**.

---

## Dataset Overview (`grab_reviews_labeled.csv`)

The dataset consists of **1,200 labeled Google Play Store reviews** collected across Grab app versions:

| Metric / Attribute | Details |
| :--- | :--- |
| **Total Records** | 1,200 labeled reviews |
| **Class Distribution** | **Positive:** 745 (62.08%) \| **Negative:** 455 (37.92%) |
| **Star Rating Statistics** | Mean: `3.58` \| Std Dev: `1.82` \| Median: `5.0` \| Min/Max: `1` – `5` |
| **Features Included** | `Review_Text`, `Rating`, `Date_Posted`, `App_Version`, `Helpful_Votes`, `Sentiment` (Target Label) |
| **Data Partitioning** | **95.8% Training / CV Set** (1,150 reviews) \| **4.2% Unlabeled Holdout Test Set** (50 reviews) |

---

## System & Pipeline Architecture

```mermaid
flowchart TD
  subgraph Step1["Step 1: Data Ingestion & Role Setting"]
    A[Read CSV: grab_reviews_labeled.csv] --> B[Nominal to Text: Review_Text]
    B --> C[Filter Examples: Sentiment Not Missing]
    C --> D[Nominal to Binominal & Set Role: Label]
  end

  subgraph Step2["Step 2: NLP Preprocessing (TF-IDF + Absolute Pruning)"]
    D --> E[Tokenize: Non-Letters]
    E --> F[Transform Cases: Lowercase]
    F --> G[Filter Stopwords: English]
    G --> H[Stemming: Porter Algorithm]
    H --> I[Vector Creation: TF-IDF\nPrune Below: 3 / Above: 9999]
  end

  subgraph Step3["Step 3: 10-Fold Cross-Validation Benchmarking"]
    I --> J[Split Data: 95.8% Train / 4.2% Holdout]
    J -->|95.8% Partition| K[Multiply Operator]
    K --> L[10-Fold CV: Naive Bayes\nLaplace Correction]
    K --> M[10-Fold CV: Linear SVM\nDot Kernel, Scale=True]
    K --> N[10-Fold CV: Random Forest\n100 Trees, Max Depth=10, Gain Ratio]
  end

  subgraph Step4["Step 4: Holdout Inference"]
    J -->|4.2% Partition: 50 Samples| O[Apply Trained Model]
    L -->|Trained Model| O
    O --> P[Final Predictions: grab_final_prediction.csv\n36 Positive / 14 Negative]
  end
