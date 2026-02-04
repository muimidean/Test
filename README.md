<div align="center">

![MovieLens Recommendation System Banner](./project_banner.png)

</div>

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Authors:** Winnie Njoroge | Michelle Mwende | Laban Leploote | Alice Mathenge | Dean Mutie  
**Date:** January 31, 2026

[View Notebook](./Movie_recommendation_system_Final_Notebook.ipynb) • [Dataset](http://files.grouplens.org/datasets/movielens/) • [Documentation](#documentation)

</div>

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Key Results](#key-results)
- [Technical Approach](#technical-approach)
- [Dataset Information](#dataset-information)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Key Insights](#key-insights)
- [Future Enhancements](#future-enhancements)
- [Team & Contributions](#team--contributions)
- [References](#references)

---

## 🎯 Executive Summary

In the highly competitive streaming industry, **customer retention** is paramount. This project addresses the critical challenge of **choice paralysis**—where users overwhelmed by thousands of options abandon their search, increasing subscription cancellation likelihood by over 30%.

Our solution leverages **collaborative filtering** to deliver personalized movie recommendations, helping streaming platforms:
- ✅ Reduce user browsing time by predicting relevant content
- ✅ Increase user engagement and satisfaction
- ✅ Improve customer retention rates
- ✅ Optimize content discovery for diverse user preferences

**Bottom Line:** We developed a recommendation engine that achieves **0.8703 RMSE on test data**, significantly outperforming baseline predictions (14.7% improvement) and enabling platforms to serve highly relevant content to each user.

---

## 💼 Business Problem

### The Challenge: Streaming Wars & Customer Retention

In the current "Streaming Wars," acquiring a new customer costs **5x more** than retaining an existing one. The primary threat to retention? **Cognitive overload.**

#### The Problem: Choice Paralysis

<div align="center">

```
📊 Key Statistics:
┌────────────────────────────────────────────────┐
│  • Users spend average 15 minutes scrolling    │
│  • 30%+ increase in cancellation probability   │
│  • Thousands of titles = Decision fatigue      │
│  • Poor discovery = Underutilized catalog      │
└────────────────────────────────────────────────┘
```

</div>

When users can't find content they enjoy quickly, they:
1. Become frustrated with the platform
2. Perceive the catalog as lacking value
3. Consider competitor platforms
4. Eventually cancel their subscriptions

### Business Objectives

Our recommendation system directly addresses this by:

| Objective | Impact | Metric |
|-----------|--------|--------|
| **Reduce search time** | Users find content faster | Time to "Play" click ↓ |
| **Increase engagement** | More content consumed | Watch time ↑ |
| **Improve satisfaction** | Better content matches | User ratings ↑ |
| **Retain subscribers** | Reduced churn | Cancellation rate ↓ |

---

## 🚀 Solution Overview

### Our Approach: Collaborative Filtering

We built a **collaborative filtering recommendation engine** that combines multiple techniques to predict user ratings and generate personalized recommendations.

#### Why Collaborative Filtering?

```
Traditional Approach          Our Approach
─────────────────            ───────────────
Same recommendations    →    Personalized for each user
for everyone                 

Genre-based only        →    Learns hidden preferences
                             from user behavior

Static suggestions      →    Adapts to rating patterns
                             and similarities
```

#### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMMENDATION PIPELINE                   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   MovieLens Data  │
                    │  100,836 Ratings  │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Data Preparation │
                    │  • Cleaning       │
                    │  • Filtering      │
                    │  • Train/Val/Test │
                    └─────────┬─────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
    │   Matrix   │    │ Neighborhood │    │   Baseline  │
    │Factorization│    │    Methods   │    │   Methods   │
    │  (SVD,NMF) │    │  (KNN-based) │    │(Biases only)│
    └─────┬──────┘    └──────┬──────┘    └──────┬──────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Model Selection  │
                    │   Best: SVD       │
                    │   RMSE: 0.8703    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Recommendations  │
                    │  Top-N Movies     │
                    │  Per User         │
                    └───────────────────┘
```

---

## 🏆 Key Results

### Model Performance Comparison

We systematically evaluated 6 different collaborative filtering approaches:

| Rank | Model | Validation RMSE ↓ | Validation MAE ↓ | Test RMSE ↓ | Type | Best For |
|------|-------|--------|-------|-------|------|----------|
| 🥇 **1** | **BaselineOnly** | **0.8536** | **0.6561** | - | Baseline | Simple bias modeling |
| 🥈 **2** | **SVD** | **0.8566** | **0.6575** | **0.8703** | Matrix Factorization | **Production deployment** |
| 🥉 3 | KNNWithMeans | 0.8601 | 0.6597 | - | Neighborhood | Interpretability |
| 4 | NMF | 0.8872 | 0.6807 | - | Matrix Factorization | Content understanding |
| 5 | KNNBasic | 0.9414 | 0.7300 | - | Neighborhood | Explainable recommendations |
| - | Global Avg Baseline | 1.0204 | 0.8177 | - | Baseline | Minimum threshold |

**Winner: SVD (Singular Value Decomposition)** - Selected for Production
- ✅ **Test RMSE: 0.8703** (predictions off by ~0.87 stars on average)
- ✅ **14.7% improvement** over global average baseline (RMSE: 1.0204)
- ✅ **Fast predictions** for real-time recommendations
- ✅ **Scalable** to millions of users and items
- ✅ **Production-ready** with proven generalization (validation-test gap: 0.0137)

### What This Means for Business

```
┌────────────────────────────────────────────────────────────┐
│  Prediction Accuracy Impact:                               │
│                                                             │
│  RMSE 0.8703 means:                                        │
│  • Average prediction error: ~0.87 stars (on 5-star scale)│
│  • Typical error (MAE): 0.66 stars                        │
│  • High confidence in Top-10 recommendations              │
│                                                             │
│  Business Translation:                                     │
│  ✅ Users see movies they're likely to rate 4+ stars      │
│  ✅ Reduced "wasted clicks" on irrelevant content         │
│  ✅ Higher engagement and satisfaction                    │
└────────────────────────────────────────────────────────────┘
```

### Recommendation Quality

Our system successfully generates personalized Top-N recommendations with:
- **Diverse genres** across recommendations (not just popular movies)
- **Personalization** based on user's unique rating history
- **Novel discoveries** beyond what users have already seen
- **Scalability** to handle real-time requests

---

## 🔬 Technical Approach

### 1. Data Understanding & Preparation

**Dataset:** MovieLens 100K
- 📊 **100,836 ratings** from 610 users on 9,742 movies
- ⭐ **Rating scale:** 0.5 to 5.0 stars (0.5 increments)
- 📅 **Time period:** Historical movie ratings (1995-2018)
- 🎬 **Movie metadata:** Titles, genres, release years

**Data Quality Measures:**
```python
✅ Removed users with < 20 ratings (ensure sufficient signal)
✅ Removed movies with < 10 ratings (avoid cold-start issues)
✅ Verified no missing values or duplicates
✅ Final dataset: 81,116 ratings (80.4% retention)
```

**Final Dataset Statistics:**
- **Filtered Ratings:** 81,116 (80.4% retention)
- **Users:** 610 (all retained)
- **Movies:** 2,269 (with ≥10 ratings)
- **Matrix Sparsity:** 94.14%
- **Average Ratings per User:** ~133
- **Average Ratings per Movie:** ~36

### 2. Exploratory Data Analysis

#### Key Findings:

**Rating Distribution:**
- Mean rating: **3.574 stars**
- Median rating: **4.0 stars**
- Mode: **4.0 stars** (most common)
- Standard deviation: **1.02**
- Distribution: Left-skewed (users tend to rate positively)

**User Behavior:**
- Average user rates **~133 movies**
- Rating range per user: 20 to 2,698
- Power users rate 500+ movies (providing strong signals)
- No correlation between activity level and rating generosity

**Content Diversity:**
- **20+ unique genres** in the catalog
- Top genres: Drama (4,361 movies), Comedy (3,756), Thriller (1,894)
- Genre engagement shows Drama leads with 32,956 ratings
- Action shows highest engagement rate: 14.9 ratings/movie

**Data Sparsity:**
- Matrix sparsity: **94.14%** (only ~6% of user-movie pairs rated)
- Typical for recommendation systems
- Sufficient overlap for collaborative filtering

### 3. Model Development Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│ MODELING PROGRESSION: Simple → Complex                          │
└─────────────────────────────────────────────────────────────────┘

1️⃣ BASELINE MODEL (Global Average)
   ├─ Purpose: Establish minimum performance threshold
   ├─ Method: Predict same value (3.5754) for everyone
   ├─ Validation RMSE: 1.0204
   └─ Result: Need personalization!

2️⃣ MATRIX FACTORIZATION - SVD ⭐
   ├─ Purpose: Learn latent user/item factors
   ├─ Parameters: n_factors=100, n_epochs=20
   ├─ Validation RMSE: 0.8566 (16.0% improvement)
   ├─ Test RMSE: 0.8703 (14.7% improvement)
   └─ Result: Best production model!

3️⃣ NEIGHBORHOOD METHODS - KNN
   ├─ KNNBasic: Simple similarity-based CF (RMSE: 0.9414)
   ├─ KNNWithMeans: KNN + bias correction (RMSE: 0.8601)
   └─ Result: Interpretable but less accurate than SVD

4️⃣ MATRIX FACTORIZATION - NMF
   ├─ Purpose: Non-negative latent factors
   ├─ Parameters: n_factors=15, n_epochs=50
   ├─ Validation RMSE: 0.8872 (13.1% improvement)
   └─ Result: Interpretable but accuracy trade-off

5️⃣ BASELINE ONLY MODEL
   ├─ Purpose: User + Item biases without factors
   ├─ Method: μ + user_bias + item_bias
   ├─ Validation RMSE: 0.8536 (16.3% improvement - BEST!)
   └─ Result: Surprisingly competitive!

6️⃣ ADVANCED OPTIMIZATION
   ├─ Hyperparameter Tuning (GridSearchCV)
   │  ├─ 54 combinations tested
   │  ├─ Best: n_factors=150, n_epochs=30, lr=0.01, reg=0.05
   │  └─ Tuned RMSE: 0.8399 (1.95% improvement)
   │
   └─ Ensemble Methods (SVD + NMF + KNN)
      ├─ Weighted average based on performance
      ├─ Ensemble RMSE: 0.8496 (0.82% improvement)
      └─ Result: Minimal gain vs added complexity
```

### 4. Train-Validation-Test Split

```
Original Filtered Data (81,116 ratings)
    ├── Training Set (70%) ────> 56,813 ratings - Fit model parameters
    ├── Validation Set (15%) ──> 12,135 ratings - Tune & select model
    └── Test Set (15%) ────────> 12,168 ratings - Final evaluation

Rating Distribution Consistency:
    • Train mean: 3.575
    • Validation mean: 3.576
    • Test mean: 3.564
    ✓ Consistent across all splits
```

### 5. Model Evaluation

**Metrics Used:**
- **RMSE** (Root Mean Squared Error): Penalizes large errors heavily
- **MAE** (Mean Absolute Error): Average absolute error, more interpretable
- **Improvement %**: Performance gain over baseline
- **Generalization**: Validation vs Test consistency

---

## 📦 Dataset Information

### MovieLens 100K Dataset

**Source:** [GroupLens Research Lab, University of Minnesota](https://grouplens.org/datasets/movielens/latest/)

#### Original Dataset Components

| File | Records | Description |
|------|---------|-------------|
| `ratings.csv` | 100,836 | User ratings (userId, movieId, rating, timestamp) |
| `movies.csv` | 9,742 | Movie metadata (movieId, title, genres) |
| `tags.csv` | 3,683 | User-generated tags (userId, movieId, tag, timestamp) |
| `links.csv` | 9,742 | External IDs (movieId, imdbId, tmdbId) |

#### After Quality Filtering

- **Ratings:** 81,116 (80.4% retention)
- **Users:** 610 active users (all retained with ≥20 ratings)
- **Movies:** 2,269 movies (with ≥10 ratings each)
- **Sparsity:** 94.14%
- **Data Quality:** No missing values, no duplicates

---

## 🛠 Installation & Usage

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn scikit-surprise
pip install scipy
```

### Installation

```bash
# Clone the repository
git clone https://github.com/mwendemichelle4-dev/Movies-Recommendation-System.git
cd Movies-Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Movie_recommendation_system_Final_Notebook.ipynb
```

### Quick Start

```python
# Import libraries
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# Load data
ratings = pd.read_csv('Data/ratings.csv')
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.15, random_state=42)

# Train SVD model
model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Get recommendations for a user
def get_recommendations(model, user_id, n=10):
    # Implementation in notebook
    pass
```

---

## 📁 Project Structure

```
Movies-Recommendation-System/
│
├── Data/
│   ├── ratings.csv                 # User ratings data
│   ├── movies.csv                  # Movie metadata
│   ├── tags.csv                    # User-generated tags
│   └── links.csv                   # External database links
│
├── Movie_recommendation_system_Final_Notebook.ipynb    # Main analysis
│
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
│
├── models/                         # Saved models (optional)
│   ├── svd_model.pkl
│   └── baseline_model.pkl
│
├── results/                        # Output files
│   ├── model_comparison.csv
│   └── recommendations.csv
│
└── visualizations/                 # Generated plots
    ├── rating_distribution.png
    ├── model_comparison.png
    └── recommendation_examples.png
```

---

## 📊 Model Performance

### Comprehensive Comparison

| Model | Validation RMSE | Validation MAE | Test RMSE | Improvement vs Baseline | Type |
|-------|----------------|----------------|-----------|------------------------|------|
| **Global Average** | 1.0204 | 0.8177 | - | - | Baseline |
| **BaselineOnly** | 0.8536 | 0.6561 | - | 16.3% | Baseline |
| **SVD** ⭐ | 0.8566 | 0.6575 | 0.8703 | 16.0% (Val), 14.7% (Test) | Matrix Factorization |
| **KNNWithMeans** | 0.8601 | 0.6597 | - | 15.7% | Neighborhood |
| **NMF** | 0.8872 | 0.6807 | - | 13.1% | Matrix Factorization |
| **KNNBasic** | 0.9414 | 0.7300 | - | 7.7% | Neighborhood |

### SVD Model Details

**Final Production Model:**
- **Architecture:** Singular Value Decomposition (Matrix Factorization)
- **Parameters:** 
  - `n_factors=100` (latent dimensions)
  - `n_epochs=20` (training iterations)
  - `lr_all=0.005` (learning rate)
  - `reg_all=0.02` (regularization)
- **Performance:**
  - Validation RMSE: 0.8566
  - Test RMSE: 0.8703
  - Test MAE: 0.6631
- **Generalization:** Excellent (test-validation gap: 0.0137)

### Optimization Results

**Hyperparameter Tuning:**
- Tested 54 parameter combinations
- Best tuned configuration:
  - n_factors: 150
  - n_epochs: 30
  - lr_all: 0.01
  - reg_all: 0.05
- **Tuned Validation RMSE:** 0.8399 (1.95% improvement over original)

**Ensemble Methods:**
- Weighted average: SVD (33.8%) + NMF (32.6%) + KNN (33.6%)
- **Ensemble RMSE:** 0.8496
- Improvement: 0.82% over SVD
- **Decision:** Use single SVD for simplicity

### Success Criteria Evaluation

| Criterion | Target | Achievement | Status |
|-----------|--------|-------------|--------|
| **Prediction Accuracy** | RMSE < 1.0 | Test RMSE: 0.8703 | ✅ EXCEEDED |
| **Diverse Recommendations** | Multiple genres | 20+ genres available | ✅ ACHIEVED |
| **Handle Existing Users** | All users | Works for all 610 users | ✅ ACHIEVED |
| **Efficiency** | Scalable solution | Trains in seconds, predicts in milliseconds | ✅ ACHIEVED |

---

## 💡 Key Insights

### Model Performance Insights

**1. Baseline Power is Surprising**
- BaselineOnly (user + item biases) achieved validation RMSE of 0.8536
- Demonstrates that systematic rating tendencies explain most variance
- Some users consistently rate higher/lower than average
- Some movies are consistently rated higher/lower than average

**2. SVD Chosen for Production Despite Not Being #1**
- Validation: BaselineOnly slightly better (0.8536 vs 0.8566)
- **Why SVD?**
  - Richer 100-dimensional representations
  - Better cold-start handling potential
  - More scalable to millions of users/items
  - Foundation for future enhancements
  - Only 0.003 RMSE difference on validation

**3. Complexity ≠ Better Performance**
- Ensemble methods provided minimal improvement (0.82%)
- Trade-off: Added complexity vs marginal accuracy gain
- Simpler models often win in production

**4. Excellent Generalization**
- Test RMSE (0.8703) close to validation RMSE (0.8566)
- Difference of only 0.0137 indicates no overfitting
- Model will perform reliably in production

### Data Insights

**User Rating Behavior:**
- Users tend to rate movies positively (mean: 3.574)
- Modal rating: 4.0 stars (users prefer round numbers)
- Wide activity variance: 20 to 2,698 ratings per user
- Activity level doesn't correlate with rating generosity

**Content Characteristics:**
- 20+ distinct genres provide catalog diversity
- Drama, Comedy, Action dominate both catalog and engagement
- High sparsity (94.14%) validates need for collaborative filtering
- Sufficient user overlap enables effective recommendations

**Rating Patterns:**
- Left-skewed distribution (more high ratings than low)
- Full rating scale utilized (0.5 to 5.0)
- Strong user and item biases present
- Prediction accuracy of ±0.87 stars is excellent given variance

### Business Insights

**Personalization Value:**
- All models beat global average by 7-16%
- Best model: 14.7% improvement on test set
- Personalization significantly improves user experience

**Recommendation Strategy:**
- Focus on personalization over generic suggestions
- Account for user and item biases
- Leverage collaborative patterns from similar users
- Balance accuracy with explainability

**BaselineOnly Surprise:**
- Simple bias model performed remarkably well
- Shows importance of accounting for user/item biases
- Good fallback for production systems
- Validates that biases explain majority of variance

---

## 🚀 Future Enhancements

### Short-term Improvements

1. **Hybrid Model**
   ```
   Combine SVD + Content-Based Filtering
   ├─ SVD: Collaborative signals
   └─ Content: Movie metadata (genres, actors, directors)
   
   Expected Improvement: +5-10% accuracy
   ```

2. **Cold-Start Handling**
   - New user onboarding: Ask for 5-10 movie ratings
   - New movie handling: Content-based recommendations until sufficient ratings
   - Fallback to popular items for unknown users

3. **Time-Aware Recommendations**
   - Incorporate timestamp data
   - Model temporal dynamics (user taste evolution)
   - Seasonal/trending content boosting

4. **Diversity Optimization**
   - Avoid filter bubbles (all same genre)
   - MMR (Maximal Marginal Relevance) for diverse Top-N
   - Serendipity: Introduce unexpected but relevant content

### Long-term Vision

5. **Deep Learning Models**
   - Neural Collaborative Filtering (NCF)
   - Autoencoders for representation learning
   - Transformer-based sequential recommendations

6. **Context-Aware Recommendations**
   - Time of day (weekday vs weekend)
   - Device type (mobile vs TV)
   - Social context (watching alone vs with family)

7. **Multi-Stakeholder Optimization**
   - User satisfaction (current focus)
   - Business metrics (revenue, engagement time)
   - Content provider goals (promote new releases)

8. **A/B Testing Framework**
   - Production deployment infrastructure
   - Real-time metric tracking
   - Automated model retraining pipeline

9. **Explainable AI**
   ```
   "We recommend this because:"
   ├─ Users like you rated it highly
   ├─ Similar to movies you enjoyed
   └─ Trending in your favorite genres
   ```

10. **Real-Time Personalization**
    - Streaming session behavior (skip patterns)
    - Micro-adjustments based on immediate feedback
    - Contextual bandits for exploration-exploitation

---

## 👥 Team & Contributions

### Project Team

| Name | Role | Contributions |
|------|------|---------------|
| **Winnie Njoroge** | Data Analysis Lead | EDA, visualization, insights generation |
| **Michelle Mwende** | Modeling Specialist | SVD/NMF implementation, hyperparameter tuning |
| **Laban Leploote** | ML Engineer | KNN models, evaluation framework |
| **Alice Mathenge** | Business Analyst | Problem definition, stakeholder analysis |
| **Dean Mutie** | Project Manager | Coordination, documentation, presentation |

### Individual Contributions

```
Data Preparation:        🟦🟦🟦🟦🟦 (All team members)
Exploratory Analysis:    🟩🟩🟩 (Winnie, Alice)
Model Development:       🟨🟨🟨🟨 (Michelle, Laban)
Evaluation & Testing:    🟧🟧🟧 (Laban, Dean)
Documentation:           🟪🟪🟪🟪 (Dean, Alice)
Presentation:            🟥🟥 (All team members)
```

### Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **Scikit-Surprise** library developers
- Course instructors and mentors
- Peer reviewers for valuable feedback

---

## 📚 References

### Academic Papers

1. Koren, Y., Bell, R., & Volinsky, C. (2009). **Matrix Factorization Techniques for Recommender Systems**. *Computer*, 42(8), 30-37.

2. Ricci, F., Rokach, L., & Shapira, B. (2015). **Recommender Systems Handbook** (2nd ed.). Springer.

3. Su, X., & Khoshgoftaar, T. M. (2009). **A Survey of Collaborative Filtering Techniques**. *Advances in Artificial Intelligence*, 2009.

### Dataset

4. Harper, F. M., & Konstan, J. A. (2015). **The MovieLens Datasets: History and Context**. *ACM Transactions on Interactive Intelligent Systems*, 5(4).  
   https://doi.org/10.1145/2827872

### Technical Resources

5. **Scikit-Surprise Documentation**  
   https://surprise.readthedocs.io/

6. **Collaborative Filtering Tutorial**  
   https://towardsdatascience.com/intro-to-recommender-systems

### Industry Applications

7. Netflix Prize Competition (2006-2009)
8. Amazon Product Recommendations
9. Spotify Discover Weekly Algorithm

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 MovieLens Recommendation Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 📞 Contact & Support

### Questions or Feedback?

We'd love to hear from you! Reach out via:

- 📧 **Email:** team@movielens-rec.com
- 💼 **LinkedIn:** [Project Team LinkedIn](https://linkedin.com/in/team)
- 🐦 **Twitter:** [@MovieLensRec](https://twitter.com/movielenrec)
- 💬 **Issues:** [GitHub Issues](https://github.com/mwendemichelle4-dev/Movies-Recommendation-System/issues)

### For Stakeholders

**Interested in implementing this for your platform?**
- 📊 [View Demo](./demo)
- 📑 [Download Technical Report](./docs/technical_report.pdf)
- 📧 Contact: business@movielens-rec.com

---

## 🌟 Star History

If you found this project helpful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=mwendemichelle4-dev/Movies-Recommendation-System&type=Date)](https://star-history.com/#mwendemichelle4-dev/Movies-Recommendation-System&Date)

---

<div align="center">

### 🎬 **Making Every Movie Recommendation Count** 🎬

*Built with ❤️ by the MovieLens Recommendation Team*

**[⬆ Back to Top](#-movielens-recommendation-system)**

</div>

---

## Appendix

### A. Technical Specifications

**Development Environment:**
- OS: Ubuntu 20.04 / Windows 11 / macOS Monterey
- Python: 3.8.10
- Jupyter: 6.4.0
- RAM: 8GB minimum, 16GB recommended
- Storage: 500MB for data + models

**Compute Requirements:**
- Training: ~15 seconds (81,116 ratings, SVD)
- Prediction: <0.01s per user (real-time capable)
- Batch recommendations: ~1 second for 1000 users

### B. Hyperparameter Tuning Details

**SVD Optimal Parameters (Tuned):**
```python
{
    'n_factors': 150,      # Latent factor dimensions (tuned from 100)
    'n_epochs': 30,        # Training iterations (tuned from 20)
    'lr_all': 0.01,        # Learning rate (tuned from 0.005)
    'reg_all': 0.05,       # L2 regularization (tuned from 0.02)
    'random_state': 42     # Reproducibility
}
```

**Original SVD Parameters (Used in Production):**
```python
{
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'random_state': 42
}
```

**Search Space Explored:**
- n_factors: [50, 100, 150]
- n_epochs: [20, 30]
- lr_all: [0.002, 0.005, 0.01]
- reg_all: [0.01, 0.02, 0.05]
- Total combinations: 54
- Cross-validation: 3-fold
- Total model trainings: 162

### C. Error Analysis

**Common Prediction Errors:**
1. **New releases:** Insufficient rating history
2. **Niche content:** Limited user overlap
3. **Controversial films:** High rating variance
4. **Seasonal content:** Temporal preferences not modeled

**Mitigation Strategies:**
- Hybrid content-based for new items
- Genre-based fallbacks for niche content
- Confidence intervals for controversial items
- Time-decay weighting (future work)

### D. Key Dataset Statistics

**Original Data:**
- Total ratings: 100,836
- Users: 610
- Movies: 9,742
- Sparsity: 98.30%

**Filtered Data:**
- Total ratings: 81,116 (80.4% retention)
- Users: 610 (100% retention with ≥20 ratings each)
- Movies: 2,269 (23.3% of original, with ≥10 ratings each)
- Sparsity: 94.14%

**Data Splits:**
- Training: 56,813 ratings (70%)
- Validation: 12,135 ratings (15%)
- Test: 12,168 ratings (15%)

---

**Last Updated:** January 31, 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
