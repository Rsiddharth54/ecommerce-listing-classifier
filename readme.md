# E-Commerce Listing Quality Classifier

ML classifier to identify low-quality product listings at scale.

## Overview

Built a Random Forest classifier to assess product title quality using 9,280 Amazon listings. Engineered 12 features from text, compared 5 models, and achieved 98% test accuracy (with noted data leakage limitations).

## Dataset

- 9,280 products (mobile + laptop categories)
- 12 engineered features from titles
- 73% high quality, 27% low quality

## Features

**Text Analysis:**
- Length metrics (character count, word count)
- Character patterns (uppercase ratio, digits, special chars)
- Spam indicators (exclamations, all-caps words, spam keywords)
- Quality signals (unique word ratio, capitalization, reviews)

## Models Tested

- Random Forest (98.2% accuracy)
- Gradient Boosting (98.7% accuracy)
- SVM (96.8% accuracy)
- Logistic Regression (76.3% accuracy)
- Decision Tree (98.7% accuracy)

## Results

Best model: Random Forest with 98.2% accuracy, 99.9% ROC-AUC

**Note:** High accuracy due to data leakage (labels created from same features used in training). Real-world deployment would need human-labeled data.

## View Analysis
```bash
# Open the HTML report
open index.html

# Or run the Jupyter notebook
jupyter notebook analysis.ipynb
```

## Files

- `analysis.ipynb` - Full analysis with visualizations
- `index.html` - HTML report
- `src/` - Python scripts for data pipeline
- `data/` - Processed datasets
- `models/` - Trained models
- `outputs/` - Confusion matrix and feature importance plots

## Tech Stack

Python, scikit-learn, pandas, matplotlib, seaborn, Jupyter

## Author

Rishi Siddharth - rs5309a@american.edu
