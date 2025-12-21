# Payment Method Classification (ML + ANN)

This project predicts customer payment method (Credit Card, Debit Card, PayPal) using leakage-free numerical retail transaction features. We compare multinomial logistic regression and a simple artificial neural network (ANN) on a curated dataset of 240 online transactions.

## Overview
All categorical variables with deterministic mappings (product category, product name, region, date) were removed to prevent data leakage. The final feature set includes:

- Unit Price  
- Units Sold  
- Total Revenue  
- Month  
- HighRevenue (binary)  
- PriceTier (Low / Mid / High)

Features were standardized before modeling.

## Methods
- **Multinomial Logistic Regression**
  - 10-fold stratified cross-validation
  - Test accuracy ≈ **0.67**
- **Artificial Neural Network**
  - 10-fold cross-validation
  - Hyperparameter Tuning
  - Test accuracy ≈ **0.63**
- **Bootstrap Resampling (1000 iterations)**
  - Stable accuracy range: **0.60–0.70**
  - Coefficients show weak but consistent numerical signal

## Key Findings
- Numerical features alone provide limited predictive capability.
- Logistic regression performs best; ANN does not uncover additional structure.
- Bootstrap analysis confirms model stability—performance limits stem from feature constraints, not model variance.

## Team Members
- Saisrijith Reddy Maramreddy  
- Seth Kaufman  
- Edvin Leon  
- Alexander Mendez  


