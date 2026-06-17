# Single-Cell Annotation Using Machine Learning

A Python machine learning pipeline for classifying cell identities from qualitative and quantitative gene-expression data.

## Overview

This project explores cell annotation using a combination of biological marker rules and machine learning.

The pipeline:
- Processes gene-expression measurements from 10,000 cells
- Applies rule-based cell classification using known marker definitions
- Trains Random Forest models to improve annotation coverage
- Uses confidence-based prediction filtering and cross-validation

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

## Methods

### Stage 1: Rule-Based Classification
Cells are assigned identities using known biological marker definitions.

### Stage 2: Machine Learning
Random Forest models are trained on confidently labeled cells using quantitative gene-expression features.

### Stage 3: Prediction Refinement
Confidence thresholds are applied to improve annotation precision and reduce ambiguous classifications.

## Skills Demonstrated

- Machine Learning
- Data Analysis
- Feature Engineering
- Computational Biology
- Classification Algorithms
- Scientific Computing

## Author

Rodas Daniel
USC Biomedical Engineering (Electrical Engineering Emphasis)
