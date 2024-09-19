# Predicting National Team Football Match Outcomes

[![PENGWIN2024TASK1](https://img.shields.io/badge/Deep%20Learning%3A%20Architectures%20%26%20Methods-FOOTBALL%20EM%20PREDICTIONS-blue)](https://github.com/larsteu/dlam)

This repository contains the code for our Deep Learning: Architectures \& Methods project (course by TU Darmstadt). This project aims to predict the outcomes of national team football matches using a novel deep learning approach.
Our final evaluation is on the European Championship 2024 matches.

## Motivation
- Existing work on football match prediction focuses on statistical models [1-5] and league games [1,2].
- **Gap:** Deep learning-based prediction for national team games
- Challenges:
  - Sparse data for national team matches
  - Players from different leagues with varying competitive levels
- Proposed solutions:
  1. Input: Per-player average match statistics from the past year in their national leagues
  2. League-dependent weighting of player statistics
  3. Two-step training process to leverage more data:
     - Step 1: Train on league data
     - Step 2: Fine-tune on national matches

## Model Architecture

Our model consists of two main components:

1. Team Embedder: Processes average player performance to create team embeddings
2. Match Classifier: Takes two team embeddings as input and predicts the match result

### Training Step 1: League Data
![Model Architecture for Training Step 1](Architecture_Step1.png)


### Training Step 2: National Team Data
![Model Architecture for Training Step 2](Architecture_Step2.png)

## Results

- Evaluated on Euro 2024 (51 matches)
- Achieved approximately 59% accuracy
- Slightly outperformed betting odds
- **But:** Needs more evaluation: Euro 2024 only has 51 games.

![Evaluation Results on Euro 2024](FinaL_Evaluation.png)

## Authors
- Paul Andre Seitz
- Leon Arne Engländer
- Lars Damian Teubner
- Daniel Kirn

## References

[1] Moustakidis et al. "Predicting Football Team Performance with Explainable AI: Leveraging SHAP to Identify Key Team-Level Performance Metrics". Future Internet, 2023.  
[2] Choi et al. "Predicting Football Match Outcomes with Machine Learning Approaches". MENDEL, 2023.  
[3] Pinasthika and Fudholi. "World Cup 2022 Knockout Stage Prediction Using Poisson Distribution Model". IJCCS, 2023.  
[4] Schauberger and Groll. "Predicting matches in international football tournaments with random forests". Statistical Modelling, 2018.  
[5] Groll et al. "Prediction of major international soccer tournaments based on team-specific regularized Poisson regression: An application to the FIFA World Cup 2014". Journal of Quantitative Analysis in Sports, 2015.  
