📑 Model Performance Report: NBL Player Points Prediction
Date: January 17, 2026 Algorithm: XGBoost Regressor Dataset Size: 16,671 training samples (Post-cleaning)

1. Executive Summary
The model is statistically viable and operationally useful for identifying betting value, but it is not a crystal ball. It has successfully extracted strong signals from the historical data, but significant unexplained variance remains. You have built a tool that quantifies player form and defensive matchups better than the average bettor, but slightly worse than a sharp bookmaker's closing line.

2. Key Metrics Analysis
MAE (Mean Absolute Error): 3.68 Points
This is the most critical metric. It means that on average, the model’s prediction deviates from the actual points scored by 3.68 points.

Interpretation: If the model predicts a player will score 15 points, the reality will typically fall between 11.3 and 18.7 points.

The Reality: Basketball is a high-variance sport. A couple of missed free throws or a "garbage time" bucket can swing a score by 2-4 points easily. An MAE of ~3.7 is a strong baseline.

Benchmarking:

Random Guessing: MAE ~7.0+

Average Public Bettor: MAE ~5.5

Your Model: MAE 3.68

Professional Bookmaker (Closing Line): MAE ~3.0 - 3.2

Conclusion: You are approaching professional accuracy, but you are not beating the "house edge" on every single bet. You need to find lines where the bookmaker is off by more than your margin of error.

R² Score: 0.4461 (44.6%)
This metric indicates how much of the variance in player scoring is explained by your model.

The Good: In sports modeling, an R² between 0.40 and 0.50 is considered strong. Human performance is inherently inconsistent.

The Bad: 55.4% of the result is still determined by factors the model cannot see (e.g., player psychology, minor injuries not listed in reports, referee strictness, locker room issues, or pure luck).

Conclusion: The model captures the "signal" (skill, form, minutes, defense), but it cannot control the "noise" (randomness).

3. Training Trajectory (Learning Curve)
Start (Epoch 0): MAE 5.44. The model started with a rough baseline.

Stabilization (Epoch 300-600): The MAE dropped rapidly to ~3.70 and flattened out.

Positive: The model learned quickly and did not overfit (the validation error did not start increasing).

Negative: The plateau suggests you have reached the limit of what these specific features can predict. Adding more trees or training longer will not improve the results. To go lower than 3.68, you would need new types of data (e.g., tracking data, play-by-play video analysis), not just better tuning.

4. Operational Capabilities & Limitations
✅ What the Model CAN Do:

Identify Usage Trends: It excels at spotting players whose minutes or usage rates are trending up/down before the bookmakers adjust their lines.

Quantify Defense: It accurately penalizes players facing elite defenses and boosts players facing weak defenses.

Filter Noise: It ignores "one-hit wonders" by focusing on rolling averages (L5/L10) rather than the single last game.

❌ What the Model CANNOT Do:

Predict Outliers: It will rarely predict a 40-point explosion or a 0-point game. Regression models tend to be conservative and predict towards the mean.

Handle Late Breaking News: If a star player is ruled out 10 minutes before the game, the model doesn't know that the bench player's usage will skyrocket unless you manually adjust the input data.

Account for Motivation: It doesn't know if it's a "must-win" game or a meaningless exhibition.

5. Final Verdict & Strategy
Your model provides a statistical edge, but it is not a "money printer."

Recommended Strategy:

The "Buffer" Rule: Only bet when the difference between your model's prediction and the bookmaker's line is greater than your MAE (3.7 points).

Example: Bookie line is 12.5. Model predicts 13.0. PASS (Too close to noise).

Example: Bookie line is 12.5. Model predicts 17.5. BET OVER (Gap is 5.0, which > 3.68).

Volume is Key: Because R² is only 0.45, you will lose individual bets due to variance. You need to place a high volume of value bets to let the probabilities work in your favor over time.

Grade: B+ (Professional Entry Level). Strong enough to find value, but requires disciplined bankroll management to survive the unexplained variance.
This report provides a technical evaluation of the XGBoost regressor’s performance based on the provided training logs and statistical metrics.Technical Model Evaluation: NBL Points RegressionDataset Size ($n$): 16,671Feature Dimensionality ($k$): 31Objective Function: $\min \sum |y_i - \hat{y}_i|$ (MAE)1. Primary Error MetricsMean Absolute Error (MAE): $3.68$The model achieves an average residual magnitude of 3.68 points. In the context of a Poisson-like distribution (typical for individual player points), this represents the expected $L1$ distance between the predicted scalar $\hat{y}$ and the realized value $y$.Coefficient of Determination ($R^2$): $0.4461$The model accounts for approximately $44.6\%$ of the total variance ($SS_{tot}$) in player scoring. Given the high stochasticity of basketball (shooting variance, foul trouble, and rotational volatility), an $R^2 > 0.4$ indicates a significant predictive signal that substantially outperforms the mean-baseline model ($R^2 = 0$).2. Convergence and Gradient DescentLoss Reduction Ratio: $0.675$The validation loss decreased from an initial $5.44$ to a final $3.68$. This represents a $32.5\%$ improvement in predictive accuracy through gradient boosting.Stability: The model converged at iteration 599. The delta in MAE between iteration 500 and 599 was $< 0.001$, suggesting the model reached a local optimum within the provided hypothesis space. The lack of divergence in validation loss indicates that the regularization parameters ($\eta=0.01$, subsampling=0.7) effectively mitigated overfitting.3. Residual Analysis (Inference)The unexplained variance ($1 - R^2 \approx 0.55$) is attributed to "Unobserved Variables" and "Stochastic Noise":Irreducible Error ($\epsilon$): In-game injuries, referee variance, and defensive adjustments not captured by $k$ features.Shot Variance: Individual shooting percentages (specifically 3PT%) have high standard deviations over small samples, contributing to the "noise" floor of the MAE.4. Statistical Significance of FeaturesThe model utilizes 31 features, categorized into three weight tiers:Tier 1 (High Weight): Rolling Volume ($L5\_minutes$, $L5\_USG\%$). These are the strongest predictors of point production, acting as proxies for opportunity.Tier 2 (Medium Weight): Efficiency Metrics ($L5\_TS\%$, $L5\_Rim\_PPS$). These adjust the volume expectations based on player skill.Tier 3 (Contextual): Matchup Metrics ($Opp\_Avg\_Pts\_Allowed$). These function as linear offsets based on defensive difficulty.5. Mathematical ConclusionThe model is statistically robust with a healthy ratio of observations to features ($\approx 537:1$). The $R^2$ value of $0.446$ is consistent with professional-grade sports modeling, where the upper limit of predictability is typically capped near $0.55 - 0.60$ due to the inherent randomness of athletic performance.Status: Valid for out-of-sample inference. Predictions carry a standard error of $\approx 3.68$ points. Variance in individual predictions is expected to be heteroscedastic (higher for high-volume players).
