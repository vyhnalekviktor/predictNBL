# NBL player prediction
a professional-grade predictive analytics tool designed for the Czech Basketball League (NBL). It utilizes machine learning (XGBoost) to forecast individual player points and calculate the statistical probability of Over/Under betting lines.

> [!WARNING]
> This tool is for analytical and portfolio purposes only. I am not providing the datasets and the training model for gamblers ;)

The system is built on Moneyball principles, focusing on advanced metrics like Usage Rate (USG%), Points Per Shot (PPS), and Rolling Form (L5/L10) rather than just basic seasonal averages.

## Key Features
**Automated Pipeline:** Integrated crawling, data enrichment, and model training.

**Advanced Feature Engineering:** Calculates rolling averages (Last 5/Last 10 games) and performance trends.

**Recency Filtering:** Automatically excludes inactive players (60-day inactivity threshold).

**Probabilistic Inference:** Uses Normal Distribution (Gaussian) curves to calculate exact % odds for any betting line.

**Unified CLI:** Simple command-line interface for all operations.

## Model Performance (Current Build)
**Algorithm:** XGBoost Regressor

**Training Samples:** 19,551 match entries, season 2022/23 - today

**MAE (Mean Absolute Error):** 3.61 points

**R2 Score:** 0.4709 (Explains ~47% of total volatility)

## Installation
1. Clone the repository: `git clone https://github.com/vyhnalekviktor/predictNBL.git; cd predictNBL`

2. Set up a virtual environment: `python -m venv .venv source .venv/bin/activate` # Windows: `.venv\Scripts\activate`

3. Install dependencies: `pip install pandas numpy xgboost scikit-learn matplotlib seaborn joblib scipy`

### CLI Usage
The entire system is controlled via console.py.

Predict a Matchup To see a full list of active players for a match and their probability of outperforming their current form: `python console.py predict "sk slavia praha" "nh ostrava" --home`

Check a Specific Betting Line To calculate the exact % Over/Under for a specific player and point line: `python console.py check "sk slavia praha" "nh ostrava" "Surname" 15.5 --home`

Model Statistics To see the current mathematical state of the model: `python console.py stats`

Betting Guide To display the tactical manual and strategy: `python console.py guide`

## Mathematical Methodology
**Rolling Windows:** The model does not look at season totals. It focuses on the Last 5 (L5) and Last 10 (L10) games to capture hot streaks or slumps. It also calculates a Trend feature (L5 minus L10) to see if a player's role is expanding.

**Probability Calculation:** The system treats the model's prediction as the mean of a normal distribution. The standard deviation (sigma) is derived from the model's MAE (3.61) using the formula: `Sigma = MAE * 1.2533` This allows the tool to calculate the area under the curve (CDF) for any given point line, providing a statistically sound probability.

## Betting Strategy (The MAE Rule)
To maintain a long-term edge over the bookmakers, the system recommends the MAE Buffer Strategy:

**High Confidence:** Only place bets where the <mark>Probability Over/Under is > 65%.

**Market Inefficiency:** Look for a gap between the model prediction and the bookmaker line that is larger than the MAE (> 3.61 points).

**Check Recency:** Verify the LAST PLAYED column to ensure the player isn't returning from a significant injury.

## Project Structure
**CLI_app/:** CLI interface for predicting.

**crawler/:** Scripts for data acquisition from FIBA LiveStats.

**training/:** Train the model.

**sample_nbl_season_data/:** sample data from FIBA via crawler scripts.
