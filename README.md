## Stock Prediction

This project was built as part of the hiring challenge of `IntelliMind` which predicts the movement of stock price.
Given the data of historical stock prices, predict the direction and potentially magnitude of the move. Overall, this
is an event modeling exercise where each event contains **actual result**, **forecasted result** and **previous result**.

The goal is to predict item `# 7` (log difference between succesive close prices) if on the day we have have `actual`,
`forecast`, `rolling_std`, `z-score` and `7`.


### Input data files

Two data files were provided for the challenge: `n.csv` and `o.csv`. These files were in raw form and needed preprocessing 
steps in order to be suitable for algorithm. All experimentation work was done in *Jupyter Notebooks* which was later
converted to `.py` files for reproducability. 


### Running the model

- In the root directory, type `pip install requirements.txt` followed by `pip install -e .`
- Next step is to train the model on data, lets say on "n.csv". Type `stocks-cmd "n"`. It does the following:
  - Generates processed data (train and test) and saves it directory `data/processed/n`
  - Saves outlier removal vectors in `models/`
  - Trains model on training data
  - Serializes model and stores it in `models/`
  - Performs prediction on test data and returns `RMSE`, `MAE` & `Accuracy`
  
  To perform the same thing on "o.csv", replace `"n"` with `"o"`
- Next step is to generate predictions on new data points for the model trained on a file, lets say "n.csv". 
Type `stocks-cmd ask <actual> <forecast> <rolling-std> <z-score> <YYYY-MM-DD>`. It outputs the value of `#7`
For generating new predictions for model trained on "o.csv", replace `"n"` with `"o"`


### Evaluation metrics

**Mean Absolute Error** and **Mean Squared Error** were the choice of metrics for regression while **Accuracy** was for
classification. For calculating accuracy the sign of the output values (`+` or `-`) were considered.



### EDA, Data Preprocessing & Feature Engineering

Experimentation was performed in the following order:

- Column **6** (volume) was dropped as it wasn't an input parameter to the algorithm
- Univariate feature visualization was performed wherein *seasonality* of columns `actual` and `forecast` were observed (2 years)
- **Causality check** was performed with the help of **Granger's causality tests**. It checks if each of the time series 
in the system influences each other where the null hypothesis is *co-efficients of past values in regression equation is zero*
- **Cointegration test** was performed to establish the presence of statistically significant connection between two or 
more columns
- **Stationarity** of every feature was performed using **Augmented Dickey-Fuller** test
- For benchmarking, `persistence model` was used which simply outputs the previous day's output.
- **Autocorrelation** plots were made for every feature in order to visualize how lags influenced stability
- Time lagged features (by 1 day) were generated for the input variables
- Performed train-test split (default training set size was kept as **80%**)
- Outlier detection and treatment was performed using **Robust Scaler** where outlier range was **(-1.5 IQR, +1.5 IQR)**
- Two new columns `year_even` (`0` if year is even) and `difference` (`actual`-`forecast`) were created after
some experimentation
- For sanity check, **Pearson's correlation co-efficient matrix** was generated

Code for the same can found in `stocks/stocks/preprocess.py` containing the class `DataPreparation`.


### Model Building

Four variants were tried out: **`Linear Regression`**, **`Decision Tree`**, **`Random Forest`** & **`Gradient Boosting`**. 
Results are shown below for both files:

- `*n.csv*`

  | Metrics | Persistence model | Linear Regression | Decision Tree | Random Forest | Gradient Boosting |
  |---|---|---|---|---|---| 
  | MAE | 0.058 | 0.039 | 0.0397 | 0.04 | 0.0375 | 
  | MSE |  0.005 | 0.0025 | 0.0024 | 0.0024 | 0.0022 |
  | Accuracy |  44% | 64.76% | 66.67% | 52.4% | 55.2% |
  
  **Linear Regression** was chosen as the go-to model for this file as it had a good balance of all three metrics. Default
  parameters were good for this model.

- `*o.csv*`
  | Metrics | Persistence model | Linear Regression | Decision Tree | Random Forest | Gradient Boosting |
  |---|---|---|---|---|---| 
  | MAE | 0.036 | 0.016 | 0.018 | 0.0177 | 0.0164 | 
  | MSE |  0.001 | 0.0004 | 0.00056 | 0.00053 | 0.00048 |
  | Accuracy |  45.7% | 62.8% | 58.1% | 53.3% | 64% |
  
  **Gradient Boosting** was chosen as the go-to model for this file as it had a good balance of all three metrics. Its 
  hyperparameters are **`criterion="mae`** & **`random_state=42`**
  
  Both the models can be found in `stocks/model.py` with class names `linearModel` and `gradientModel`.
  
  
  
  
  






