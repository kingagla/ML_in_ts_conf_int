import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet


# Objective function for Holt-Winters
def objective_prophet(trial, y, x=None, **prophet_params):

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    changepoint_range = trial.suggest_float('changepoint_range', 0.0, 1.0)
    seasonality_mode = trial.suggest_categorical('seasonality_mode', ["additive", "multiplicative"])

    y_train, y_val = y[:-12], y[-12:]

    model = Prophet(changepoint_range=changepoint_range, seasonality_mode=seasonality_mode)
    model.__dict__.update(prophet_params)

    model = model.fit(y_train, x)
    fh = pd.date_range(y_val.index[0], end=y_val.index[-1], freq='M')
    y_pred = model.predict(fh)

    error = mean_squared_error(y_val, y_pred)

    return error  # An objective value linked with the Trial object.


# Objective function for ARIMA
def objective_arima(trial, y, x=None, **arima_params):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    seasonal = trial.suggest_categorical('seasonal', [True, False])
    stationary = trial.suggest_categorical('stationary', [True, False])

    y_train, y_val = y[:-12], y[-12:]

    model = AutoARIMA(seasonal=seasonal, stationary=stationary)

    model.__dict__.update(arima_params)

    model = model.fit(y_train, x)
    fh = pd.date_range(y_val.index[0], end=y_val.index[-1], freq='M')
    y_pred = model.predict(fh)

    error = mean_squared_error(y_val, y_pred)

    return error


# Objective function for Random-Forest
def objective_rf(trial, y, x=None, **rf_params):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    n_estimators = trial.suggest_int('n_estimators', 20, 300, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 10, log=True)

    x_train, x_val = x[:-12], x[-12:]
    y_train, y_val = y[:-12], y[-12:]

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)

    model.__dict__.update(rf_params)

    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    error = mean_squared_error(y_val, y_pred)

    return error