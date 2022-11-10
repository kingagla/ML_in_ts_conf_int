import warnings

import numpy as np
import optuna
import os
import pandas as pd
from prophet import Prophet
from scipy.stats import t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet as skProphet
from statsmodels.tsa.seasonal import seasonal_decompose
import objectives
import train_pred
from pathlib import Path
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def read_and_prep_data(filepath, predicted_column, start_date=None, remove_outliers=True):
    df = pd.read_csv(filepath)
    df.columns = ['date'] + [col.lower() for col in df.columns[1:]]
    if filepath.split('/')[-1] == "Month_Value_1.csv":
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    else:
        df['date'] = pd.to_datetime(df['date'])
    if start_date:
        df = df[df['date'] > pd.to_datetime(start_date)]
    df['month'] = df['date'].apply(lambda x: x.month)
    df.set_index('date', inplace=True)
    df = df.resample("M").mean()

    # remove outliers if necessary
    if remove_outliers:
        adjusted = prophet_adjust(df[predicted_column], interval_width=0.8)['adjusted']
        df = df.merge(adjusted, left_index=True, right_index=True)
        df.rename({"adjusted": predicted_column + "_adjusted"}, axis=1, inplace=True)
        predicted_column = predicted_column + "_adjusted"
    df = df[[predicted_column, 'month']]

    for i in range(1, 13):
        df[f'lag_{i}'] = df[predicted_column].shift(i)

    return df.dropna()


def prophet_adjust(y, interval_width=0.8):
    prophet_mod = Prophet(yearly_seasonality='auto', seasonality_mode='additive',
                          interval_width=interval_width)
    y_proph = y.reset_index()
    y_proph.columns = ['ds', 'y']
    prophet_mod.fit(y_proph)
    pred = prophet_mod.predict(y_proph)[['ds', 'yhat_lower', 'yhat_upper', 'yhat']]
    pred['actual'] = y_proph['y']
    pred['is_outlier'] = (pred['actual'] < pred['yhat_lower']) | (pred['actual'] > pred['yhat_upper'])
    pred['adjusted'] = pred['actual'].mask(pred['is_outlier'], pred['yhat'])
    return pred.set_index('ds').asfreq('M')


def create_train_test_df(df, n_periods, predicted_column, years):
    df_train, df_test = df.iloc[:-n_periods-1], df.iloc[-n_periods-1:]

    result = seasonal_decompose(df_train[predicted_column])
    df_train['seasonal'] = result.seasonal
    df_train['trend'] = result.trend
    df_train['trend'] = df_train['trend'].fillna(df_train['trend'].rolling(13, min_periods=1, center=True).mean())
    model = LinearRegression()
    n = min(df_train.shape[0], 12 * years)
    m = df_test.shape[0]
    model.fit(np.linspace(1, n, n).reshape(-1, 1), df_train['trend'].iloc[-n:])
    df_test['trend'] = model.predict((np.linspace(1, m, m) + n).reshape(-1, 1))
    season = df_train[['month', 'seasonal']].groupby('month').max().reset_index()
    idx = df_test.index
    df_test = df_test.merge(season, on='month').sort_values('trend')
    df_test.set_index(idx, inplace=True)
    ordered_cols = [predicted_column, 'month', 'trend', 'seasonal'] + [f'lag_{i}' for i in range(1, 13)]
    return df_train[ordered_cols], df_test[ordered_cols]


def assign_predictions():
    pass


def train_models(y_train, x_train_multi):
    # tarin models
    prophet_model = train_pred.train_series(skProphet(), objectives.objective_prophet, 5, y_train)
    arima_model = train_pred.train_series(AutoARIMA(), objectives.objective_arima, 1, y_train, sp=12)

    rf_model_multi = train_pred.train_series(RandomForestRegressor(), objectives.objective_rf, 5, y_train.values,
                                             x=x_train_multi)

    return prophet_model, arima_model, rf_model_multi


def get_confidence_intervals(y_train, y_fitted, y_pred, train_horizon, test_horizon, quant=1.96, model_name=""):
    fitted_error = y_train - y_fitted
    model_var = fitted_error.var()
    full_df = pd.DataFrame(index=train_horizon.append(test_horizon))
    full_df.loc[y_fitted.index, "y_fitted"] = y_fitted.values
    full_df.loc[y_fitted.index, 'actuals'] = y_train.values
    full_df.loc[test_horizon, 'forecast'] = y_pred.values

    delta_df = pd.DataFrame(full_df['actuals'].fillna(full_df['forecast']))
    delta_df = delta_df.diff()
    for i in range(1, 7):
        delta_df[f'lag_{i}'] = delta_df['actuals'].shift(i)
    delta_df.dropna(inplace=True)

    delta_df.loc[delta_df.index.isin(y_train.index), 'train_test_split'] = "train"
    delta_df['train_test_split'].fillna("test", inplace=True)

    X_train_delta = delta_df.loc[
        delta_df['train_test_split'] == 'train', [item for item in delta_df.columns if 'lag' in item]]
    X_test_delta = delta_df.loc[
        delta_df['train_test_split'] == 'test', [item for item in delta_df.columns if 'lag' in item]]
    y_train_delta = delta_df.loc[delta_df['train_test_split'] == 'train', 'actuals']
    y_test_delta = delta_df.loc[delta_df['train_test_split'] == 'test', 'actuals']

    rf = RandomForestRegressor()
    rf.fit(X_train_delta, y_train_delta)

    y_pred_delta = rf.predict(X_test_delta)
    delta_error = y_test_delta - y_pred_delta

    full_df['model_var'] = model_var
    full_df['delta_var_cumsum'] = delta_error.expanding().var().cumsum()

    full_df['sigma_h'] = np.sqrt(full_df['model_var'] + full_df['delta_var_cumsum'])
    full_df['sigma_h'].fillna(np.sqrt(model_var), inplace=True)
    full_df['lower'] = full_df['forecast'] - quant * full_df['sigma_h']
    full_df['upper'] = full_df['forecast'] + quant * full_df['sigma_h']

    cols_to_return = ['actuals','forecast', 'lower', 'upper']

    if model_name:
        full_df.columns += f"_{model_name}"

    return full_df[[item + f"_{model_name}" if model_name else item for item in cols_to_return]]

def save_prediction(filepath, dest_path, predicted_column, start_date, n_periods, n_lags, years_for_trend, remove_outliers=True, alpha=0.05):

    df = read_and_prep_data(filepath, predicted_column, start_date, remove_outliers)
    # to always start prediction in the same point
    if n_periods == 24:
        df = df[:-12]
    elif n_periods == 12:
        df = df[:-24]


    x_uni_cols = ['trend', 'seasonal', 'lag_12']
    x_multi_cols = ['trend', 'seasonal'] + [f'lag_{i}' for i in range(1, n_lags + 1)]

    predicted_column = predicted_column + "_adjusted" if remove_outliers else predicted_column
    df_train, df_test = create_train_test_df(df, n_periods, predicted_column, years_for_trend)

    x_train_uni, y_train = df_train[x_uni_cols], df_train[predicted_column]
    x_test_uni, y_test = df_test[x_uni_cols], df_test[predicted_column]
    x_test_uni.iloc[12:, -1] = np.nan

    x_train_multi = df_train[x_multi_cols]
    x_test_multi = df_test[x_multi_cols]
    x_test_multi.loc[1:, [f'lag_{i}' for i in range(1, 13)]] = np.nan
    x_test_multi = x_test_multi.dropna(how='all', axis=1)

    prophet_model, arima_model, rf_model_multi = train_models(y_train, x_train_multi)


    fh = pd.date_range(y_test.index[0], periods=n_periods, freq='M')
    fh_fitted = pd.date_range(y_train.index[0], periods=y_train.shape[0], freq='M')
    n = y_test.shape[0] - 1

    quant = t.ppf(1 - alpha / 2, n)

    models = [prophet_model, arima_model, rf_model_multi]  #
    col_names = ['Prophet', 'ARIMA', 'RF_multi']

    df_preds = pd.DataFrame()
    for i, (model, col_name) in enumerate(zip(models, col_names)):
        y_pred = train_pred.predict_series(model, x=x_test_multi, fh=fh)
        y_fitted = train_pred.predict_series(model, x=x_train_multi, fh=fh_fitted)
        df_pred = get_confidence_intervals(y_train, y_fitted, y_pred, fh_fitted, fh, quant, model_name=col_name)
        if i == 0:
            df_preds = df_pred.copy()
            df_preds.rename({f'actuals_{col_name}': "actuals"}, axis=1, inplace=True)
        else:
            df_preds = df_preds.merge(df_pred.drop(f'actuals_{col_name}', axis=1), left_index=True, right_index=True)

        if col_name in ('Prophet', 'ARIMA'):
            temp_df = model.predict_interval(fh)
            temp_df.columns = [x[2] + f"_{col_name}_original" for x in temp_df.columns]

            df_preds = df_preds.merge(temp_df, left_index=True, right_index=True)


    df_preds.to_csv(dest_path)
    print(dest_path, "Saved successfully")

def main():
    start_date = '2013-01-01'
    remove_outliers = True
    predicted_column = 'close'  # "#passengers" #
    n_periods = 36
    n_lags = 12
    years_for_trend = 5
    alpha = 0.1

    for alpha in [0.1, 0.5, 0.2, 0.05]:
        for file in os.listdir('../data'):
            try:
                filepath = f'../data/{file}'
                dest_path = Path(f'../data_predicted_CI_{int((1-alpha)*100)}/{file}')
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                save_prediction(filepath, dest_path, predicted_column, start_date, n_periods, n_lags,
                                years_for_trend, remove_outliers, alpha)
            except (ValueError, KeyError):
                pass




if __name__ == '__main__':
    main()
