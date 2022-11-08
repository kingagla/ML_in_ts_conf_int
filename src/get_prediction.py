import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, \
    mean_absolute_percentage_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.fbprophet import Prophet as skProphet
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

import objectives
import train_pred

from plots import plot_with_zoom

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
    df_train, df_test = df.iloc[:-n_periods], df.iloc[-n_periods:]

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
    prophet_model = train_pred.train_series(skProphet(), objectives.objective_prophet, 20, y_train)
    arima_model = train_pred.train_series(AutoARIMA(), objectives.objective_arima, 1, y_train, sp=12)

    rf_model_multi = train_pred.train_series(RandomForestRegressor(), objectives.objective_rf, 20, y_train.values,
                                             x=x_train_multi)

    return prophet_model, arima_model, rf_model_multi


def main():
    filepath = '../data/wig20_m.csv'
    start_date = '2013-01-01'
    remove_outliers = True
    predicted_column = 'close' # "#passengers" #

    n_periods = 36
    n_lags = 12
    years_for_trend = 5
    df = read_and_prep_data(filepath, predicted_column, start_date, remove_outliers)

    # to always start prediction in the same point
    if n_periods == 24:
        df = df[:-12]
    elif n_periods == 12:
        df = df[:-24]

    predicted_column = predicted_column+"_adjusted" if remove_outliers else predicted_column

    x_uni_cols = ['trend', 'seasonal', 'lag_12']
    x_multi_cols = ['trend', 'seasonal'] + [f'lag_{i}' for i in range(1, n_lags + 1)]

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
    models = [prophet_model, arima_model, rf_model_multi]  #
    col_names = ['Prophet', 'ARIMA', 'RF_multi']
    for model, col_name in zip(models, col_names):

        prediction = train_pred.predict_series(model, x=x_test_multi, fh=fh)


    #TODO:
    # dorobic inne plotowania - samych wartości + wartosci z przedzialami
    # dorobić przedziały ufności zgodnie z metodologią
    # porównać skuteczność przedziałów z ARIMA i z Prophet
    # zrobic funkcje do predykcji


if __name__ == '__main__':
    main()