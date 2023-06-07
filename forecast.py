from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
import xgboost as xgb


def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def make_forecast_xgboost(column, train, test):
    train.index = pd.to_datetime(train.index)
    test.index = pd.to_datetime(test.index)

    train_f = create_features(train)
    test_f = create_features(test)
    test_f = test_f.fillna(0)

    train_f['weekofyear'] = train_f['weekofyear'].astype('int')
    test_f['weekofyear'] = test_f['weekofyear'].astype('int')

    FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year', 'dayofmonth', 'weekofyear']
    TARGET = column

    X_train = train_f[FEATURES]
    y_train = train_f[TARGET]

    X_test = test_f[FEATURES]
    y_test = test_f[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    return reg.predict(X_test)


def make_forecast_SARIMAX(column, train, test):
    y = train[column]
    ARMAmodel = SARIMAX(y, order=(5, 0, 2))
    ARMAmodel = ARMAmodel.fit()

    pred = ARMAmodel.get_forecast(len(test.index))
    pred_df = pred.conf_int(alpha=0.05)
    pred_df["Predictions"] = ARMAmodel.predict(start=pred_df.index[0], end=pred_df.index[-1])
    pred_df.index = test.index

    return pred_df


def make_forecast_ARIMA(column, train, test):
    y = train[column]
    ARIMAmodel = ARIMA(y, order=(10, 0, 2))
    ARIMAmodel = ARIMAmodel.fit()

    y_pred = ARIMAmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index

    return y_pred_df


def make_forecast_SARIMAX_seasonal(column, train, test):
    y = train[column]
    SARIMAXmodel = SARIMAX(y, order=(5, 0, 2), seasonal_order=(1, 0, 1, 12))
    SARIMAXmodel = SARIMAXmodel.fit()

    y_pred = SARIMAXmodel.get_forecast(len(test.index))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["Predictions"] = SARIMAXmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df.index = test.index

    return y_pred_df


def make_forecast_Random_Forest_Regressor(column, train, test, max_depth, n_estimators, lags):
    regressor = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
    forecaster = ForecasterAutoreg(
        regressor=regressor,
        lags=lags
    )

    forecaster.fit(y=train[column])
    predictions = forecaster.predict(steps=len(test))
    return predictions


def make_forecast_Lasso(column, train, test, lags):
    steps = len(test)
    forecaster = ForecasterAutoregDirect(
        regressor=Lasso(random_state=123),
        transformer_y=StandardScaler(),
        steps=steps,
        lags=lags
    )
    forecaster.fit(y=train[column])
    predictions = forecaster.predict()
    return predictions


def make_forecast_LinearRegression(column, train, test, lags):
    steps = len(test)
    forecaster = ForecasterAutoreg(
        regressor=LinearRegression(),
        lags=15
    )

    forecaster.fit(y=train[column])

    predictions = forecaster.predict_interval(
        steps=steps,
        interval=[1, 99],
        n_boot=500
    )
    return predictions['pred'], predictions['lower_bound'], predictions['upper_bound']


def make_train_test(df, days):
    dateMinusMonth = datetime.now() - relativedelta(days=days)
    dateMinusMonth = dateMinusMonth.strftime('%Y-%m-%d')

    train = df[df.index <= pd.to_datetime(dateMinusMonth, format='%Y-%m-%d')]
    test = df[df.index > pd.to_datetime(dateMinusMonth, format='%Y-%m-%d')]

    return train, test


def add_more_dates(test, count):
    indexes = []
    values = []
    cur_date = datetime.now()
    cur_date_str = cur_date.strftime('%Y-%m-%d')

    for i in range(0, count):
        values.append(np.nan)
        indexes.append(cur_date_str)
        cur_date = cur_date + relativedelta(days=1)
        cur_date_str = cur_date.strftime('%Y-%m-%d')

    test_add = pd.DataFrame(values, index=indexes)
    test_add.columns = ['Confirmed']
    test_add.index.name = 'Date'

    test = pd.concat([test, test_add])
    return test
