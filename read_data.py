import os
import pandas as pd
import numpy as np


def get_relative_path(name, current_folder=""):
    file_name = os.path.abspath(os.getcwd())
    if current_folder != "":
        file_name = os.path.join(file_name, current_folder)
    file_name = os.path.join(file_name, name)
    return file_name


def read_df(name, separator):
    file_name = get_relative_path(name)

    if os.path.exists(file_name):
        df = pd.read_csv(file_name, sep=separator)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print("File {filepath} not found...".format(filepath=file_name))


def read_COVID_data():
    df_names = ['confirmed_global', 'deaths_global', 'recovered_global']
    df_list = [pd.DataFrame() for df in df_names]
    df_dict = dict(zip(df_names, df_list))
    url_part = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_'

    for key, value in df_dict.items():
        value = pd.read_csv(url_part + key + '.csv', parse_dates=[0])

        value.rename(columns={'Province/State': 'Province_State', 'Country/Region': 'Country_Region'}, inplace=True)

        dim_col = value.columns[0:4]
        date_col = value.columns[4:]

        value = value.melt(id_vars=dim_col, value_vars=date_col, var_name='Date', value_name=key)

        value['Date'] = pd.to_datetime(value['Date'])

        df_dict[key] = value

    join_on_col = ['Province_State', 'Country_Region', 'Lat', 'Long', 'Date']
    df_COVID = df_dict['confirmed_global'].merge(df_dict['deaths_global'], on=join_on_col, how='outer').merge(
        df_dict['recovered_global'], on=join_on_col, how='outer')
    df_COVID.rename(
        columns={'confirmed_global': 'Confirmed', 'deaths_global': 'Deaths', 'recovered_global': 'Recovered'},
        inplace=True)
    # to fill the NaN in 'Province_State' columns with Countries name in 'Country_Region'
    df_COVID['Province_State'] = np.where(df_COVID['Province_State'] == 'NaN', df_COVID['Country_Region'],
                                          df_COVID['Province_State'])
    # to fill the NaN in last three columns
    df_COVID.iloc[0:, -4:] = df_COVID.iloc[0:, -4:].fillna(0)

    return df_COVID


def write_df(df, name):
    out_file_name = os.path.join(os.getcwd(), name)
    df.to_csv(out_file_name, index=False)
