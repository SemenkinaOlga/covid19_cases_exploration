import dash
import pandas as pd
import numpy as np
from dash import dash_table, dcc, html


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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

join_on_col = ['Province_State','Country_Region','Lat','Long','Date']
df_COVID = df_dict['confirmed_global'].merge(df_dict['deaths_global'], on=join_on_col, how='outer').merge(df_dict['recovered_global'], on=join_on_col, how='outer')
df_COVID.rename(columns={'confirmed_global':'Confirmed', 'deaths_global':'Deaths', 'recovered_global':'Recovered'}, inplace = True)
# to fill the NaN in 'Province_State' columns with Countries name in 'Country_Region'
df_COVID['Province_State'] = np.where(df_COVID['Province_State'] == 'NaN', df_COVID['Country_Region'], df_COVID['Province_State'])
# to fill the NaN in last three columns
df_COVID.iloc[0:,-3:] = df_COVID.iloc[0:,-3:].fillna(0)
print(df_COVID.head())

df = df_COVID[:10]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    children=[
        html.H1(children="COVID19 cases exploration", ),
        html.P(
            children="Analyze the COVID19 cases in Europe",
        ),
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    ]
)
if __name__ == '__main__':
    app.run_server(debug=True)
