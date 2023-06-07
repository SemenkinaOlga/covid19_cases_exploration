import math
import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from dateutil.relativedelta import relativedelta
import codecs

import forecast
import mapping
import preprocess as pp
import read_data as rd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df_country_region = rd.read_df('df_country_region.csv', ',')
print("Country region data has been read")

df_COVID = rd.read_COVID_data()
print("COVID data has been read")

df_COVID = pp.merge_COVID_and_region_data(df_COVID, df_country_region)
print("COVID data has been merged")

country_names = sorted(df_COVID["Country_Region"].unique())
dict_df_country = pp.df_to_dict(country_names, df_COVID, 'Country_Region')

mesoregions = sorted(df_COVID['meso_region'].unique())
dict_df_mesoregion = pp.df_to_dict(mesoregions, df_COVID, 'meso_region')

macroregions = sorted(df_COVID["macro_region"].unique())
dict_df_macroregion = pp.df_to_dict(macroregions, df_COVID, 'macro_region')

df_world = df_COVID.groupby('Date').agg(
    {'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()
df_world = df_world.sort_values(['Date'], ascending=True)
df_world['Confirmed'] = df_world['Confirmed'] - df_world['Confirmed'].shift()
df_world['Confirmed'] = np.where(df_world['Confirmed'] <= 0, df_world['Confirmed'].shift(), df_world['Confirmed'])
df_world['Deaths_pure'] = df_world['Deaths'] - df_world['Deaths'].shift()
df_world['Deaths_pure'] = np.where(df_world['Deaths_pure'] <= 0,
                                   df_world['Deaths_pure'].shift(), df_world['Deaths_pure'])

df_COVID_summary = df_COVID.groupby(['Country_Region', 'meso_region', 'macro_region', 'code']).agg(
    {'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'}).reset_index()
mapping.update_coordinates(df_COVID_summary)
print("coordinates has been updated")

total_cases = math.floor(df_COVID_summary[df_COVID_summary['Country_Region'] == 'Bulgaria']['Confirmed'])
total_death = math.floor(df_COVID_summary[df_COVID_summary['Country_Region'] == 'Bulgaria']['Deaths'])

print("COVID data")
print(df_COVID.head())

init_df_for_map = df_COVID[df_COVID['Country_Region'] == 'Bulgaria'].groupby(['Country_Region', 'meso_region',
                                                                              'macro_region', 'code']).agg(
    {'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).reset_index()
init_df_for_map.head()
init_map = mapping.make_map(df_COVID_summary)
init_map.save(rd.get_relative_path("init_map.html"))


def delete_tile_for_map(file_name):
    f = codecs.open(file_name, 'r')
    html_str = f.read()
    txt_to_rep = "\\u0026copy; \\u003ca href=\\\"http://www.openstreetmap.org/copyright\\\"\\u003eOpenStreetMap\\u003c/a\\u003e contributors \\u0026copy; \\u003ca href=\\\"http://cartodb.com/attributions\\\"\\u003eCartoDB\\u003c/a\\u003e, CartoDB \\u003ca href =\\\"http://cartodb.com/attributions\\\"\\u003eattributions\\u003c/a\\u003e"
    html_str = html_str.replace(txt_to_rep, "")
    Html_file = open(file_name, "w")
    Html_file.write(html_str)
    Html_file.close()


delete_tile_for_map("init_map.html")

df_table = df_COVID[:10]

end_date = datetime.now() - relativedelta(days=1)
start_date = end_date - relativedelta(days=365)
min_date_allowed = min(df_COVID['Date'])
max_date_allowed = max(df_COVID['Date'])

fig_init = go.Figure()
fig_init.update_layout(title='', plot_bgcolor='black', template="plotly_dark", height=350, font_family="Georgia",
                       title_font_family="Georgia")
fig_init.update_xaxes(title_text='Date', gridcolor='black',
                      zeroline=True, zerolinewidth=2, zerolinecolor='gray')
fig_init.update_yaxes(title_text='Cases', showgrid=True, gridwidth=1, gridcolor='gray',
                      zeroline=True, zerolinewidth=2, zerolinecolor='gray')

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    children=[
        html.H2(children="COVID19 cases exploration", style={'textAlign': 'center', "color": "white"}),
        html.Div(id='country_map_plot', children=[
            html.Div(id='settings_select', children=[
                html.Div(id='country_select_block', children=[
                    html.H6('General settings'),
                    dcc.RadioItems(id='country_type_radio', options=['Country', 'Mesoregion', 'Macroregion', 'World'],
                                   value='World', labelStyle={'display': 'inline-block'}),
                    dcc.Dropdown(
                        id='country_select_dropdown', options=country_names, value='Bulgaria',
                        clearable=False, style={'display': 'block'}),
                    dcc.Dropdown(
                        id='mesoregion_select_dropdown', options=mesoregions, value='Eastern Europe',
                        clearable=False, style={'display': 'block'}),
                    dcc.Dropdown(
                        id='macroregion_select_dropdown', options=macroregions, value='Europe',
                        clearable=False, style={'display': 'block'}),
                    html.P(id='date_type_text', children=['Choose time interval'],
                           style={'textAlign': 'center', 'align-items': 'center'}),
                    dcc.Dropdown(
                        id='date_type_selection',
                        options=['1 month', '3 month', '6 month', '1 year', 'All time', 'Choose dates'],
                        value='1 year',
                        clearable=False, style={'display': 'block'}),
                    html.P(id="choose_dates_text", children=["Choose dates between " +
                                                             min_date_allowed.strftime("%d-%m-%Y")
                                                             + " and " + max_date_allowed.strftime("%d-%m-%Y")],
                           style={'textAlign': 'center', 'justifyContent': 'center', 'align-items': 'center'}),
                    dcc.DatePickerRange(
                        id='my_date_picker_range',
                        display_format='DD-MM-YYYY',
                        min_date_allowed=min_date_allowed,
                        max_date_allowed=max_date_allowed,
                        initial_visible_month=datetime.now(),
                        end_date=end_date,
                        start_date=start_date,
                        style={'background-color': "rgb(40,40,40)", 'fontSize': 10}
                    ),
                ], style={'width': '60%', 'float': 'left', 'display': 'inline-block', "color": "white",
                          'textAlign': 'center', 'justifyContent': 'center', 'align-items': 'center'}),
                html.Div(id='date_select_block', children=[
                    html.H6('Forecast settings'),
                    dcc.Checklist(id='check_smoothed',
                                  options=['Smoothed data'], value=['Smoothed data']
                                  ),
                    dcc.Checklist(id='check_test',
                                  options=['Test forecast'], value=['Test forecast']
                                  ),
                    html.P(children=['Additional days for forecasting'],
                           style={'textAlign': 'center', 'align-items': 'center'}),
                    dcc.Input(id="days_after", type="number", value=60,
                              style={'textAlign': 'center', 'align-items': 'center'}),
                    html.P(id="days_before_text", children=["Amount of days for test"],
                           style={'textAlign': 'center', 'justifyContent': 'center',
                                  'align-items': 'center'}),
                    dcc.Input(id="days_before", type="number", value=60,
                              style={'textAlign': 'center', 'align-items': 'center'}),
                    html.P(id="days_before_warning", children=["Amount of days for test must be less then whole "
                                                               "chosen time interval"],
                           style={'display': 'none'}),
                ], style={'width': '40%', 'float': 'left', 'display': 'inline-block', "color": "white",
                          'textAlign': 'center'}),
            ], style={'width': '46%', 'float': 'left', 'display': 'inline-block', "color": "white",
                      'textAlign': 'center'}),
            html.Div(id='country_plot', children=[
                dcc.Graph(id='graph_country_cases', figure=fig_init)
            ], style={'width': '52%', 'display': 'inline-block', 'height': 350})
        ]),
        html.Div(id='cases_forecast', children=[
            html.Div(id='total_cases', children=[
                html.Div(id='main_map_div', children=[
                    html.Iframe(
                        id='map-main',
                        style={'border': 'none', 'width': '100%', 'height': 310},
                        srcDoc=open(rd.get_relative_path("init_map.html"), 'r').read()
                    )
                ], style={'width': '100%'})
            ], style={'width': '31%', 'float': 'left'}),
            html.Div(id='cases_total_box', children=[
                html.Div(id='total_covid', children=[
                    html.P(children="Total cases", style={'textAlign': 'center', "color": "white"}),
                    html.P(id='total_cases_num', children="{:,}".format(total_cases),
                           style={'textAlign': 'center', "color": "white"}),
                ], style={'width': '45%', 'float': 'left', 'display': 'inline-block', "backgroundColor": "#044038",
                          "border": "5px black solid"}),
                html.Div(id='death_covid', children=[
                    html.P(children="Total deaths", style={'textAlign': 'center', "color": "white"}),
                    html.P(id='death_cases_num', children="{:,}".format(total_death),
                           style={'textAlign': 'center', "color": "white"})
                ], style={'width': '45%', 'float': 'left', 'display': 'inline-block',
                          "backgroundColor": "rgb(40,40,40)", "border": "5px black solid"}),
                html.Div(id='regression_block', children=[
                    html.P(children="Choose regression type", style={'textAlign': 'center', "color": "white"}),
                    dcc.RadioItems(id='regression_type', options=['ARIMA', 'SARIMA', 'Lasso', 'Random Forest',
                                                                  'Linear Regression', 'Xgboost'],
                                   value='Linear Regression', style={'color': 'white'}, labelStyle={'display': 'block'})
                ], style={'width': '95%', 'float': 'left', 'display': 'inline-block',
                          "backgroundColor": "black", "border": "5px black solid"}),
            ], style={'width': '15%', 'display': 'inline-block', 'height': 350}),
            html.Div(id='forecast_block', children=[
                dcc.Graph(id='graph_forecast', figure=fig_init)
            ], style={'width': '52%', 'display': 'inline-block'})

        ])
    ], style={"widh": "100vw",
              "height": "100vh",
              "border": "10px solid rgb(0, 0, 0)",
              "outline": "10px solid rgb(0, 0, 0)",
              "margin": "0px 0px 0px 0px",
              "padding": "0px 0px 0px 0px",
              "backgroundColor": "black",
              "backgroundSize": "auto"}
)


@app.callback(
    Output(component_id='my_date_picker_range', component_property='style'),
    Output(component_id='choose_dates_text', component_property='style'),
    [Input(component_id='date_type_selection', component_property='value')])
def show_hide_element(date_type):
    if date_type == 'Choose dates':
        return {'display': 'block'}, {'display': 'block'}
    return {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output(component_id='days_before', component_property='style'),
    Output(component_id='days_before_text', component_property='style'),
    [Input(component_id='check_test', component_property='value')])
def show_hide_element(check_test):
    if 'Test forecast' in check_test:
        return {'textAlign': 'center', 'align-items': 'center'}, \
               {'textAlign': 'center', 'align-items': 'center'}
    return {'display': 'none'}, {'display': 'none'}


def define_start_end_dates(start_date, end_date, date_type):
    if date_type == '1 month':
        end_date = datetime.now() - relativedelta(days=1)
        start_date = end_date - relativedelta(days=30)
    elif date_type == '3 month':
        end_date = datetime.now() - relativedelta(days=1)
        start_date = end_date - relativedelta(days=90)
    elif date_type == '6 month':
        end_date = datetime.now() - relativedelta(days=1)
        start_date = end_date - relativedelta(days=180)
    elif date_type == '1 year':
        end_date = datetime.now() - relativedelta(days=1)
        start_date = end_date - relativedelta(days=365)
    elif date_type == 'All time':
        end_date = datetime.now() - relativedelta(days=1)
        start_date = min_date_allowed
    else:
        start_date = datetime.fromisoformat(start_date)
        end_date = datetime.fromisoformat(end_date)
    return start_date, end_date


@app.callback(
    Output('days_before_warning', 'style'),
    Output('days_before_warning', 'children'),
    Output('graph_forecast', 'figure'),
    [Input('country_select_dropdown', 'value'),
     Input('mesoregion_select_dropdown', 'value'),
     Input('macroregion_select_dropdown', 'value'),
     Input('country_type_radio', 'value'),
     Input('check_smoothed', 'value'),
     Input('check_test', 'value'),
     Input('days_before', 'value'),
     Input('days_after', 'value'),
     Input('my_date_picker_range', 'start_date'),
     Input('my_date_picker_range', 'end_date'),
     Input('date_type_selection', 'value'),
     Input('regression_type', 'value')])
def update(country, mesoregion, macroregion, region_type, smoothed_check, check_test, days_before, days_after,
           start_date, end_date, date_type, regression_type):
    print('graph_forecast update')
    current_df = []
    smoothed = False
    if 'Smoothed data' in smoothed_check:
        smoothed = True
    if 'Test forecast' not in check_test:
        days_before = 0
    title = 'Forecast COVID-19 cases in '
    if region_type == 'Country':
        current_df = dict_df_country[country]
        title = title + country
    elif region_type == 'Mesoregion':
        current_df = dict_df_mesoregion[mesoregion]
        title = title + mesoregion
    elif region_type == 'Macroregion':
        current_df = dict_df_macroregion[macroregion]
        title = title + macroregion
    elif region_type == 'World':
        current_df = df_world
        title = title + 'the World'

    fig = go.Figure()
    fig.update_layout(title=title, plot_bgcolor='black', template="plotly_dark", height=350, font_family="Georgia",
                      title_font_family="Georgia")
    fig.update_xaxes(title_text='Date', gridcolor='black',
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_yaxes(title_text='Cases', showgrid=True, gridwidth=1, gridcolor='gray',
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    print('graph_forecast add default fig')

    start_date, end_date = define_start_end_dates(start_date, end_date, date_type)
    days = (end_date - start_date).days
    if days_before > days - 1:
        return {'textAlign': 'center', 'align-items': 'center', 'color': '#6e1b15'}, \
               "Amount of days for test must be less then whole  chosen time interval", fig
    elif days - days_before < 301:
        return {'textAlign': 'center', 'align-items': 'center', 'color': '#6e1b15'}, \
               "Amount of days for train must be more then 300, got " + str(days - days_before), fig
    current_df = current_df[(current_df["Date"] >= start_date) & (current_df["Date"] <= end_date)]

    column = 'Confirmed'
    if smoothed:
        column = 'Confirmed_smoothed'
        current_df['Confirmed_smoothed'] = current_df['Confirmed'].ewm(span=25).mean()

    df_ts = current_df[['Date', column]]
    df_ts.index = pd.to_datetime(df_ts['Date'], format='%Y-%m-%d')
    del df_ts['Date']

    df_ts_freq = df_ts.asfreq('D')
    df_ts_freq[column] = df_ts_freq[column].fillna(0)

    train, test = forecast.make_train_test(df_ts_freq, days_before)
    test = forecast.add_more_dates(test, days_after)

    color = '#2fdecd'
    if smoothed:
        color = '#5f77e1'

    fig.add_trace(go.Scatter(x=train.index, y=train[column],
                             line=dict(color=color, width=2),
                             mode='lines', name='Confirmed per day TRAIN'))
    fig.add_trace(go.Scatter(x=test.index, y=test[column],
                             line=dict(color=color, width=2, dash='dot'),
                             mode='lines', name='Confirmed per day TEST'))

    print('graph_forecast start predictions')

    if regression_type == 'ARIMA':
        pred_df = forecast.make_forecast_ARIMA(column, train, test)
        fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predictions"],
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Forecast ARIMA'))
    elif regression_type == 'SARIMA':
        pred_df2 = forecast.make_forecast_SARIMAX(column, train, test)
        fig.add_trace(go.Scatter(x=pred_df2.index, y=pred_df2["Predictions"],
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Forecast SARIMA'))
    elif regression_type == 'Lasso':
        pred_Lasso = forecast.make_forecast_Lasso(column, train, test, 50)
        fig.add_trace(go.Scatter(x=pred_Lasso.index, y=pred_Lasso,
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Forecast Lasso'))
    elif regression_type == 'Linear Regression':
        pred_LR, lower_bound_LR, upper_bound_LR = forecast.make_forecast_LinearRegression(column, train, test, 50)
        fig.add_trace(go.Scatter(x=pred_LR.index, y=pred_LR,
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Linear Regression'))
        fig.add_trace(go.Scatter(x=lower_bound_LR.index, y=lower_bound_LR,
                                 line=dict(color='white', width=3, dash='dot'),
                                 mode='lines', name='Linear Regression lower bound'))
        fig.add_trace(go.Scatter(x=upper_bound_LR.index, y=upper_bound_LR,
                                 line=dict(color='white', width=3, dash='dash'),
                                 mode='lines', name='Linear Regression upper bound'))
    elif regression_type == 'Random Forest':
        pred_df_RFT = forecast.make_forecast_Random_Forest_Regressor(column, train, test, 10, 500, 100)
        fig.add_trace(go.Scatter(x=pred_df_RFT.index, y=pred_df_RFT,
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Forecast Random Forest Regressor'))
    elif regression_type == 'Xgboost':
        pred_df_xgboost = forecast.make_forecast_xgboost(column, train, test)
        fig.add_trace(go.Scatter(x=test.index, y=pred_df_xgboost,
                                 line=dict(color='white', width=3),
                                 mode='lines', name='Forecast Xgboost'))

    return {'display': 'none'}, '', fig


@app.callback(
    [Output(component_id='total_cases_num', component_property='children'),
     Output(component_id='death_cases_num', component_property='children')],
    [Input('country_select_dropdown', 'value'),
     Input('mesoregion_select_dropdown', 'value'),
     Input('macroregion_select_dropdown', 'value'),
     Input('country_type_radio', 'value'),
     Input('my_date_picker_range', 'start_date'),
     Input('my_date_picker_range', 'end_date'),
     Input(component_id='date_type_selection', component_property='value')])
def show_hide_element(country, mesoregion, macroregion, region_type, start_date, end_date, date_type):
    if region_type == 'Country':
        current_df = dict_df_country[country]
    elif region_type == 'Mesoregion':
        current_df = dict_df_mesoregion[mesoregion]
    elif region_type == 'Macroregion':
        current_df = dict_df_macroregion[macroregion]
    else:
        current_df = df_world
    start_date, end_date = define_start_end_dates(start_date, end_date, date_type)
    current_df = current_df[(current_df["Date"] >= start_date) & (current_df["Date"] <= end_date)]
    total_cases = math.floor(current_df['Confirmed'].sum())
    total_death = math.floor(current_df['Deaths_pure'].sum())
    return "{:,}".format(total_cases), "{:,}".format(total_death)


@app.callback(
    Output('map-main', 'srcDoc'),
    [Input('country_select_dropdown', 'value'),
     Input('mesoregion_select_dropdown', 'value'),
     Input('macroregion_select_dropdown', 'value'),
     Input('country_type_radio', 'value')
     ])
def update(country, mesoregion, macroregion, region_type):
    print('map-main update')
    df_for_map = df_COVID
    if region_type == 'Country':
        df_for_map = df_for_map[df_for_map['Country_Region'] == country].groupby(
            ['Country_Region', 'meso_region', 'macro_region', 'code']).agg(
            {'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'}).reset_index()
    elif region_type == 'Mesoregion':
        df_for_map = df_for_map[df_for_map['meso_region'] == mesoregion].groupby(
            ['Country_Region', 'meso_region', 'macro_region', 'code']).agg(
            {'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'}).reset_index()
    elif region_type == 'Macroregion':
        df_for_map = df_for_map[df_for_map['macro_region'] == macroregion].groupby(
            ['Country_Region', 'meso_region', 'macro_region', 'code']).agg(
            {'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'}).reset_index()
    elif region_type == 'World':
        df_for_map = df_COVID.groupby(
            ['Country_Region', 'meso_region', 'macro_region', 'code']).agg(
            {'Confirmed': 'max', 'Deaths': 'max', 'Recovered': 'max'}).reset_index()
    current_map = mapping.make_map(df_for_map)
    current_map.save(rd.get_relative_path("map.html"))
    delete_tile_for_map("map.html")
    return open(rd.get_relative_path("map.html"), 'r').read()


@app.callback(
    Output(component_id='country_select_dropdown', component_property='style'),
    [Input(component_id='country_type_radio', component_property='value')])
def show_hide_element(region):
    if region == 'Country':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output(component_id='mesoregion_select_dropdown', component_property='style'),
    [Input(component_id='country_type_radio', component_property='value')])
def show_hide_element(region):
    if region == 'Mesoregion':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output(component_id='macroregion_select_dropdown', component_property='style'),
    [Input(component_id='country_type_radio', component_property='value')])
def show_hide_element(region):
    if region == 'Macroregion':
        return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('graph_country_cases', 'figure'),
    [Input('country_select_dropdown', 'value'),
     Input('mesoregion_select_dropdown', 'value'),
     Input('macroregion_select_dropdown', 'value'),
     Input('country_type_radio', 'value'),
     Input('my_date_picker_range', 'start_date'),
     Input('my_date_picker_range', 'end_date'),
     Input(component_id='date_type_selection', component_property='value')])
def update(country, mesoregion, macroregion, region_type, start_date, end_date, date_type):
    print('graph_country_cases update')
    current_df = []
    title = ''
    if region_type == 'Country':
        current_df = dict_df_country[country]
        title = 'COVID-19 cases in ' + country
    elif region_type == 'Mesoregion':
        current_df = dict_df_mesoregion[mesoregion]
        title = 'COVID-19 cases in ' + mesoregion
    elif region_type == 'Macroregion':
        current_df = dict_df_macroregion[macroregion]
        title = 'COVID-19 cases in ' + macroregion
    elif region_type == 'World':
        current_df = df_world
        title = 'COVID-19 cases in the World'
    start_date, end_date = define_start_end_dates(start_date, end_date, date_type)
    current_df = current_df[(current_df["Date"] >= start_date) & (current_df["Date"] <= end_date)]
    current_df['Confirmed_smoothed'] = current_df['Confirmed'].ewm(span=25).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=current_df['Date'], y=current_df['Confirmed'],
                             line=dict(color='#2fdecd', width=2),
                             mode='lines', name='Confirmed per day'))
    fig.add_trace(go.Scatter(x=current_df['Date'], y=current_df['Confirmed_smoothed'], fill='tozeroy',
                             line=dict(color='#5f77e1', width=2),
                             mode='lines', name='Confirmed per day smoothed'))
    fig.add_trace(go.Scatter(x=current_df['Date'], y=current_df['Deaths'],
                             line=dict(color='rgb(150,150,150)', width=3, dash='dash'),
                             mode='lines', name='Deaths total'))
    fig.update_traces(visible="legendonly", selector=lambda t: t.name in ["Deaths total"])
    fig.update_layout(title=title, plot_bgcolor='black', template="plotly_dark", height=350, font_family="Georgia",
                      title_font_family="Georgia")
    fig.update_xaxes(title_text='Date', gridcolor='black',
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_yaxes(title_text='Cases', showgrid=True, gridwidth=1, gridcolor='gray',
                     zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
