# import dependencies
## dash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_daq as daq
import dash_bootstrap_components as dbc

## plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

## python packages
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re

# ## local packages
import helper.config as config
from helper.functions import get_sales, q25, q75, get_attribute_values

# initialize app
app = dash.Dash(
    __name__, 
    title=config.title,
    meta_tags=config.meta_tags
    )
app.index_string = config.html_header
application = app.server

# initialize data and dropdown attributes
collections = list(config.collection_attributes.keys()) + ['all']
marketplace_sales = get_sales('all', config.DEFAULT_LOOKBACK_WINDOW, 'MAGIC')
attributes_dfs = get_attribute_values(marketplace_sales, config.collection_attributes)

# create app layout
app.layout = html.Div([
    dcc.Location(id='url-input', refresh=False),
    dcc.Location(id='url-output', refresh=False),
    html.Div([
        html.Img(id='treasureLogo', src=app.get_asset_url('img/treasure_logo.png')),
        html.H1('Treasure NFT Sales', id='bannerTitle')
        ], className='bannerContainer'),
    html.Div([
        html.Div([
            html.Div([
                html.Div('NFT Collection:', className='headlineControlText'),
                dcc.Dropdown(
                    id='collection_dropdown',
                    options=[{'label': i.title().replace('_', ' '), 'value': i} for i in collections],
                    value='all',
                    clearable=False,
                    style=config.dropdown_style,
                    className='headlineDropdown')], 
                className='headlineControl'),
            html.Div([
                html.Div('Display Currency:', className='headlineControlText'),
                dcc.Dropdown(
                    id='pricing_unit',
                    options=[{'label': key, 'value': value} for key, value in config.pricing_unit_options.items()],
                    value='sale_amt_magic',
                    clearable=False,
                    style=config.dropdown_style,
                    className='headlineDropdown')],
                className='headlineControl'),
            html.Div([
                html.Div('Lookback Window:', className='headlineControlText'),
                dcc.Dropdown(
                    id='time_window',
                    options=[{'label': key, 'value': value} for key, value in config.lookback_window_options.items()],
                    value=7,
                    clearable=False,
                    style=config.dropdown_style,
                    className='headlineDropdown')],
                className='headlineControl')],
                id='headlineControlContainer'),
        html.Div(id='attributeDropdownContainer', children=[])], id='controls'),
    html.Div([
            html.Div([html.Div('Number of Sales: ', className='summaryStatLabel'), html.Div(id='n_sales', className='summaryStatMetric')], className='summaryStatBox'),
            html.Div([html.Div('Min Sale Price: ', className='summaryStatLabel'), html.Div(id='min_sale', className='summaryStatMetric')], className='summaryStatBox'),
            html.Div([html.Div('Avg Sale Price: ', className='summaryStatLabel'), html.Div(id='avg_sale', className='summaryStatMetric')], className='summaryStatBox'),
            html.Div([html.Div('Total Volume: ', className='summaryStatLabel'), html.Div(id='volume', className='summaryStatMetric')], className='summaryStatBox')],
        id='summaryStatsContainer'),
    html.Div([
        html.Div('Outliers'),
        daq.ToggleSwitch(
            id='outlier_toggle',
            label=['Show', 'Hide'],
            color='#374251',
            value=True
        ),
    ], id='outlierToggleContainer'),
    dcc.Graph(id='sales_scatter'),
    html.Div([
        dbc.Col('Frequency:'),
        dbc.Col(dcc.Dropdown(
            id='time_interval',
            options=[{'label': key, 'value': value} for key, value in config.date_interval_options.items()],
            value='1d',
            clearable=False,
            style=config.dropdown_style
        ))
    ], id='frequencyIntervalContainer'),
    dcc.Graph(id='volume_floor_prices'),
])

# function to dynamically update attribute inputs based on the collection
@app.callback(
    Output('attributeDropdownContainer', 'children'),
    Input('collection_dropdown', 'value'), 
    State('attributeDropdownContainer', 'children'))
def display_dropdowns(collection_value, children):
    if collection_value=='all':
        id_columns = [np.nan]
    else:
        id_columns = config.collection_attributes[collection_value]
        if 'is_one_of_one' in id_columns:
            id_columns.remove('is_one_of_one')
    children = []
    if (not pd.isnull(id_columns[0])): 
        attributes_df = attributes_dfs[collection_value]
        attributes_df = attributes_df.fillna('N/A')
        for attribute in id_columns:
            new_dropdown = html.Div([
                html.Div(
                    id={
                        'type':'filter_label',
                        'index':attribute
                    },
                    className='attributeText'
                ),
                dcc.Dropdown(
                    id={
                        'type':'filter_dropdown',
                        'index':attribute
                    },
                    options=[{'label': i, 'value': i} for i in list(attributes_df[attribute].unique()) + ['any']],
                    value='any',
                    clearable=False,
                    style=config.dropdown_style
                )
            ], className='attributeBox')
            children.append(new_dropdown)
        button = html.Div([
            html.Div('blank', id='buttonLabel'),
            html.Div(html.Button('Reset',id='attributeResetButton', n_clicks=0, style=config.dropdown_style))
        ], id = 'attributeResetContainer')
        children.append(button)
    return children

# function to reset attribute values
@app.callback(
    Output({'type': 'filter_dropdown', 'index': ALL}, 'value'),
    Input('attributeResetButton', 'n_clicks'),
    State({'type': 'filter_dropdown', 'index': ALL}, 'id'),
    Input('url-input', 'pathname')
)
def reset_attributes(reset, id_value, url_path):
    ctx = dash.callback_context

    reset_val_list = []
    for id in id_value:
        if ctx.triggered[0]['prop_id']=='attributeResetButton.n_clicks':
            reset_val_list.append('any')
        else:
            reset_val_list.append(re.findall("(?<={}=)[a-zA-Z10-9%/_-]+".format(id['index']), url_path)[0].replace('%20', ' '))
    return reset_val_list

# function to update the attribute labels
@app.callback(
    Output({'type': 'filter_label', 'index': MATCH}, 'children'),
    Input({'type': 'filter_dropdown', 'index': MATCH}, 'id'),
)
def display_output(id):
    title = id['index'].replace('_', ' ')
    if title == 'nft subcategory':
        title = 'Type' 
    return html.Div('{}:'.format(title))

# function to dynamically update inputs for brains and bodies based on gender
@app.callback(
    Output({'type': 'filter_dropdown', 'index': ALL}, 'options'),
    Input({'type': 'filter_dropdown', 'index': ALL}, 'value'),
    State('collection_dropdown', 'value'),
    State({'type': 'filter_dropdown', 'index': ALL}, 'id'),
)
def filter_attributes_gender(filter_value, collection_value, filter_id):
    if 'gender' not in  config.collection_attributes[collection_value]:
        raise PreventUpdate
    if 'male' in filter_value:
        gender_value = 'male'
    elif 'female' in filter_value:
        gender_value = 'female'
    else:
        gender_value = 'any'
    attributes_df = attributes_dfs[collection_value]
    attributes_df = attributes_df.fillna('N/A')
    attributes_df_filtered = attributes_df.loc[attributes_df['gender'].isin([gender_value] if gender_value!='any' else attributes_df['gender'].unique())].copy()

    options = []
    for item in filter_id:
        if item['index']=='gender':
            options.append([{'label': i, 'value': i} for i in list(attributes_df[item['index']].unique()) + ['any']])
        else:
            options.append([{'label': i, 'value': i} for i in list(attributes_df_filtered[item['index']].unique()) + ['any']])
    return options

# callback to filter data based on inputs
@app.callback(
    Output('n_sales', 'children'),
    Output('min_sale', 'children'),
    Output('avg_sale', 'children'),
    Output('volume', 'children'),
    Output('sales_scatter', 'figure'),
    Output('volume_floor_prices', 'figure'),
    Output('url-output', 'pathname'),
    Input('collection_dropdown', 'value'),
    Input({'type': 'filter_dropdown', 'index': ALL}, 'value'),
    State({'type': 'filter_dropdown', 'index': ALL}, 'id'),
    Input('pricing_unit', 'value'),
    Input('time_window', 'value'),
    Input('outlier_toggle', 'value'),
    Input('time_interval', 'value'),
    )
def update_stats(collection_value, value_columns, filter_columns, pricing_unit_value, time_window_value, outlier_toggle_value, time_interval_value):
    pricing_unit_label = 'MAGIC'
    if pricing_unit_value == 'sale_amt_usd':
        pricing_unit_label = 'USD'
    if pricing_unit_value == 'sale_amt_eth':
        pricing_unit_label = 'ETH'

    new_url = "/collection={}".format(collection_value)
    marketplace_sales_filtered = get_sales(collection_value, time_window_value, pricing_unit_label)

    if len(marketplace_sales_filtered)==0:
        marketplace_sales_filtered = marketplace_sales.loc[marketplace_sales['nft_collection']=='000000'] # cruft to allow us to take the column names
        marketplace_sales_filtered['sale_amt_eth'] = np.nan
        marketplace_sales_filtered['sale_amt_usd'] = np.nan
    if collection_value=='all':
        id_columns = [np.nan]
    else:
        id_columns = config.collection_attributes[collection_value]
        marketplace_sales_filtered = marketplace_sales_filtered.loc[marketplace_sales_filtered['nft_collection']==collection_value]
    if len(id_columns) > 1:
        attributes_df = attributes_dfs[collection_value]
        attributes_df = attributes_df.fillna('N/A')
        marketplace_sales_filtered = marketplace_sales_filtered.merge(attributes_df, how='inner',left_on='nft_id', right_on='id')

    if filter_columns:
        for filt, val in zip(filter_columns, value_columns):
            marketplace_sales_filtered = marketplace_sales_filtered.loc[marketplace_sales_filtered[filt['index']].isin([val]) if val!='any' else marketplace_sales_filtered[filt['index']].isin(marketplace_sales_filtered[filt['index']].unique())]
            new_url = new_url + '&{}={}'.format(filt['index'], val)
    marketplace_sales_filtered = marketplace_sales_filtered.loc[marketplace_sales_filtered['datetime'] >= pd.to_datetime(dt.datetime.now() - dt.timedelta(days = time_window_value))]
    new_url = new_url + '&lookback={}'.format(time_window_value)

    sales = marketplace_sales_filtered[pricing_unit_value].count()
    min_price = marketplace_sales_filtered[pricing_unit_value].min()
    avg_price = marketplace_sales_filtered[pricing_unit_value].mean()
    volume = marketplace_sales_filtered[pricing_unit_value].sum()
    new_url = new_url + '&priceunit={}'.format(pricing_unit_value)


    if outlier_toggle_value:
        # use daily IQR
        outlier_calc = marketplace_sales_filtered.groupby('date', as_index=True).agg({pricing_unit_value:[q25, q75]})
        outlier_calc.columns = outlier_calc.columns.droplevel(0)
        outlier_calc = outlier_calc.rename_axis(None, axis=1)
        outlier_calc['cutoff'] = (outlier_calc['q75'] - outlier_calc['q25']) * 1.5
        outlier_calc['upper'] = outlier_calc['q75'] + outlier_calc['cutoff']
        outlier_calc['lower'] = outlier_calc['q25'] - outlier_calc['cutoff']

        marketplace_sales_filtered = marketplace_sales_filtered.merge(outlier_calc, how='inner', on='date')
        marketplace_sales_filtered = marketplace_sales_filtered.loc[marketplace_sales_filtered[pricing_unit_value] <= marketplace_sales_filtered['upper']]
        marketplace_sales_filtered = marketplace_sales_filtered.loc[marketplace_sales_filtered[pricing_unit_value] >= marketplace_sales_filtered['lower']]
        
        new_url = new_url + '&outliers=true'
    else:
        new_url = new_url + '&outliers=false'

    marketplace_sales_filtered['nft_collection_formatted'] = [x.title().replace("_", " ")  for x in marketplace_sales_filtered['nft_collection']]
    marketplace_sales_filtered.loc[pd.isnull(marketplace_sales_filtered['nft_subcategory']), 'nft_subcategory'] = ''
    fig1 = px.scatter(marketplace_sales_filtered,
                     x='datetime',
                     y=pricing_unit_value,
                    #  trendline='ols',
                     custom_data=['nft_collection_formatted', 'nft_id', 'nft_subcategory'],
                     color_discrete_sequence=config.plot_color_palette)
    fig1.update_traces(hovertemplate='%{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<br><br>Date: %{x}<br>Sale Amount: %{y}')

    # trendline
    if len(marketplace_sales_filtered) > 1:
        Y = marketplace_sales_filtered[pricing_unit_value]
        X = pd.to_datetime(marketplace_sales_filtered['datetime']).map(dt.datetime.toordinal)
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        regression = model.fit()
        marketplace_sales_filtered['preds'] = regression.predict(X)
        marketplace_sales_filtered.sort_values(by='datetime', inplace=True)

        fig1.add_trace(
            go.Scatter(
                x=marketplace_sales_filtered['datetime'], 
                y=marketplace_sales_filtered['preds'], 
                mode='lines', 
                hoverinfo='skip', 
                marker={'color':config.plot_color_palette[0]}, 
                line = {'shape':'spline', 'smoothing':1.3})
            )
    fig1.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    fig1.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        hovermode='closest',
        showlegend=False)
    
    fig1.update_xaxes(title = '',
                     type='date',
                     gridcolor='#222938')
    fig1.update_yaxes(title='{}'.format(pricing_unit_label),
                     type='linear',
                     gridcolor='#8292a4')

    marketplace_sales_agg = marketplace_sales_filtered.copy()
    marketplace_sales_agg['datetime'] = marketplace_sales_agg['datetime'].dt.floor(time_interval_value)
    marketplace_sales_agg = marketplace_sales_agg.groupby('datetime').agg({pricing_unit_value:['sum', 'min','mean']})
    marketplace_sales_agg.columns = marketplace_sales_agg.columns.droplevel(0)
    marketplace_sales_agg = marketplace_sales_agg.rename_axis(None, axis=1)

    new_url = new_url + '&timeinterval={}'.format(time_interval_value)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fig2.add_scatter(x=marketplace_sales_agg.index,
                     y=marketplace_sales_agg['mean'],
                     name='Average Sale',
                     mode='lines',
                     secondary_y=True,
                     marker={'color':config.plot_color_palette[0], 'line':{'width':50}})
    fig2.add_scatter(x=marketplace_sales_agg.index,
                     y=marketplace_sales_agg['min'],
                     name='Minimum Sale',
                     mode='lines',
                     secondary_y=True,
                     marker={'color':config.plot_color_palette[2], 'line':{'width':50}})
    fig2.add_bar(x=marketplace_sales_agg.index,
                     y=marketplace_sales_agg['sum'],
                     name='Volume',
                     marker={'color':config.plot_color_palette[1], 'line': {'width':1.5, 'color':'DarkSlateGrey'}})
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        font_color='white',
        legend=dict(
        yanchor="bottom",
        y=-0.4,
        xanchor="left",
        x=0.85))
    fig2.update_xaxes(type='date')
    fig2.update_yaxes(title='Volume, {}'.format(pricing_unit_label),
                     type='linear',
                     gridcolor='#8292a4')
    fig2['layout']['yaxis2']['showgrid'] = False
    fig2['layout']['yaxis2']['title'] = 'Avg Sale Amount'

    return '{:,.0f}'.format(sales),\
            '{:,.2f}'.format(min_price),\
            '{:,.2f}'.format(avg_price),\
            '{:,.2f}'.format(volume),\
            fig1,\
            fig2,\
            new_url

# callback to filter data based on URL
@app.callback(
    Output('collection_dropdown', 'value'),
    Output('pricing_unit', 'value'),
    Output('time_window', 'value'),
    Output('outlier_toggle', 'value'),
    Output('time_interval', 'value'),
    Input('url-input', 'pathname')
    )
def update_inputs(url_path):
    return re.findall("(?<=collection=)[a-z0-9_]+", url_path)[0],\
        re.findall("(?<=priceunit=)[a-z0-9_]+", url_path)[0],\
        int(re.findall("(?<=lookback=)[a-z0-9_]+", url_path)[0]),\
        True if re.findall("(?<=outliers=)[a-z0-9_]+", url_path)[0]=='true' else False,\
        re.findall("(?<=timeinterval=)[a-z0-9_]+", url_path)[0]

if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_silence_routes_logging = False, dev_tools_props_check = False)
    # application.run(debug=False, port=8080)