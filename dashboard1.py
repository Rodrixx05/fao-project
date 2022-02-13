from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

pd.options.mode.chained_assignment = None

df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_global_format.pkl'))
coord_df = pd.read_pickle(os.path.join('dataframes_pkl','df_coordenadas.pkl'))
continent_df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_continentes.pkl'))
income_df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_income.pkl')).set_index('Area')
price_df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_precios.pkl'))
gdp_df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_gdp.pkl'))

items_list = list(df['Item'].unique())
items_list.sort()
items_list.remove('Population - Est. & Proj.')

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = html.Div(
    [
    dbc.Row(dbc.Col(html.H1('Food distribution and environmental impact overview', style={'textAlign': 'center'}), style={'margin-top': 30, 'margin-bottom': 15})),
    dbc.Row(dbc.Col(dcc.Dropdown(id = 'food_dropdown', options = items_list, placeholder = 'Select a food', value = 'Meat, chicken'), width = 6, style={'margin-top': 15, 'margin-bottom': 30}), justify='center', align='center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'graph_map', figure = {}), width = 8),
        dbc.Col(html.Div(id='table_map'), width = 4, lg = {'size': 4, 'offset': 0})], align = 'center', style = {'margin-right': 5, 'margin-left': 10}),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor', options = ['Emissions', 'Production'], value = 'Emissions', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 2}),
        dbc.Col(dcc.RadioItems(id = 'population_map', options = ['Totals', 'Per Capita'], value = 'Totals', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 2})]),
    dbc.Row(dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_map'), width = {'size': 6, 'offset': 3}, style={'margin-top': 15, 'margin-bottom': 30})),
    dbc.Row(dbc.Col(dcc.Graph(id = 'graph_comparison', figure = {}, style = {'height': '65vh'}), width = 10), justify = 'center', align = 'center'),
    dbc.Row(
        [dbc.Col(dcc.RadioItems(id = 'population_graph', options = ['Totals', 'Per Capita'], value = 'Totals', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 1}, style = {'margin-left': 5}),
        dbc.Col(dcc.Dropdown(id = 'country_dropdown', options = df['Area'].unique(), multi = True, placeholder = 'Select a country', value = ['Spain', 'Italy', 'Netherlands']), width = {'size': 5, 'offset': 0}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_comparison'), width = {'size': 3, 'offset': 0}, style = {'margin-right': 5, 'margin-bottom': 30})], justify='center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'graph_reg', figure = {}, style = {'height': '75vh'}), width = 8, style = {'margin-left': 5}),
        dbc.Col([dbc.Row(
            html.Div(id='table_reg_most')
        ), dbc.Row(
            html.Div(id='table_reg_least')
        )], width = 3, align = 'center', style = {'margin-right': 5})]),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor_area', options = ['Country', 'Continent', 'GDP'], value = 'Country', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = 3, style={'margin-bottom': 30}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(),id='year_slider_reg', step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}), width = 5, style={'margin-bottom': 30})],
        justify = 'center'),
    dbc.Row(dbc.Col(dcc.Graph(id = 'graph_prices', figure = {}, style = {'height': '75vh'}), width = 10), justify = 'center', align = 'center'),
    dbc.Row(dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_prices'), width = {'size': 6, 'offset': 3}, style={'margin-top': 15, 'margin-bottom': 30})),
    dbc.Row(dbc.Col(dcc.Graph(id = 'graph_losses', figure = {}, style = {'height': '75vh'}), width = 10), justify = 'center', align = 'center'),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor_losses', options = ['Totals', 'Per Capita'], value = 'Totals', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = 3, style={'margin-bottom': 30}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(),id='year_slider_losses', step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}), width = 5, style={'margin-bottom': 30})],
        justify = 'center'),    
    ]
)

@app.callback(
    [Output('graph_map', 'figure'),
    Output('table_map', 'children')],
    Input('food_dropdown', 'value'),
    Input('year_slider_map', 'value'),
    Input('valor', 'value'),
    Input('population_map', 'value')
)

def update_map(food_dropdown, year_slider_map, valor, population_map):
    if valor == 'Emissions':
        elemento = 'Emissions (CO2eq)'
    elif valor == 'Production':
        elemento = 'Production'

    graph_df = df[(df['Item'] == food_dropdown) & (df['Element'] == elemento) & (df['Year'] == year_slider_map)].copy()
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)
    
    if population_map == 'Per Capita':

        pop_df = df[(df['Element'] == 'Total Population - Both sexes') & (df['Year'] == year_slider_map)]
        
        pop_df.set_index('Area', inplace = True)

        factor = 1 if valor == 'Production' else 1000

        graph_df.loc[:, 'Value'] = graph_df.apply(lambda x: x['Value'] * factor / pop_df.loc[x['Area'], 'Value'], axis = 1)
    
    title = f'{food_dropdown} {valor.lower()} quantity in the world during {year_slider_map}'

    if population_map == 'Per Capita':
        labels = {'Value': 'Tfood / 1000p'} if valor == 'Production' else {'Value': 'TCO2 / 1000p'}
    else:
        labels = {'Value': 'Tfood'} if valor == 'Production' else {'Value': 'TCO2'}
    
    fig = px.choropleth(graph_df, locations = 'Area', locationmode = 'country names', color = 'Value', color_continuous_scale = px.colors.sequential.YlOrBr, title = title, projection = 'equirectangular', labels = labels)

    fig.update_layout(title={"yref": "paper", "y" : 1, "yanchor" : "bottom"},
                        title_pad = {'b': 10}, autosize=True, 
                        margin = dict(l=10, r=5, b=10, t=30, pad=2, autoexpand=True))

    return_df = graph_df[['Area', 'Value']].sort_values(['Value'], ascending = False).head(5)
    return_df['Value'] = return_df['Value'].map('{:.2f}'.format)
    labels['Area'] = 'Top Countries'
    return_df.rename(labels, axis = 1, inplace = True)

    return fig, dbc.Table.from_dataframe(return_df, striped=True, bordered=True)

@app.callback(
    Output('graph_comparison', 'figure'),
    Input('food_dropdown', 'value'),
    Input('country_dropdown', 'value'),
    Input('year_slider_comparison', 'value'),
    Input('population_graph', 'value')
)

def update_comparison(food_dropdown, country_dropdown, year_slider_comparison, population_graph):
    elements = ['Emissions (CO2eq)','Import Transport Emissions Quantity', 'Import Production Emissions Quantity']
    graph_df = df[(df['Area'].isin(country_dropdown)) & (df['Element'].isin(elements)) & (df['Item'] == food_dropdown) & (df['Year'] == year_slider_comparison)].copy()
    graph_df['Value'] = graph_df.apply(lambda x: x['Value'] * 1000 if x['Element'] == 'Emissions (CO2eq)' else x['Value'], axis = 1)
    graph_df['Element'] = graph_df['Element'].apply(lambda x: 'Production Emissions Quantity' if x == 'Emissions (CO2eq)' else x)
    graph_df['Unit'] = graph_df['Unit'].apply(lambda x: 'tonnes CO2' if x == 'gigagrams CO2' else x)

    if population_graph == 'Per Capita':

        pop_df = df[(df['Element'] == 'Total Population - Both sexes') & (df['Year'] == year_slider_comparison)]
        
        pop_df.set_index('Area', inplace = True)

        graph_df.loc[:, 'Value'] = graph_df.apply(lambda x: x['Value'] / pop_df.loc[x['Area'], 'Value'], axis = 1)

    labels = {'Area': 'Country', 'Element': 'Emissions type', 'Partner Countries': 'Partner Country'}

    if population_graph == 'Per Capita':
        labels['Value'] = 'TCO2 / 1000p'
    else:
        labels['Value'] = 'TCO2'

    title = f"{food_dropdown} emissions' origin comparator between countries during {year_slider_comparison}"
    fig = px.bar(graph_df, x="Area", y="Value", color="Element", barmode = 'group', range_y = [0, graph_df['Value'].max() * 1.2], labels = labels, title = title, hover_data=['Partner Countries'])
    return fig

@app.callback(
    [Output('graph_reg', 'figure'),
        Output('table_reg_most', 'children'),
        Output('table_reg_least', 'children')],
    Input('food_dropdown', 'value'),
    Input('valor_area', 'value'),
    Input('year_slider_reg', 'value'),
)

def update_reg(food_dropdown, valor_area, year_slider_reg):
    elements = ['Emissions (CO2eq)', 'Production']
    graph_df = df[(df['Element'].isin(elements)) & (df['Item'] == food_dropdown) & (df['Area'] == df['Partner Countries']) & (df['Year'] == year_slider_reg)]
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)

    graph_df.loc[:, 'Value'] = graph_df.apply(lambda x: x['Value'] * 1000 if x['Element'] == 'Emissions (CO2eq)' else x['Value'], axis = 1).copy()
    graph_df.loc[:, 'Unit'] = graph_df['Unit'].apply(lambda x: 'tonnes CO2' if x == 'gigagrams CO2' else x).copy()
    
    graph_df = graph_df.pivot_table(values = 'Value', index = 'Area', columns = ['Element'])
    graph_df.columns = ['CO2 Emissions [tonnes]', 'Production [tonnes]']
    graph_df['Emissions Intensity'] = graph_df['CO2 Emissions [tonnes]'] / graph_df['Production [tonnes]']
    title = f'CO2 emissions derived from {food_dropdown.lower()} production during {year_slider_reg}'
    category_orders = {}
    color = graph_df.index

    if valor_area == 'Continent':
        graph_df = pd.merge(graph_df, continent_df, 'left', left_index = True, right_on = 'Area').set_index('Area').copy()
        color = graph_df['Continent']
    elif valor_area == 'GDP':
        income_year_df = income_df[income_df['Year'] == year_slider_reg].copy()
        graph_df = pd.merge(graph_df, income_year_df, 'left', left_index = True, right_on = 'Area')
        color = graph_df['Income Group']
        category_orders = {'Income Group': ['Low Income', 'Low Middle Income', 'Upper Middle Income', 'High Income']}

    fig = px.scatter(
        data_frame = graph_df, 
        x = 'Production [tonnes]', 
        y = 'CO2 Emissions [tonnes]', 
        trendline = 'ols', 
        trendline_color_override = '#fd8585',
        trendline_scope = 'overall', 
        hover_name = graph_df.index,
        hover_data = {'Emissions Intensity': ':.4'},
        color = color,
        category_orders = category_orders, 
        title = title)
    
    if valor_area == 'Country':
        fig.update_layout(showlegend = False)

    return_df_most = graph_df.reset_index()[['Area', 'Emissions Intensity']].sort_values(['Emissions Intensity'], ascending = False).head(5)
    return_df_most['Emissions Intensity'] = return_df_most['Emissions Intensity'].map('{:.3f}'.format)
    return_df_most.rename({'Area': 'Top Countries', 'Emissions Intensity': 'TCO2/Tfood'}, axis = 1, inplace = True)

    return_df_least = graph_df.reset_index()[['Area', 'Emissions Intensity']].sort_values(['Emissions Intensity']).head(5)
    return_df_least['Emissions Intensity'] = return_df_least['Emissions Intensity'].map('{:.3f}'.format)
    return_df_least.rename({'Area': 'Least Countries', 'Emissions Intensity': 'TCO2/Tfood'}, axis = 1, inplace = True)
    
    return fig, dbc.Table.from_dataframe(return_df_most, striped=True, bordered=True), dbc.Table.from_dataframe(return_df_least, striped=True, bordered=True)

@app.callback(
    Output('graph_prices', 'figure'),
    Input('food_dropdown', 'value'),
    Input('year_slider_prices', 'value')
)

def update_graph_gdp(food_dropdown, year_slider_prices):
    graph_df = df[(df['Element'] == 'Emissions intensity') & (df['Item'] == food_dropdown) & (df['Year'] == year_slider_prices)]
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)
    graph_df.drop(columns = ['Partner Countries', 'Partner Country Code'], inplace = True)

    price_graph_df = price_df[price_df['Year'].astype(int) == year_slider_prices]

    graph_df = graph_df.append(price_graph_df)

    graph_df = graph_df.pivot_table(values = 'Value', index = 'Area', columns = ['Element'])

    gdp_graph_df = gdp_df[gdp_df['Year'].astype(int) == year_slider_prices].set_index('Area').drop(columns = 'Year')
    graph_df = pd.merge(graph_df, gdp_graph_df, left_index = True, right_index = True)

    graph_df = pd.merge(graph_df, continent_df, 'left', left_index = True, right_on = 'Area').set_index('Area').copy()
    graph_df.dropna(inplace = True)
    graph_df.reset_index(inplace = True)
    
    title = f'Relation between emissions intensity and {food_dropdown.lower()} price in the world during {year_slider_prices}'
    labels = {'Emissions intensity': 'Emissions Intensity [TCO2/Tfood]', 'Producer Price (USD/tonne)': 'Producer Price [USD/Tfood]', 'GDP_PCAP': 'GDP/Cap'}

    fig = px.scatter(
        data_frame = graph_df, 
        y = 'Emissions intensity', 
        x = 'Producer Price (USD/tonne)',
        size = graph_df['GDP_PCAP'] ** 0.7,
        color = 'Continent',
        labels = labels,
        title = title,
        custom_data = ['Area'])
    
    fig.update_traces(hovertemplate = "<b>%{customdata[0]}</b><br><br>" +
        "Production price: %{x:.2f}$/Tfood<br>" +
        "Emissions Intensity: %{y:.4f}TCO2/Tfood<br>" +
        "GDP per Capita: %{marker.size:$,.0f}" +
        "<extra></extra>")
    
    return fig

@app.callback(
    Output('graph_losses', 'figure'),
    Input('food_dropdown', 'value'),
    Input('valor_losses', 'value'),
    Input('year_slider_losses', 'value')
)

def update_graph_losses(food_dropdown, valor_losses, year_slider_losses):
    elements = ['Loss', 'Emissions intensity']
    graph_df = df[(df['Element'].isin(elements)) & (df['Item'] == food_dropdown) & (df['Year'] == year_slider_losses)]   
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)

    graph_df = graph_df.pivot_table(values = 'Value', index = 'Area', columns = ['Element']).reset_index()
    graph_df.dropna(inplace = True)
    graph_df['Loss Emissions'] = graph_df['Emissions intensity'] * graph_df['Loss']
    labels = {'Loss Emissions': 'Loss Emissions [TC02]', 'Area': 'Country'}
    title = f'Top 10 countries in CO2 emissions due to {food_dropdown.lower()} losses during {year_slider_losses}'

    if valor_losses == 'Per Capita':

        pop_df = df[(df['Element'] == 'Total Population - Both sexes') & (df['Year'] == year_slider_losses)]
        
        pop_df.set_index('Area', inplace = True)

        graph_df.loc[:, 'Loss Emissions'] = graph_df.apply(lambda x: x['Loss Emissions'] / pop_df.loc[x['Area'], 'Value'], axis = 1)

        labels = {'Loss Emissions': 'Loss Emissions [TC02/1000p]', 'Area': 'Country'}
    
    graph_df = graph_df.sort_values(by = 'Loss Emissions', ascending = False).head(10)

    fig = px.bar(graph_df, x = 'Area', y = 'Loss Emissions', labels = labels, title = title, color = 'Loss Emissions', color_continuous_scale = px.colors.sequential.YlOrBr)

    return fig    


if __name__ == '__main__':
    app.run_server(debug=True)



    