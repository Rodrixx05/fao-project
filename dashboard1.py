from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_global_format.pkl'))
coord_df = pd.read_pickle(os.path.join('dataframes_pkl','df_coordenadas.pkl'))

uk_index = df[df['Area'] == 'United Kingdom'].index
df.loc[uk_index, 'Area'] = 'United Kingdom of Great Britain and Northern Ireland'

items_list = list(df['Item'].unique())
items_list.remove('Population - Est. & Proj.')

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = html.Div(
    [
    dbc.Row(dbc.Col(html.H1('Food distribution and environmental impact overview', style={'textAlign': 'center'}))),
    dbc.Row(dbc.Col(dcc.Dropdown(id = 'food_dropdown', options = items_list, placeholder = 'Select a food', value = 'Meat, chicken'), width = 6), justify='center', align='center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'graph_map', figure = {}), width = 8),
        dbc.Col(html.Div(id='table_map'), width = 4)], align = 'center', style = {'margin-right': 5}),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor', options = ['Emissions', 'Production'], value = 'Emissions', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 2}),
        dbc.Col(dcc.RadioItems(id = 'population_map', options = ['Totals', 'Per Capita'], value = 'Totals', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 2})]),
    dbc.Row(dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_map'), width = {'size': 6, 'offset': 3}, style={'margin-top': 15, 'margin-bottom': 30})),
    dbc.Row(dbc.Col(dcc.Graph(id = 'graph_comparison', figure = {}), width = 10), justify = 'center', align = 'center'),
    dbc.Row(
        [dbc.Col(dcc.RadioItems(id = 'population_graph', options = ['Totals', 'Per Capita'], value = 'Totals', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = {'size': 3, 'offset': 1}, style = {'margin-left': 5}),
        dbc.Col(dcc.Dropdown(id = 'country_dropdown', options = df['Area'].unique(), multi = True, placeholder = 'Select a country', value = ['Spain', 'Italy', 'Netherlands']), width = {'size': 5, 'offset': 0}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_comparison'), width = {'size': 3, 'offset': 0}, style = {'margin-right': 5, 'margin-bottom': 30})], justify='center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'graph_reg', figure = {}), width = 7, style = {'margin-left': 5}),
        dbc.Col([dbc.Row(
            html.Div(id='table_reg_most')
        ), dbc.Row(
            html.Div(id='table_reg_least')
        )], width = 4, align = 'center', style = {'margin-right': 5})]),
    dbc.Row(dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year_slider_reg'), width = {'size': 6, 'offset': 3}))
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
                        title_pad = {'b': 1}, autosize=True, 
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
    Input('year_slider_reg', 'value'),
)

def update_reg(food_dropdown, year_slider_reg):
    elements = ['Emissions (CO2eq)', 'Production']
    graph_df = df[(df['Element'].isin(elements)) & (df['Item'] == food_dropdown) & (df['Area'] == df['Partner Countries']) & (df['Year'] == year_slider_reg)]
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)

    graph_df.loc[:, 'Value'] = graph_df.apply(lambda x: x['Value'] * 1000 if x['Element'] == 'Emissions (CO2eq)' else x['Value'], axis = 1).copy()
    graph_df.loc[:, 'Unit'] = graph_df['Unit'].apply(lambda x: 'tonnes CO2' if x == 'gigagrams CO2' else x).copy()

    graph_df = graph_df.pivot_table(values = 'Value', index = 'Area', columns = ['Element'])
    graph_df.columns = ['CO2 Emissions [tonnes]', 'Production [tonnes]']
    graph_df['Emissions Intensity'] = graph_df['CO2 Emissions [tonnes]'] / graph_df['Production [tonnes]']
    title = f'CO2 emissions derived from {food_dropdown.lower()} production during {year_slider_reg}'

    fig = px.scatter(
        data_frame = graph_df, 
        x = 'Production [tonnes]', 
        y = 'CO2 Emissions [tonnes]', 
        trendline = 'ols', 
        trendline_color_override = '#fd8585',
        trendline_scope = 'overall', 
        hover_name = graph_df.index,
        hover_data = {'Emissions Intensity': ':.4'},
        color = graph_df.index, 
        title = title)
    
    fig.update_layout(showlegend = False)

    return_df_most = graph_df.reset_index()[['Area', 'Emissions Intensity']].sort_values(['Emissions Intensity'], ascending = False).head(3)
    return_df_most['Emissions Intensity'] = return_df_most['Emissions Intensity'].map('{:.3f}'.format)
    return_df_most.rename({'Area': 'Top Countries', 'Emissions Intensity': 'TCO2/Tfood'}, axis = 1, inplace = True)

    return_df_least = graph_df.reset_index()[['Area', 'Emissions Intensity']].sort_values(['Emissions Intensity']).head(3)
    return_df_least['Emissions Intensity'] = return_df_least['Emissions Intensity'].map('{:.3f}'.format)
    return_df_least.rename({'Area': 'Least Countries', 'Emissions Intensity': 'TCO2/Tfood'}, axis = 1, inplace = True)
    
    return fig, dbc.Table.from_dataframe(return_df_most, striped=True, bordered=True), dbc.Table.from_dataframe(return_df_least, striped=True, bordered=True)

if __name__ == '__main__':
    app.run_server(debug=True)

