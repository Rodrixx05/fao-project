from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import country_converter as coco
import os

df = pd.read_pickle(os.path.join('dataframes_pkl', 'df_global_format.pkl'))
coord_df = pd.read_pickle(os.path.join('dataframes_pkl','df_coordenadas.pkl'))


uk_index = df[df['Area'] == 'United Kingdom'].index
df.loc[uk_index, 'Area'] = 'United Kingdom of Great Britain and Northern Ireland'

items_list = list(df['Item'].unique())
items_list.sort()
items_list.remove('Population - Est. & Proj.')

countries_list = list(df['Area'].unique())
countries_list.sort()

print()

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

app.layout = html.Div(
    [
    dbc.Row(dbc.Col(html.H1('Food distribution and environmental impact for a chosen country', style={'textAlign': 'center'}), style={'margin-top': 30, 'margin-bottom': 30})),
    dbc.Row(
        [dbc.Col(dcc.Dropdown(id = 'food_dropdown', options = items_list, placeholder = 'Select a food', value = 'Apples'), width = 4),
        dbc.Col(dcc.Dropdown(id = 'country_dropdown', options = countries_list, placeholder = 'Select a country', value = 'Spain'), width = 4)], 
        justify='center', align='center'),
    dbc.Row(dbc.Col(dcc.Graph(id = 'graph_sankey', figure = {}), width = 10), justify = 'center', align = 'center'),
    dbc.Row(dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(),id='year_slider_sankey', step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}), width = 6, style={'margin-bottom': 30}), justify = 'center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'map_imp_exp', figure = {}), width = 8),
        dbc.Col(html.Div(id='table_map_imp_exp'), width = 4)], align = 'center', style = {'margin-right': 5}),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor_imp_exp', options = ['Imports', 'Exports'], value = 'Imports', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = 3, style={'margin-bottom': 30}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(),id='year_slider_imp_exp', step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}), width = 5, style={'margin-bottom': 30})],
        justify = 'center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'map_imp_emissions', figure = {}), width = 8),
        dbc.Col(html.Div(id='table_map_imp_emissions'), width = 4)], align = 'center', style = {'margin-right': 5}),
    dbc.Row(  
        [dbc.Col(dcc.RadioItems(id = 'valor_imp_emissions', options = ['Production Emissions', 'Transport Emissions', 'Total Emissions'], value = 'Production Emissions', inputStyle={"margin-left": "20px", "margin-right": "5px"}), width = 5, style={'margin-bottom': 30}),
        dbc.Col(dcc.Slider(df['Year'].min(), df['Year'].max(),id='year_slider_imp_emissions', step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}), width = 5, style={'margin-bottom': 30})],
        justify = 'center'),
    ])

@app.callback(
    Output('graph_sankey', 'figure'),
    Input('food_dropdown', 'value'),
    Input('country_dropdown', 'value'),
    Input('year_slider_sankey', 'value'),
)

def update_sankey(food_dropdown, country_dropdown, year_slider_sankey):
    graph_df = df[df['Area'] == df['Partner Countries']]

    elements = ['Export Quantity', 'Import Quantity', 'Production', 'Loss', 'Processed', \
        'Food supply quantity (tonnes)', 'Feed', 'Seed','Other uses (non-food)']

    graph_df = graph_df[(graph_df['Area'] == country_dropdown) & (graph_df['Item'] == food_dropdown) & (graph_df['Element'].isin(elements)) & (graph_df['Year'] == year_slider_sankey)]
    
    sources = ['Import Quantity', 'Production']

    targets = [element for element in elements if element not in sources]

    labels = list(graph_df['Element'].unique()) + ['Supply']
    color = 'blue'

    graph_df['Source'] = graph_df['Element'].apply(lambda x: labels.index(x) if x in sources else labels.index('Supply'))
    graph_df['Target'] = graph_df['Element'].apply(lambda x: labels.index(x) if x in targets else labels.index('Supply'))

    title = f'{food_dropdown} supply flow in {country_dropdown} during {year_slider_sankey}'
    
    fig = go.Figure(data = [go.Sankey(
    arrangement = "snap",
    valueformat = ".2s",
    valuesuffix = "Tonnes",
    # Define nodes
    node = dict(
        pad = 15,
        thickness = 15,
        line = dict(color = "black", width = 0.5),
        label =  labels,
    ),
        link = dict(
        source =  graph_df['Source'],
        target =  graph_df['Target'],
        value =  graph_df['Value'],
    ))])
    fig.update_layout(title = title)
    
    return fig

@app.callback(
    [Output('map_imp_exp', 'figure'),
    Output('table_map_imp_exp', 'children')],
    Input('food_dropdown', 'value'),
    Input('country_dropdown', 'value'),
    Input('valor_imp_exp', 'value'),
    Input('year_slider_imp_exp', 'value')
)

def update_map_imp_exp(food_dropdown, country_dropdown, valor_imp_exp, year_slider_imp_exp):
    elementos = {'Imports': 'Import Quantity', 'Exports': 'Export Quantity'}

    graph_df = df[(df['Area'] == country_dropdown) & (df['Item'] == food_dropdown) & (df['Element'] == elementos[valor_imp_exp]) & (df['Year'] == year_slider_imp_exp)].copy()
    graph_df.drop(graph_df[(graph_df['Area'] == graph_df['Partner Countries']) & (graph_df['Element'] == elementos[valor_imp_exp])].index, inplace = True)
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)
    
    title = f'{food_dropdown} {elementos[valor_imp_exp].lower()} in {country_dropdown} during {year_slider_imp_exp}'
    
    fig = px.choropleth(graph_df, locations = 'Partner Countries', locationmode = 'country names', color = 'Value', color_continuous_scale = px.colors.sequential.YlOrBr, title = title, projection = 'equirectangular', labels = {'Value': 'Tonnes'})
    
    color_pais = 'Plum'
    fig.add_traces(go.Choropleth(locations=[country_dropdown],
                            locationmode = 'country names',
                            z = [1],
                            colorscale = [[0, color_pais],[1, color_pais]],
                            colorbar=None,
                            showscale = False,
                            hovertemplate = f'Selected Country= {country_dropdown}<extra></extra>'))

    fig.update_layout(title={"yref": "paper", "y" : 1, "yanchor" : "bottom"},
                        title_pad = {'b': 1}, autosize=True, 
                        margin = dict(l=10, r=5, b=10, t=30, pad=2, autoexpand=True))
    
    fig.update_geos(fitbounds="locations")
    
    return_df = graph_df[['Partner Countries', 'Value']].sort_values(['Value'], ascending = False).head(5)
    return_df.rename({'Partner Countries': 'Top Countries', 'Value': 'Tfood'}, axis = 1, inplace = True)

    return fig, dbc.Table.from_dataframe(return_df, striped=True, bordered=True)

@app.callback(
    [Output('map_imp_emissions', 'figure'),
    Output('table_map_imp_emissions', 'children')],
    Input('food_dropdown', 'value'),
    Input('country_dropdown', 'value'),
    Input('valor_imp_emissions', 'value'),
    Input('year_slider_imp_emissions', 'value')
)

def update_map_imp_emissions(food_dropdown, country_dropdown, valor_imp_emissions, year_slider_imp_emissions):

    elementos = {
        'Production Emissions': 'Import Production Emissions Quantity',
        'Transport Emissions': 'Import Transport Emissions Quantity'}

    graph_df = df[(df['Area'] == country_dropdown) & (df['Item'] == food_dropdown) & (df['Year'] == year_slider_imp_emissions)].copy()

    if valor_imp_emissions == 'Total Emissions':
        graph_df = graph_df[graph_df['Element'].isin(list(elementos.values()))]
    
    else:
        graph_df = graph_df[graph_df['Element'] == elementos[valor_imp_emissions]]

    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)
    
    if valor_imp_emissions == 'Total Emissions':
        graph_df = graph_df.groupby(['Area', 'Partner Countries', 'Item']).agg({'Unit': lambda x: x.mode()[0], 'Value': 'sum'}, axis = 1).reset_index()
    
    title = f'{food_dropdown} {valor_imp_emissions.lower()} quantity due to importations by {country_dropdown} during {year_slider_imp_emissions}'
    
    fig = px.choropleth(graph_df, locations = 'Partner Countries', locationmode = 'country names', color = 'Value', color_continuous_scale = px.colors.sequential.YlOrBr, title = title, projection = 'equirectangular', labels = {'Value': 'Tonnes CO2'})
    
    color_pais = 'Plum'
    fig.add_traces(go.Choropleth(locations=[country_dropdown],
                            locationmode = 'country names',
                            z = [1],
                            colorscale = [[0, color_pais],[1, color_pais]],
                            colorbar=None,
                            showscale = False,
                            hovertemplate = f'Selected Country= {country_dropdown}<extra></extra>'))

    fig.update_layout(title={"yref": "paper", "y" : 1, "yanchor" : "bottom"},
                        title_pad = {'b': 1}, autosize=True, 
                        margin = dict(l=10, r=5, b=10, t=30, pad=2, autoexpand=True))
    
    fig.update_geos(fitbounds="locations")
    
    return_df = graph_df[['Partner Countries', 'Value']].sort_values(['Value'], ascending = False).head(5)
    return_df['Value'] = return_df['Value'].map('{:.2f}'.format)
    return_df.rename({'Partner Countries': 'Top Countries', 'Value': 'TCO2'}, axis = 1, inplace = True)

    return fig, dbc.Table.from_dataframe(return_df, striped=True, bordered=True)

if __name__ == '__main__':
    app.run_server(debug=True)