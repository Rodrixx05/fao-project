from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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

app = Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

app.layout = html.Div(
    [
    dbc.Row(dbc.Col(html.H1('Food distribution and environmental impact for a chosen country', style={'textAlign': 'center'}), style={'margin-top': 30, 'margin-bottom': 30})),
    dbc.Row(
        [dbc.Col(dcc.Dropdown(id = 'food_dropdown', options = items_list, placeholder = 'Select a food', value = 'Apples'), width = 4, style={'margin-bottom': 30}),
        dbc.Col(dcc.Dropdown(id = 'country_dropdown', options = countries_list, placeholder = 'Select a country', value = 'Spain'), width = 4, style={'margin-bottom': 30})], 
        justify='center', align='center'),
    dbc.Row(
        [dbc.Col(dcc.Graph(id = 'time_series', figure = {}), width = 4),
        dbc.Col(dcc.Graph(id = 'gauge_autoconsum', figure = {}), width = 4),
        dbc.Col(dcc.Graph(id = 'distribution_emissions', figure = {}), width = 4)],
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
    [Output('time_series', 'figure'),
    Output('gauge_autoconsum', 'figure'),
    Output('distribution_emissions', 'figure')],
    Input('food_dropdown', 'value'),
    Input('country_dropdown', 'value')
)

def update_summary(food_dropdown, country_dropdown):
    graph1_df = df[(df['Element'] == 'Emissions (CO2eq)') & (df['Area'] == country_dropdown) & (df['Item'] == food_dropdown)].copy()
    graph1_df['Year'] = pd.to_datetime(graph1_df['Year'], format = '%Y')
    graph1_df['Value'] = graph1_df['Value'] * 1000
    labels1 = {'Value': 'CO2 Emissions [T]'}
    
    f'CO2 Emissions between {graph1_df["Year"].dt.year.min()} and {graph1_df["Year"].dt.year.max()}'

    fig1 = px.area(graph1_df, x = 'Year', y = 'Value', labels = labels1)
    fig1.update_layout(showlegend = False,     
        title = {
            'text': "Local CO2 Emissions time-series",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

    elementos_graph2 = ['Import Quantity', 'Export Quantity', 'Production']

    graph2_df = df[(df['Element'].isin(elementos_graph2)) & (df['Area'] == country_dropdown) & (df['Item'] == food_dropdown) & (df['Area'] == df['Partner Countries'])].copy()
    graph2_df = graph2_df.groupby(['Area', 'Item', 'Element']).sum().reset_index()[['Element', 'Value']]
    graph2_df.set_index('Element', inplace = True)

    for element in elementos_graph2:
        if element not in graph2_df.index:
            graph2_df = pd.concat([graph2_df, pd.Series([0], index = [element])])

    ratio = (graph2_df.loc['Production', 'Value'] - graph2_df.loc['Export Quantity', 'Value']) / (graph2_df.loc['Production', 'Value'] + graph2_df.loc['Import Quantity', 'Value']) * 100

    if ratio < 25:
        color = 'red'
    elif 25 <= ratio < 50:
        color = 'orange'
    elif 50 <= ratio < 75:
        color = 'gold'
    else:
        color = 'green'

    fig2= go.Figure(go.Indicator(
    mode = "gauge+number",
    value = ratio,
    title = {'text': "Self-consumption"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    number = {'valueformat': '.2r', 'suffix': '%'},
    gauge = {'axis': {'range': [None, 100], 'tickmode': 'linear', 'tick0': 0, 'dtick': 25}, 'bar': {'color': color}}))

    elementos_graph3 = ['Import Production Emissions Quantity', 'Import Transport Emissions Quantity']
    import_emissions = df[(df['Element'].isin(elementos_graph3)) & (df['Area'] == country_dropdown) & (df['Item'] == food_dropdown)]['Value'].sum()
    local_emissions = graph1_df['Value'].sum()
    graph3_df = pd.DataFrame({'Origin': ['Local Emissions', 'Imports Emissions'], 'CO2 Emissions [T]': [local_emissions, import_emissions]})
    color_map = {'Local Emissions': '636EFA', 'Imports Emissions': 'EF553B'}
    fig3 = px.pie(graph3_df, values = 'CO2 Emissions [T]', names = 'Origin', color = 'Origin', color_discrete_map = color_map)

    fig3.update_layout(title = {
        'text': "CO2 Emissions' origin",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, legend = dict(
        orientation="h"))

    return fig1, fig2, fig3

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