from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import os

os.chdir('dataframes_pkl')

df = pd.read_pickle('df_global_format.pkl')
coord_df = pd.read_pickle('df_coordenadas.pkl')

uk_index = df[df['Area'] == 'United Kingdom'].index
df.loc[uk_index, 'Area'] = 'United Kingdom of Great Britain and Northern Ireland'

app = Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Dropdown(id = 'food_dropdown', options = df['Item'].unique(), placeholder = 'Select a food', value = 'Rice, paddy'),
        dcc.Graph(id = 'graph_map', figure = {}),
        # dcc.Graph(id = 'graph_figure', figure = {}),
        dcc.Slider(df['Year'].min(), df['Year'].max(), step = None, value = df['Year'].min(), marks={str(year): str(year) for year in df['Year'].unique()}, id='year-slider'),
        dcc.RadioItems(id = 'valor', options = ['Emissions', 'Production'], value = 'Emissions'),
        dcc.RadioItems(id = 'population', options = ['Total', 'Per Capita'], value = 'Total')
    ]
)

@app.callback(
    Output('graph_map', 'figure'),
    # Output('graph_figure', 'figure'),
    Input('food_dropdown', 'value'),
    Input('year_slider', 'value'),
    Input('valor', 'value'),
    Input('population', 'value')
)

def update_map(food_dropdown, year_slider, valor, population):
    if valor == 'Emissions':
        elemento = 'Emissions (CO2eq)'
    elif valor == 'Production':
        elemento = 'Production'

    graph_df = df[(df['Item'] == food_dropdown) & (df['Element'] == elemento) & (df['Year'] == year_slider)].copy()
    graph_df.drop(graph_df[graph_df['Value'] == 0].index, inplace = True)
    
    if population == 'Per Capita':

        pop_df = df[(df['Element'] == 'Total Population - Both sexes') & (df['Year'] == year_slider)]
        
        pop_df.set_index('Area', inplace = True)

        factor = 1 if valor == 'Production' else 1000

        graph_df.loc[:, 'Value'] = graph_df.apply(lambda x: x['Value'] * factor / pop_df.loc[x['Area'], 'Value'], axis = 1)
    
    title = f'{food_dropdown} {valor} quantity in the world during {year_slider}'

    labels = {'Value': 'Tfood / 1000p'} if valor == 'Production' else {'Value': 'TCO2 / 1000p'}
    
    fig = px.choropleth(graph_df, locations = 'Area', locationmode = 'country names', color = 'Value', color_continuous_scale = px.colors.sequential.YlOrBr, title = title, projection = 'equirectangular', labels = labels)

    fig.update_layout(
        autosize=False,
        margin = dict(l=10, r=5, b=10, t=45, pad=4, autoexpand=True),
        width=900)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

