# fao-project
Analysis of the food production and exchange around the world, focusing on the environmental impact caused by the different activites involved in the process.

The files found in this repository are organized in the following way:

- **datos_fao**: folder with CSV files from FAOstat website used in this project. It contains the basic data about food supply, trades, emissions, prices and population. It also contains the CSV file extracted from Our World in Data with information about the emissions intensity for additional food.

- **notebook_fao_analysis**: notebook with the detailed process of the EDA on the FAOstat data, the generation of a unique dataset, the addition of the OWID information on emissions intensity, and the calculation of the emissions derived from food transportation and production in the country of origin (for traded food)

- **dataframes_pkl**: folder with pickle files that contain the main DataFrames used in the dashboards

- **dashboard1**: first interactive dashboard where you can select a food, and a detailed analysis is displayed on its environmental impact in a global level.

- **dashboard2**: second interactive dashboard where you can select a food and a country, and a detailed analysis is displayed on its environmental impact in the specified country.

- **generadores_dfs**: folder with a notebook for each additional DataFrame created: food prices from the FAOstat CSV file, continents from the python library country-converter, and gdp information from WorldBank.
