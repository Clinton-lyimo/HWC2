# import pandas as pd
# import geopandas as gpd
# import dash
# from dash import dcc, html, Input, Output
# import folium
# from folium.plugins import MarkerCluster
# import os
#
# # Load cleaned HWC data
# data_dir = "D:/HWC/"
# path = os.path.join(data_dir, "CLEANED_Land_for_Life_HWC.csv")
# df = pd.read_csv(path)
# df = df.dropna(subset=['latitude', 'longitude', 'date'])
# df['date'] = pd.to_datetime(df['date'], errors='coerce')
# df['year'] = df['date'].dt.year
#
# # Initialize Dash app
# app = dash.Dash(__name__)
# app.title = "HWC Conflict Dashboard"
#
# app.layout = html.Div([
#     html.H2("Human-Wildlife Conflict Incidents (2022–2024)"),
#
#     html.Label("Select Year:"),
#     dcc.Dropdown(
#         id='year-dropdown',
#         options=[{'label': str(y), 'value': y} for y in sorted(df['year'].unique())],
#         value=2022,
#         clearable=False
#     ),
#
#     html.Iframe(id="conflict-map", width="100%", height="600px")
# ])
#
# @app.callback(
#     Output('conflict-map', 'srcDoc'),
#     Input('year-dropdown', 'value')
# )
# def update_map(selected_year):
#     filtered = df[df['year'] == selected_year]
#
#     # Create Folium map centered on Tanzania
#     m = folium.Map(location=[-6.8, 39.27], zoom_start=8, tiles="OpenStreetMap")
#
#     # Add Marker Cluster
#     marker_cluster = MarkerCluster().add_to(m)
#
#     for _, row in filtered.iterrows():
#         folium.Marker(
#             location=[row["latitude"], row["longitude"]],
#             popup=f"Date: {row['date'].strftime('%Y-%m-%d')}",
#             icon=folium.Icon(color="red")
#         ).add_to(marker_cluster)
#
#     # Save interactive map as HTML
#     output_file = os.path.join(data_dir, "conflict_hotspot_map.html")
#     m.save(output_file)
#
#     return open(output_file, 'r').read()
#
# # Run app
# if __name__ == '__main__':
#     app.run_server(debug=True)


# THE ABOVE IS GOOD SHOWING MARKERS SAVE IT FOR LATER USE

# push test






import pandas as pd
import dash
from dash import dcc, html, Input, Output
import folium
from folium.plugins import HeatMap
import plotly.express as px
import os

# ---- Load Cleaned HWC Data ----
data_dir = "D:/HWC/"
path = os.path.join(data_dir, "CLEANED_Land_for_Life_HWC.csv")
df = pd.read_csv(path)

# Drop rows with missing coordinates
df = df.dropna(subset=['latitude', 'longitude', 'date'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# ---- Set Map Focus on Lake Natron AOI ----
map_center = [-2.136, 36.085]  # User-defined focus area

# ---- Initialize Dash App ----
app = dash.Dash(__name__)
app.title = "HWC Conflict Dashboard"

app.layout = html.Div([
    html.H2("Human-Wildlife Conflict Hotspot Analysis - Lake Natron"),

    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(y), 'value': y} for y in sorted(df['year'].unique())],
            value=2022,
            clearable=False
        ),
    ], style={'width': '30%', 'margin-bottom': '20px'}),

    # ---- Side-by-Side Layout ----
    html.Div([
        # Map on the left
        html.Div([
            html.Iframe(id="conflict-heatmap", width="100%", height="500px"),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # Graph on the right
        html.Div([
            dcc.Graph(id="monthly-trend-graph"),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])

# ---- Callback for Heatmap ----
@app.callback(
    Output('conflict-heatmap', 'srcDoc'),
    Input('year-dropdown', 'value')
)
def update_map(selected_year):
    filtered = df[df['year'] == selected_year]

    m = folium.Map(location=map_center, zoom_start=10, tiles="OpenStreetMap")

    # Prepare data for HeatMap
    heat_data = [[row["latitude"], row["longitude"]] for _, row in filtered.iterrows()]
    HeatMap(heat_data, radius=15, blur=12, min_opacity=0.5).add_to(m)

    # Save interactive heatmap as HTML
    # output_file = os.path.join(data_dir, "conflict_heatmap.html")
    output_file = "conflict_heatmap.html";
    m.save(output_file)

    return open(output_file, 'r').read()

# ---- Callback for Monthly Trend Graph ----
@app.callback(
    Output("monthly-trend-graph", "figure"),
    Input("year-dropdown", "value")
)
def update_graph(selected_year):
    monthly_trend = df[df["year"] == selected_year].groupby("month").size().reset_index(name="incident_count")

    # Convert month numbers to names
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    monthly_trend["month"] = monthly_trend["month"].map(month_names)  # Replace numbers with names

    # Create line graph
    fig = px.line(
        monthly_trend,
        x="month",
        y="incident_count",
        markers=True,
        title=f"Monthly Conflict Trends in {selected_year}",
        labels={"month": "Month", "incident_count": "Total Incidents"}
    )

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(month_names.values())),  # Ensures correct order
        margin={"r":0,"t":40,"l":0,"b":0}
    )

    return fig

# ---- Run Dash Server ----
if __name__ == '__main__':
    app.run_server(debug=True)



# THE ABOVE IS GOOD SHOWING MARKERS SAVE IT FOR LATER USE



