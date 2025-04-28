
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import folium
from folium.plugins import HeatMap
import plotly.express as px
import os

# ---- Load Cleaned HWC Data ----
path = "./assets/CLEANED_Land_for_Life_HWC.csv"
df = pd.read_csv(path)

# Data Cleaning
df = df.dropna(subset=['latitude', 'longitude', 'date'])
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Map Center
map_center = [-2.536, 36.585]
# ---- Initialize Dash App ----
external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "HWC Conflict Dashboard"

# ---- App Layout ----
app.layout = html.Div([
    html.Div([
        html.H2("🐾 Human-Wildlife Conflict Hotspot Analysis - Lake Natron", className="text-center mb-4 mt-2"),
        
        # Dropdown
        html.Div([
            html.Label("Select Year:", className="form-label"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in sorted(df['year'].unique())],
                value=2022,
                clearable=False,
                className="mb-3"
            ),
        ], className="col-md-4 mx-auto"),

        # Map and Graph
        html.Div([
            html.Div([
                html.Iframe(id="conflict-heatmap", style={"width": "100%", "height": "500px", "border": "none"}),
            ], className="col-12 col-md-6 mb-4"),

            html.Div([
                dcc.Graph(id="monthly-trend-graph"),
            ], className="col-12 col-md-6 mb-4"),
        ], className="row justify-content-center"),

    ], className="container-fluid")
])

# ---- Callback for Heatmap ----
@app.callback(
    Output('conflict-heatmap', 'srcDoc'),
    Input('year-dropdown', 'value')
)
def update_map(selected_year):
    filtered = df[df['year'] == selected_year]

    m = folium.Map(location=map_center, zoom_start=8, tiles="OpenStreetMap")

    heat_data = [[row["latitude"], row["longitude"]] for _, row in filtered.iterrows()]
    HeatMap(heat_data, radius=15, blur=12, min_opacity=0.5).add_to(m)

    output_file = "conflict_heatmap.html"
    m.save(output_file)

    return open(output_file, 'r').read()

# ---- Callback for Monthly Trend Graph ----
@app.callback(
    Output("monthly-trend-graph", "figure"),
    Input("year-dropdown", "value")
)
def update_graph(selected_year):
    monthly_trend = df[df["year"] == selected_year].groupby("month").size().reset_index(name="incident_count")

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    monthly_trend["month"] = monthly_trend["month"].map(month_names)

    fig = px.line(
        monthly_trend,
        x="month",
        y="incident_count",
        markers=True,
        title=f"📈 MCT in {selected_year}",
        labels={"month": "Month", "incident_count": "Total Incidents"}
    )

    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(month_names.values())),
        margin={"r":5,"t":40,"l":5,"b":0},
        height=450
    )

    return fig

# ---- Run App ----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run_server(debug=True, host='0.0.0.0', port=port)
