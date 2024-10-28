import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Load the data
data = pd.read_csv('electricity-consumption.csv')

# Data preprocessing
data.drop(columns=['prectotcorr', 'date_2'], inplace=True)
data.dropna(inplace=True)

# Calculate monthly averages
monthly_avg = data.groupby('month')[['total_demand(mw)', 'max_generation(mw)']].mean().reset_index()

def create_dashboard_plots():
    # Create the first plot (Temperature vs Demand)
    fig1 = px.scatter(data, x='temp2_ave(c)', y='total_demand(mw)', 
                      title='Temperature vs Electricity Demand',
                      labels={'temp2_ave(c)': 'Average Temperature (°C)', 
                              'total_demand(mw)': 'Total Demand (MW)'})

    # Create the second plot (Monthly Average Demand and Generation)
    fig2 = make_subplots(rows=1, cols=2, 
                         specs=[[{'type':'domain'}, {'type':'xy'}]],
                         subplot_titles=('Average Monthly Demand Distribution', 
                                         'Demand vs Generation Trend'))

    # Add pie chart for average monthly demand
    fig2.add_trace(go.Pie(
        labels=monthly_avg['month'],
        values=monthly_avg['total_demand(mw)'],
        name='Avg Monthly Demand',
        hole=0.4,
        hoverinfo='label+percent+value',
        textinfo='label+percent'
    ), row=1, col=1)

    # Add line chart for demand vs generation trend
    fig2.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['total_demand(mw)'],
        name='Avg Demand',
        line=dict(color='blue', width=2)
    ), row=1, col=2)

    fig2.add_trace(go.Scatter(
        x=monthly_avg['month'],
        y=monthly_avg['max_generation(mw)'],
        name='Avg Max Generation',
        line=dict(color='red', width=2)
    ), row=1, col=2)

    # Update layout for fig2
    fig2.update_layout(
        title='Electricity Demand and Generation Analysis',
        height=600,
        width=1200
    )

    # Update axes for the line chart in fig2
    fig2.update_xaxes(title_text='Month', row=1, col=2)
    fig2.update_yaxes(title_text='Megawatts (MW)', row=1, col=2)

    # Create the third plot (Wind Speed vs Demand)
    fig3 = px.scatter(data, x='wind_speed50_ave(m/s)', y='total_demand(mw)', 
                      title='Wind Speed vs Electricity Demand',
                      labels={'wind_speed50_ave(m/s)': 'Average Wind Speed (m/s)', 
                              'total_demand(mw)': 'Total Demand (MW)'})

    # Create the fourth plot (Time Series of Demand)
    fig4 = px.line(data, x='year', y='total_demand(mw)',
               title='Electricity Demand Over Time',
               labels={'year': 'Year', 'total_demand(mw)': 'Total Demand (MW)'})

    # Customize the layout
    fig4.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Demand (MW)",
        font=dict(size=12),
        hovermode="x unified",
        legend_title_text='',
        plot_bgcolor="white",
    )

    # Customize the traces
    fig4.update_traces(
        line=dict(color='#1f77b4', width=2),
        mode='lines',
    )

    # Customize the axes
    fig4.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=2,
        linecolor='black',
    )

    fig4.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        showline=True,
        linewidth=2,
        linecolor='black',
    )
    
    return fig1, fig2, fig3, fig4  # Return all the figures you want to display

# If you want to test the function directly in this file
if __name__ == "__main__":
    fig1, fig2, fig3, fig4 = create_dashboard_plots()
    # You can add code here to display or save the figures if needed