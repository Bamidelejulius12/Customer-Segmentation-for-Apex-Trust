import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt



class CustomerSegmentPerformanceAnalyzer:
    def __init__(self):
        self.colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    def plot_normalized_segment_radar_chart(self, cluster_profiles):

        # Normalize the data for radar chart
        metrics_to_plot = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_Account_Balance']
        normalized = cluster_profiles[metrics_to_plot].copy()
        for col in metrics_to_plot:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

        # Plotly interactive radar chart
        fig = go.Figure()
        for segment in cluster_profiles.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized.loc[segment, metrics_to_plot].values,
                theta=metrics_to_plot,
                fill='toself',
                name=segment
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Segment Profile Comparison (Normalized)",
            showlegend=True
        )
        fig.show()
    
    
    def plot_segment_revenue_distribution(self, cluster_profiles: pd.DataFrame):
        plt.figure(figsize=(12, 8))

        # Create donut chart
        wedges, texts, autotexts = plt.pie(cluster_profiles['Total_Monetary_Contribution'],
                                            labels=None,  # No labels on the chart
                                            autopct=lambda p: f'{p:.1f}%',
                                            colors=self.colors_contrib,
                                            explode=[0.05, 0.05, 0, 0],
                                            shadow=True,
                                            pctdistance=0.85,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})

        # Add a circle in the middle to make it a donut
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)

        # Make percentage text white and bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        # Create legend with detailed info
        legend_labels = []
        for segment in cluster_profiles.index:
            total = cluster_profiles.loc[segment, 'Total_Monetary_Contribution']
            pct = total / cluster_profiles['Total_Monetary_Contribution'].sum() * 100
            legend_labels.append(f'{segment}: ${total:,.0f} ({pct:.1f}%)')

        plt.legend(wedges, legend_labels, title="Segments",
                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=10)

        plt.title('Revenue Distribution by Customer Segment', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def plot_revenue_vs_customer_comparison(self, cluster_profiles):
        """
        Side-by-side bar chart comparing revenue contribution vs customer contribution
        across customer segments.
        """

        # Copy to avoid modifying original data
        df = cluster_profiles.copy()

        # Ensure cluster names exist
        df['Cluster_Name'] = df.index

        # Create total revenue
        df['Total_Revenue'] = df['Avg_Monetary'] * df['Customer_Count']

        # Normalize values for fair comparison
        revenue = df['Total_Revenue']
        customers = df['Customer_Count']

        revenue_norm = revenue / revenue.sum()
        customers_norm = customers / customers.sum()

        # Sort by revenue (important for storytelling)
        df = df.sort_values('Total_Revenue', ascending=False)
        revenue_norm = revenue_norm.loc[df.index]
        customers_norm = customers_norm.loc[df.index]

        # X-axis setup
        x = np.arange(len(df))
        width = 0.35

        # Plot
        plt.figure(figsize=(12, 6))

        plt.bar(x - width/2, revenue_norm, width, label='Revenue Contribution')
        plt.bar(x + width/2, customers_norm, width, label='Customer Contribution')

        # Labels and styling
        plt.xticks(x, df['Cluster_Name'], rotation=45)
        plt.ylabel('Proportion of Total')
        plt.title('Revenue vs Customer Contribution by Segment')
        plt.legend()
        plt.tight_layout()

        plt.show()