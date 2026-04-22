import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go


class CustomerSegmentPerformanceAnalyzer:
    def __init__(self):
        self.colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    def plot_segment_revenue_distribution(self, cluster_profiles: pd.DataFrame):
    
        fig, ax = plt.subplots(figsize=(12, 8))
        df = cluster_profiles.copy()
        df['Total_Monetary_Contribution'] = df['Avg_Monetary'] * df['Customer_Count']
        
        # Create dynamic explode - highlight top 2 segments
        n_segments = len(df)
        explode = [0.05 if i < 2 else 0 for i in range(n_segments)]
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(
            df['Total_Monetary_Contribution'],
            labels=None,
            autopct=lambda p: f'{p:.1f}%',
            colors=self.colors_contrib,
            explode=explode,
            shadow=True,
            pctdistance=0.85,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Add a circle in the middle to make it a donut
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Make percentage text white and bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # Create legend with detailed info
        legend_labels = []
        for segment in cluster_profiles.index:
            total = df.loc[segment, 'Total_Monetary_Contribution']
            pct = total / df['Total_Monetary_Contribution'].sum() * 100
            legend_labels.append(f'Segment {segment}: ${total:,.0f} ({pct:.1f}%)')
        
        plt.legend(wedges, legend_labels, title="Segments",
                loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                fontsize=10)
        
        ax.set_title('Revenue Distribution by Customer Segment', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig

    def plot_revenue_vs_customer_comparison(self, cluster_profiles):

        df = cluster_profiles.copy()
        df['Cluster_Name'] = df.index
        df['Total_Revenue'] = df['Avg_Monetary'] * df['Customer_Count']

        revenue = df['Total_Revenue']
        customers = df['Customer_Count']

        revenue_norm = revenue / revenue.sum()
        customers_norm = customers / customers.sum()

        df = df.sort_values('Total_Revenue', ascending=False)

        revenue_norm = revenue_norm.loc[df.index]
        customers_norm = customers_norm.loc[df.index]

        x = np.arange(len(df))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.bar(x - width/2, revenue_norm, width, label='Revenue Contribution')
        ax.bar(x + width/2, customers_norm, width, label='Customer Contribution')

        ax.set_xticks(x)
        ax.set_xticklabels(df['Cluster_Name'], rotation=45)
        ax.set_ylabel('Proportion of Total')
        ax.set_title('Revenue vs Customer Contribution by Segment')
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def plot_normalized_segment_radar_chart(self, cluster_profiles):

        metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_Account_Balance']

        normalized = cluster_profiles[metrics].copy()

        for col in metrics:
            normalized[col] = (
                (normalized[col] - normalized[col].min()) /
                (normalized[col].max() - normalized[col].min())
            )

        fig = go.Figure()

        for segment in cluster_profiles.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized.loc[segment, metrics].values,
                theta=metrics,
                fill='toself',
                name=segment
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Segment Profile Comparison (Normalized)"
        )

        return fig.to_html(full_html=False)