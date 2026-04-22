import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Visualization_plots:
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

    def customer_segment_visualization(self, rfm_df: pd.DataFrame):
        segment_counts = rfm_df['Cluster_Name'].value_counts()

        fig, ax = plt.subplots(figsize=(12, 8))

        explode = [0.05] * len(segment_counts)

        ax.pie(
            segment_counts.values,
            labels=segment_counts.index,
            autopct=lambda p: f'{p:.1f}%',
            startangle=90,
            colors=self.colors[:len(segment_counts)],
            explode=explode,
            shadow=True
        )

        ax.set_title('RFM Customer Segmentation Distribution', fontsize=16, fontweight='bold')
        ax.axis('equal')

        return fig

    def customer_rfm_segment(self, cluster_profiles: pd.DataFrame):

        fig, ax = plt.subplots(figsize=(12, 8))

        rfm_metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
        x_pos = np.arange(len(cluster_profiles))
        width = 0.25

        for i, metric in enumerate(rfm_metrics):
            values = cluster_profiles[metric].values
            if metric == 'Avg_Monetary':
                values = values / 1000

            ax.bar(x_pos + i * width, values, width, label=metric.replace('Avg_', ''))

        ax.set_xlabel('Customer Segments')
        ax.set_ylabel('Values (Monetary in Thousands)')
        ax.set_title('Average RFM Values by Segment')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(cluster_profiles.index, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def customer_segment_comparison(self, cluster_profiles: pd.DataFrame):

        fig, ax = plt.subplots(figsize=(12, 8))

        segment_sizes = cluster_profiles.sort_values('Customer_Count')['Customer_Count']

        ax.barh(
            range(len(segment_sizes)),
            segment_sizes.values,
            color=self.colors[:len(segment_sizes)]
        )

        ax.set_yticks(range(len(segment_sizes)))
        ax.set_yticklabels(segment_sizes.index)
        ax.set_xlabel('Number of Customers')
        ax.set_title('Segment Size Comparison')

        for i, v in enumerate(segment_sizes.values):
            ax.text(v + 100, i, f'{v:,}', va='center', fontweight='bold')

        return fig