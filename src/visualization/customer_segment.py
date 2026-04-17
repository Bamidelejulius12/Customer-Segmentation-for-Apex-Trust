import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn
class Visualization_plots:
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    def customer_segment_visualization(self, rfm_df: pd.DataFrame):

        # --- Big Pie Chart for Customer Distribution ---
        segment_counts = rfm_df['Cluster_Name'].value_counts()
        plt.figure(figsize=(12, 8))
        explode = [0.05] * len(segment_counts)
        plt.pie(segment_counts.values,
                labels=segment_counts.index,
                autopct=lambda p: f'{p:.1f}%\n({int(p*sum(segment_counts.values)/100):,})',
                startangle=90,
                colors=self.colors[:len(segment_counts)],
                explode=explode,
                shadow=True)
        plt.title('RFM Customer Segmentation Distribution\n', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.show()


    def customer_rfm_segment(cluster_profiles: pd.DataFrame):
            
        # --- Bar Chart - Average RFM by Segment ---
        plt.figure(figsize=(10, 6))
        rfm_metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
        x_pos = np.arange(len(cluster_profiles))
        width = 0.25

        for i, metric in enumerate(rfm_metrics):
            values = cluster_profiles[metric].values
            if metric == 'Avg_Monetary':
                values = values / 1000  # Thousands
            plt.bar(x_pos + i*width, values, width, label=metric.replace('Avg_', ''), alpha=0.8)

        plt.xlabel('Customer Segments')
        plt.ylabel('Values (Monetary in Thousands)')
        plt.title('Average RFM Values by Segment')
        plt.xticks(x_pos + width, cluster_profiles.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def customer_segment_comparison(self, cluster_profiles: pd.DataFrame):
            
        # --- Horizontal Bar Chart - Segment Sizes ---
        plt.figure(figsize=(10, 6))
        segment_sizes = cluster_profiles.sort_values('Customer_Count')['Customer_Count']
        plt.barh(range(len(segment_sizes)), segment_sizes.values, color=self.colors[:len(segment_sizes)])
        plt.yticks(range(len(segment_sizes)), segment_sizes.index)
        plt.xlabel('Number of Customers')
        plt.title('Segment Size Comparison')
        for i, v in enumerate(segment_sizes.values):
            plt.text(v + 100, i, f'{v:,}', va='center', fontweight='bold')
        plt.tight_layout()
        plt.show()

