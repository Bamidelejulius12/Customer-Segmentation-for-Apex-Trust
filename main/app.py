import matplotlib
matplotlib.use("Agg")  
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from functools import lru_cache
import matplotlib.pyplot as plt  
import base64
import io


from src.modelling.segment import segment_engine
from src.visualization.customer_segment import Visualization_plots
from src.visualization.customer_segment_performance import CustomerSegmentPerformanceAnalyzer
from src.modelling.clusters import clustering_engine
app = FastAPI(title="Customer Segmentation API")
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format= "%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize visualization classes
viz = Visualization_plots()
perf_viz = CustomerSegmentPerformanceAnalyzer()


@lru_cache()
def get_pipeline_data():
    return segment_engine.cluster_grouper()

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64

@app.post("/refresh")
def refresh_data():
    get_pipeline_data.cache_clear()
    return {"message": "Cache cleared. Data will be recomputed on next request."}


segmented_rfm_data, customer_segment_data = get_pipeline_data()


@app.get("/segments")
def get_segments():
   

    return {
        "segmented_data": segmented_rfm_data.to_dict(orient="records"),
        "cluster_summary": customer_segment_data.reset_index().to_dict(orient="records")
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():

    fig1 = viz.customer_segment_visualization(segmented_rfm_data)
    distribution = fig_to_base64(fig1)

    fig2 = viz.customer_rfm_segment(customer_segment_data)
    rfm = fig_to_base64(fig2)

    fig3 = viz.customer_segment_comparison(customer_segment_data)
    segment_size = fig_to_base64(fig3)

    fig4 = perf_viz.plot_segment_revenue_distribution(customer_segment_data)
    revenue = fig_to_base64(fig4)

    fig5 = perf_viz.plot_revenue_vs_customer_comparison(customer_segment_data)
    revenue_vs_customers = fig_to_base64(fig5)

    radar = perf_viz.plot_normalized_segment_radar_chart(customer_segment_data)

    return f"""
    <html>
        <head>
            <title>Customer Dashboard</title>
        </head>
        <body>
            <h2>Customer Distribution</h2>
            <img src="data:image/png;base64,{distribution}" />

            <h2>RFM Analysis</h2>
            <img src="data:image/png;base64,{rfm}" />

            <h2>Segment Size</h2>
            <img src="data:image/png;base64,{segment_size}" />

            <h2>Revenue</h2>
            <img src="data:image/png;base64,{revenue}" />

            <h2>Revenue vs Customers</h2>
            <img src="data:image/png;base64,{revenue_vs_customers}" />

            <h2>Radar Chart</h2>
            {radar}
        </body>
    </html>
    """

@app.get("/retrain")
def retrain():
    try:
        cluster_engine = clustering_engine()
        cluster_engine.train_and_log_model()
        logging.info("Model has been succesfully trained")
    except Exception as e:
        logging.info("error occurred while training model, {e}")