import plotly.express as px

def plot_correlation(corr_matrix):
    if corr_matrix is None or corr_matrix.empty:
        return None

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Stocks",
        yaxis_title="Stocks"
    )
    return fig
