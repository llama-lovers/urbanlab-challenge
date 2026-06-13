FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab \
    polars \
    pyarrow \
    geopandas \
    folium \
    matplotlib \
    plotly \
    ipywidgets \
    sqlalchemy \
    psycopg2-binary \
    ipykernel

WORKDIR /home/jovyan/work

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
