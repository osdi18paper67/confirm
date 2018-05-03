CONFIRM -- CONFIdence-based Repetition Meter
=========================================

Dashboard for analysis of CloudLab's benchmarking data for disk, memory, and network performance.

Key elements:
 * `main.py` is a Python program that uses [Bokeh](https://bokeh.pydata.org/en/latest/) for interactive visualizations.
 * The rest of Python files provide necessary functionality for dashboard interactions, processing of [Pandas](https://pandas.pydata.org/) dataframes and statistical analysis. 
 * Datasets with benchmarking results are stored in `csv` files in `/data`. 

## Dependencies:

Below are the currently used versions of the required Python modules:

```
bokeh==0.12.10 (newer versions have not been thoroughly tested but caused errors in some cases)
tornado==4.5.3
pandas==0.20.3
numpy==1.14.2
seaborn==0.8 (not essential but used for color palettes)
```

## Running:

Currently, the dashboard is run using: 

```
bokeh serve <.|path_to_this_directory> --show --allow-websocket-origin='*' &
``` 

After starting the app, the dashboard should be available at: `http://<hostname>:5006/confirm`
