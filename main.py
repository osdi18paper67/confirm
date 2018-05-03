''' CONFIRM dashboard

Use the ``bokeh serve`` command to run the app from its directory:

    bokeh serve .

at your command prompt. To allow external access, consider running:

    bokeh serve . --allow-websocket-origin='*'

Then navigate to the URL

    http://<hostname>:5006/
'''
try:
    from functools import lru_cache
except ImportError:
    # Python 2 does stdlib does not have lru_cache so let's just
    # create a dummy decorator to avoid crashing
    print ("WARNING: Cache for this example is available on Python 3 only.")
    def lru_cache():
        def dec(f):
            def _(*args, **kws):
                return f(*args, **kws)
            return _
        return dec

import os, datetime, time
from os.path import dirname, join
import pandas as pd
from functools import partial
from datetime import datetime as dt
import numpy as np
import seaborn as sns
import operator as op
import copy
from bokeh.client import push_session

import bokeh.io as io
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure
from bokeh.models.widgets import CheckboxButtonGroup, CheckboxGroup
from bokeh.models.widgets import Div
from bokeh.layouts import widgetbox
from bokeh.models.widgets import RadioButtonGroup
from bokeh.models.renderers import GlyphRenderer
from bokeh.models import Range1d
from bokeh.colors import RGB
from bokeh.models import Span
from bokeh.models.glyphs import ImageURL
from bokeh.models import DatetimeTickFormatter
from bokeh.models import Title
from bokeh.models import HoverTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import Legend
import multiprocessing as mp

from dataframe_tools import *
from misc_tools import *
from selectors import *
from plotting import *


# Define useful defaults
full_w = 1200
col_w = 600
tools = 'pan,wheel_zoom,reset,save'
allowed_err = 0.01

default_plot_width=600
default_plot_height=400

#post_selection_label_style = {"font-stle": "italic", "font-weight": "bold", "color": "#85144b"}

def column2val(df, column):
    """ For a given dataframe and column name, return the only unique value in the column;
     If there are multiple unique values, print an error and return the first one.
    """
    vals = df[column].unique()
    if len(vals) == 1:
        return vals[0]
    else:
        print "Error: multiple unique values are found in column: %s" % column
        return vals

def get_cmap(df, column, palette="bright", custom=None):
    """ Assign unique colors to unique values in the specified column in the dataframe.
    If custom dict is provided, add it to the resulting color map.
    """
    color_labels = df[column].unique().tolist()
    rgb_values = sns.color_palette(palette, len(color_labels))
    color_map = dict(zip(color_labels, rgb_values))
    if custom:
        color_map.update(custom)
    return color_map

# Load data from csv files into dataframes
def load_data(dir="data/"):
    data = {}
    for f in os.listdir(dir):
        if ".csv" in f:
            try:
                df = pd.read_csv(dir + f)
                if "timestamp" in df.columns:
                    # Convert epoch timestamps to proper datetimes
                    df["timestamp"] = df["timestamp"].apply(lambda x: dt.fromtimestamp(x))
                data[os.path.splitext(f)[0]] = df
            except:
                print "Could not load a csv file: %s. Skipping it." % (dir + f)
    return data

# For bokeh plot, display x axis labels as datetimes
def timestamps_on_x(p):
    """ Set datetime formatter for x axis on the given figure object """
    p.xaxis.formatter=DatetimeTickFormatter(
            hours=["%d %B %Y"],
            days=["%d %B %Y"],
            months=["%d %B %Y"],
            years=["%d %B %Y"])

def dataset_callback(attr, old, new):
    selected_dataset = dataset.options[new]

    global curr_selectors
    curr_selectors = selectors[selected_dataset]

    data = all_data[selected_dataset]

    if selected_dataset in timelines.keys():
        # If available, use loaded timeline instead of recomputing it
        timeline = timelines[selected_dataset]
    else:
        # If unavailable, create and save it
        timeline = count_timeline(data)
        timeline.to_csv("timelines/%s.csv" % selected_dataset, index=False)

    # Remove old data from timeline plot
    # Example: https://stackoverflow.com/questions/46480571/deleting-line-from-figure-in-bokeh
    old = timeline_plot.select_one({'name': 'default'})
    if old:
        timeline_plot.renderers.remove(old)

    timeline_plot.vbar(x=timeline.Day, top=timeline.Count, width=10.0, alpha=0.4, name='default')
    timestamps_on_x(timeline_plot)

    timeline_plot_subtitle.text = "Total number of measurements: %s" % (format(len(data), ','))
    timeline_plot_title.text = "%s Data: number of collected measurements by day" % (selected_dataset.capitalize())

    #layout.children[-1] = column([s.widgets() for s in curr_selectors])
    layout.children[2].children = []
    for idx in range(len(curr_selectors)):
        layout.children[2].children.append(curr_selectors[idx].widgets())

    update_plots()

def y_at_zero_callback(attr, old, new, plot_obj):
    #spinner = Div(text="", css_classes=["tuner-loader"])
    #row.children[0].children.append(spinner)

    curr_range = plot_obj.p.y_range
    if new == [0]:
        #curr_range.start = 0
        #curr_range.end = curr_selectors[-1].post_df["mean"].max() * 1.05
        curr_range.start = 0
        curr_range.end = plot_obj.y_end
    else:
        #curr_range.start = curr_selectors[-1].post_df["mean"].min() * 0.995
        #curr_range.end = curr_selectors[-1].post_df["mean"].max() * 1.005
        curr_range.start = plot_obj.y_start
        curr_range.end = plot_obj.y_end

    #row.children[0].children = row.children[0].children[:-1]
    return

def allowed_err_callback(attr, old, new, plot_obj):

    spinner = Div(text="", css_classes=["tuner-loader"])
    layout.children[2].children[-1].children[0].children.append(spinner)
    layout.children[2].children[-1].children = layout.children[2].children[-1].children[:1]

    plot_df = curr_selectors[-1].post_df

    dynamic_aggr_ci.plot(plot_df, curr_selectors, saved_analysis)

    # Assume that dynamic_aggr_ci is last element in dynamic column in layout
    new_widgets = dynamic_aggr_ci.widgets()
    dynamic_aggr_ci.stop_spinner()

    for idx in range(1, len(new_widgets.children)):
        layout.children[2].children[-1].children.append(new_widgets.children[idx])

    # Remove spinner
    layout.children[2].children[-1].children[0].children = layout.children[2].children[-1].children[0].children[:-1]

def trial_count_callback(attr, old, new, plot_obj):
    # Basically, the same code as allowed_err_callback(): redraw plot, replace widgets, add&remove a spinner

    spinner = Div(text="", css_classes=["tuner-loader"])
    layout.children[2].children[-1].children[0].children.append(spinner)
    layout.children[2].children[-1].children = layout.children[2].children[-1].children[:1]

    plot_df = curr_selectors[-1].post_df

    dynamic_aggr_ci.plot(plot_df, curr_selectors, saved_analysis)

    # Assume that dynamic_aggr_ci is last element in dynamic column in layout
    new_widgets = dynamic_aggr_ci.widgets()
    dynamic_aggr_ci.stop_spinner()

    for idx in range(1, len(new_widgets.children)):
        layout.children[2].children[-1].children.append(new_widgets.children[idx])

    # Remove spinner
    layout.children[2].children[-1].children[0].children = layout.children[2].children[-1].children[0].children[:-1]

def resample_callback():
    # Similar code as above; in redrawing the plot, make sure that resampling (gen new data, don't used saved results) is forced

    spinner = Div(text="", css_classes=["tuner-loader"])
    layout.children[2].children[-1].children[0].children.append(spinner)
    layout.children[2].children[-1].children = layout.children[2].children[-1].children[:1]

    plot_df = curr_selectors[-1].post_df

    dynamic_aggr_ci.plot(plot_df, curr_selectors, saved_analysis, force_resample=True)

    # Assume that dynamic_aggr_ci is last element in dynamic column in layout
    new_widgets = dynamic_aggr_ci.widgets()
    dynamic_aggr_ci.stop_spinner()

    for idx in range(1, len(new_widgets.children)):
        layout.children[2].children[-1].children.append(new_widgets.children[idx])

    # Remove spinner
    layout.children[2].children[-1].children[0].children = layout.children[2].children[-1].children[0].children[:-1]

def selector_callback(attr, old, new, criterion):

    global being_processed

    if being_processed:
        #print "selector_callback(): skipping because being_processed flag is set."
        pass
    else:
        being_processed = True

        # Preserve current states, options, and selections
        states = [s.state for s in curr_selectors]
        options = [s.options for s in curr_selectors]
        idxes = [s.value_idx for s in curr_selectors]

        all_cr = [s.criterion for s in curr_selectors]

        criterion_idx = all_cr.index(criterion)
        criterion_selector = curr_selectors[criterion_idx]

        layout.children[2].children = layout.children[2].children[:criterion_idx+1]
        criterion_selector.w_spinner.css_classes = ["small-loader"]

        criterion_selector.select(new)

        idx = criterion_idx + 1
        while idx < len(curr_selectors):
            #print "selector_callback(): about to UPDATE selector with idx: %d" % idx
            new_state = curr_selectors[idx].update()
            idx += 1

        # Important: must replace existing widgets with updated widgets in order to
        # avoid the issue where widgets are hidden
        for idx in range(criterion_idx+1, len(curr_selectors)):
            layout.children[2].children.append(curr_selectors[idx].widgets())

        criterion_selector.w_spinner.css_classes = ["hidden"]

        update_plots()

        being_processed = False

def update_plots():

    curr_selectors[-1].w_spinner.css_classes = ["small-loader"]

    if set([s.is_selected() for s in curr_selectors]) == set([True]):

        plot_df = curr_selectors[-1].post_df

        scatter.plot(plot_df)
        static_node_ci.plot(plot_df, curr_selectors, saved_analysis)
        dynamic_node_ci.plot(plot_df, curr_selectors, saved_analysis, allowed_err=0.01)
        dynamic_aggr_ci.plot(plot_df, curr_selectors, saved_analysis)

        layout.children[2].children.append(scatter.widgets())
        layout.children[2].children.append(static_node_ci.widgets())
        layout.children[2].children.append(dynamic_node_ci.widgets())
        layout.children[2].children.append(dynamic_aggr_ci.widgets())

        static_node_ci.stop_spinner()
        scatter.stop_spinner()
        dynamic_node_ci.stop_spinner()
        dynamic_aggr_ci.stop_spinner()

    curr_selectors[-1].w_spinner.css_classes = ["hidden"]

def pre_process_data(all_data):
    """ Perform all necessary pre-processing before data is used"""

    # Replace wisc with wisconsin
    for t,df in all_data.iteritems():
        df["site"] = df["site"].replace(to_replace="wisc", value="wisconsin")

    all_data["network"]["rack_local"] = all_data["network"]["rack_local"].map({True: "yes", False: "no"})

def pre_process_timelines(timelines):
    """ Perform all necessary pre-processing for timelines"""

    # Datetime conversion
    for t,df in timelines.iteritems():
        df["Day"] = df["Day"].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))

def toggle_callback(new, sel_obj):

    if len(sel_obj.w_selector.active) < len(sel_obj.w_selector.labels):
        selector_callback("active", [], range(len(sel_obj.w_selector.labels)), sel_obj.criterion)
    else:
        selector_callback("active", [], [], sel_obj.criterion)

def generate_selectors(all_data):
    """ Called only once -- initially to create empty/disabled selector. """

    site_info = """CloudLab sites are described at:<br /><a target='_blank' href='https://www.cloudlab.us/hardware.php'>https://www.cloudlab.us/hardware.php</a>"""
    hw_type_info = """Available hardware types and their specs:<br /><a target='_blank' href='http://docs.cloudlab.us/hardware.html'>http://docs.cloudlab.us/hardware.html</a>"""
    socket_num_info = """<i>socket_num:</i><br/>
    This parameter refers to which CPU socket we bound our workloads to.  In the case of
    multi-socket machines, we restrict our workloads to each CPU socket at a time for
    both single-threaded and multi-threaded workloads."""
    dvfs_info = """<i>dvfs:</i><br/>
    Stands for dynamic voltage and frequency scaling.  Changing the state of dvfs in
    the context of our benchmark refers to enabling/disabling Intel's Turbo Boost
    technology where applicable, and whether or not we changed the default
    performance governor setting for the clock frequency."""
    iodepth_info = """<i>iodepth:</i><br/>
        Refers to how many I/Os are issued to the target device at any given time.  We
        set this either to a single (1) I/O at a time, or many (4096) I/Os at a time."""
    disk_test_info = """<i>read</i>: issues 4k sequential read requests.<br/>
                       <i>randread</i>: issues 4k random read requests.<br/>
                       <i>write</i>: issues 4k sequential write requests.<br/>
                       <i>randwrite</i>: Issues 4k random write requests."""
    node_info = """<p>&#8987; Processing takes some time when <u>all or many</u> nodes are selected.</p>"""
    membench_info = """<p>Benchmarks are described in the <a target="_blank" href='https://docs.google.com/document/d/1S0m0jteck_hB1USX6ms7yOT3iEP-HK864oj93eU1kZ0/'> document</a>.</p>"""
    storage_info = """<p>Storage devices are described in the <a target="_blank" href='https://docs.google.com/document/d/17DpAW1PQBxZQZ-CgVZltSvgGiegonLAu7ks5moOkhik/'> document</a>.</p>"""
    network_test_info = """<p><i>latency</i>: Runs a simple ICMP ping with a default size of 56 bytes in flood mode over a
                        shared VLAN to a fixed destination.  This test is only run in the forward
                       direction.</p>
                       <p><i>bandwidth</i>: Sends 128kB blocks over TCP to a fixed destination for 60 seconds to try to
                       achieve maximum bandwidth.  This test is run in both directions.</p>"""
    directionality_info = """<p><i>forward</i>: Selected node being tested acts as the sender, and the fixed destination node acts
                           as the receiving server.</p>
                           <p><i>reverse</i>: The node being tested acts as the receiving server, and the fixed destination
                           node acts as the sender (avaliable only for the bandwidth test).</p>"""
    rack_local_info = """<p><i>rack_local</i>: Whether or not the node being tested shares a top-of-rack switch with the fixed destination node.</p>"""

    selectors = {}
    ds1 = Selector("site",     prev_selector=None, step_number=1, info=site_info, selector_callback=partial(selector_callback, criterion="site"))
    ds2 = Selector("hw_type",  prev_selector=ds1,  step_number=2, info=hw_type_info, selector_callback=partial(selector_callback, criterion="hw_type"))
    ds3 = Selector("device",   prev_selector=ds2,  step_number=3, info=storage_info, selector_callback=partial(selector_callback, criterion="device"))
    ds4 = Selector("testname", prev_selector=ds3,  step_number=4, info=disk_test_info, selector_callback=partial(selector_callback, criterion="testname"))
    ds5 = Selector("iodepth",  prev_selector=ds4,  step_number=5, info=iodepth_info, numeric=True, selector_callback=partial(selector_callback, criterion="iodepth"))
    ds6 = Selector("nodeid",   prev_selector=ds5,  step_number=6, info=node_info, multiple_values=True, toggle_callback=toggle_callback, selector_callback=partial(selector_callback, criterion="nodeid"))
    ds1.enable(all_data["disk"])
    selectors["disk"] = [ds1, ds2, ds3, ds4, ds5, ds6]

    ms1 = Selector("site",       prev_selector=None, step_number=1, info=site_info, selector_callback=partial(selector_callback, criterion="site"))
    ms2 = Selector("hw_type",    prev_selector=ms1,   step_number=2, info=hw_type_info, selector_callback=partial(selector_callback, criterion="hw_type"))
    ms3 = Selector("testname",   prev_selector=ms2,   step_number=3, info=membench_info, selector_callback=partial(selector_callback, criterion="testname"))
    ms4 = Selector("dvfs",       prev_selector=ms3,   step_number=4, info=dvfs_info, selector_callback=partial(selector_callback, criterion="dvfs"))
    ms5 = Selector("socket_num", prev_selector=ms4,   step_number=5, numeric=True, info=socket_num_info, selector_callback=partial(selector_callback, criterion="socket_num"))
    ms6 = Selector("nodeid",     prev_selector=ms5,   step_number=6, info=node_info, multiple_values=True, toggle_callback=toggle_callback, selector_callback=partial(selector_callback, criterion="nodeid"))
    ms1.enable(all_data["memory"])
    selectors["memory"] = [ms1, ms2, ms3, ms4, ms5, ms6]

    ns1 = Selector("site",            prev_selector=None, step_number=1, info=site_info, selector_callback=partial(selector_callback, criterion="site"))
    ns2 = Selector("hw_type",         prev_selector=ns1,   step_number=2, info=hw_type_info, selector_callback=partial(selector_callback, criterion="hw_type"))
    ns3 = Selector("rack_local",      prev_selector=ns2,   step_number=3, info=rack_local_info, selector_callback=partial(selector_callback, criterion="rack_local"))
    ns4 = Selector("test",            prev_selector=ns3,   step_number=4, info=network_test_info, selector_callback=partial(selector_callback, criterion="test"))
    ns5 = Selector("directionality",  prev_selector=ns4,   step_number=5, info=directionality_info, selector_callback=partial(selector_callback, criterion="directionality"))
    ns6 = Selector("nodeid",          prev_selector=ns5,   step_number=6, info=node_info, multiple_values=True, toggle_callback=toggle_callback, selector_callback=partial(selector_callback, criterion="nodeid"))
    ns1.enable(all_data["network"])
    selectors["network"] = [ns1, ns2, ns3, ns4, ns5, ns6]

    return selectors

# -----------------------------------------------------------------------------

# Various global flags
being_processed = False
plots_drawn = False

# Load and preprocess data, including available timeline
# (timeline is a saved dataframe with number of samples per day for each dataset;
# if a timeline can't be found, it will be created and saved for future use / fast loading)
all_data = load_data(dir="data/")
pre_process_data(all_data)
timelines = load_data(dir="timelines/")
pre_process_timelines(timelines)

saved_analysis = load_data(dir="saved-analysis/")

selectors = generate_selectors(all_data)
curr_selectors = []

# Create plot objects
scatter = ScatterPlot(y_at_zero_callback=y_at_zero_callback)
static_node_ci = StaticNodeCIPlot(y_at_zero_callback=y_at_zero_callback)
dynamic_node_ci = DynamicNodeCIPlot(y_at_zero_callback=y_at_zero_callback)
dynamic_aggr_ci = DynamicAggregateCIPlot(y_at_zero_callback=y_at_zero_callback,
                                         allowed_err_callback=allowed_err_callback,
                                         trial_count_callback=trial_count_callback,
                                         resample_callback=resample_callback)

#layout = column([s.widgets() for s in selectors["disk"]])

# Define globally available objects

# Start the page with a header (html is loaded from a file)
with open('header.txt', 'r') as content_file:
    header_content = content_file.read()
header = Div(text="""%s""" % header_content, width=full_w)

timeline_plot = figure(title="",
                       plot_width=col_w, plot_height=350,
                       min_border_left=50, min_border_bottom=100, tools=tools)

# Plot styling
timeline_plot_subtitle = Title(text="", text_font_style="italic")
timeline_plot.add_layout(timeline_plot_subtitle, 'above')
timeline_plot_title = Title(text="")
timeline_plot.add_layout(timeline_plot_title, 'above')
timeline_plot_bottom_info = Title(text="* Varying availability of testbed resources results in the varying number of measurements.", text_font_size="8pt", text_font_style="normal")
timeline_plot.add_layout(timeline_plot_bottom_info, 'below')

layout = column(header, column(), column())

# # Simple selector for dataset
dataset = MinimalSelector("Select one of the available datasets:", sorted(all_data.keys()),
                          callback=dataset_callback)

# Trigger dataset update and save its widgets
dataset_callback("active", None, 0)
dw = dataset.widgets()

summary = row(column(dw, css_classes=["dataset-selector"]), timeline_plot)
layout.children[1] = summary

curdoc().add_root(layout)
curdoc().title = "CONFIRM dashboard"
