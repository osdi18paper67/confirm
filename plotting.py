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
from bokeh.models.glyphs import Text
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
from bokeh.models.widgets import RadioGroup
from bokeh.models.widgets import Button
from bokeh.io import export_svgs

import json
import hashlib

from dataframe_tools import *

def col2RGB(c):
    # From color in seaborn and matplotlib to RGB color in bokeh
    return RGB(int(255*c[0]), int(255*c[1]), int(255*c[2]))

def my_hash(x):
    # Return a unique hash -- returns even when tmp is a dict with nested lists and dicts
    json_str = json.dumps(x, sort_keys=True)
    return hashlib.md5(json_str).hexdigest()

def unique_dataset_hash(curr_selectors, additional_params=None):
    # Hash for a dict with selections and optional additional_params
    selected = {s.criterion: s.value for s in curr_selectors}
    if additional_params:
        selected.update(additional_params)
    return my_hash(selected)

class GenPlot(object):
    """ Generic plot class that includes control widgets, plot itself, and information label """

    def __init__(self, source=None, y_at_zero_callback=None):
        self.source = source
        self.is_plotted = False

        # Label desribing this plot
        self.info = Div(text="")

        # Spinner
        self.spinner = Div(text="")

        # Figure object and its defaults
        self.p = figure(plot_width=550, plot_height=400)
        self.tools = 'pan,wheel_zoom,reset,save'
        self.toolbar_location = "above"

        # Widgets for asjusting/customizing this plot
        self.y_at_zero = CheckboxButtonGroup(labels=["Start y at 0"], active=[])

        # For a callback responsible for processing clicks on the "Start y at 0" widget,
        # make sure that the specified callback function y_at_zero_callback gets an additional argument -- instance of this class
        if y_at_zero_callback:
            self.y_at_zero.on_change("active", partial(y_at_zero_callback, plot_obj=self))

        self.y_start = None
        self.y_end = None
        # Fraction added to min and max y values to set y_start and y_end for the plots
        self.y_padding = 0.03

        # To apply css styles, unhide/show all widgets
        self.style()

    def plot(self):
        """ Create and populate plot object appropriately"""
        self.is_plotted = True

        # Derived classes will implement custom plotting logic here
        # FIXME: REMOVE TEST Example

    # DEPRECATED:
    # def update(self, new_df):
    #     """ Modify plot's data source source """
    #     self.source = self.df2source(new_df)

    def widgets(self):
        """ Return row or column of elements/widgets to be added to the main document """

        self.spinner.css_classes=["plot-loader"]
        return row(column(self.y_at_zero, width=150), self.p, column(self.spinner, width=50), self.info)

    def stop_spinner(self):
        self.spinner.css_classes=["hidden"]

    def style(self):
        self.y_at_zero.css_classes = ["plot-tuner"]
        self.p.css_classes = []
        self.info.css_classes = ["infolabel"]

    def empty_plot(self, title, text):

        self.p = figure(title=title, toolbar_location=self.toolbar_location,
                                   x_range = Range1d(0,100), y_range = Range1d(0,100),
                                   plot_width=550, plot_height=400, tools=[])

        text_source = ColumnDataSource(dict(x=[50], y=[50], text=[text]))
        glyph = Text(x="x", y="y", text="text", angle=0.0, text_align="center", text_color="#FF851B")
        self.p.add_glyph(text_source, glyph)

        self.p.xgrid.visible = False
        self.p.ygrid.visible = False
        self.p.xaxis.visible = False
        self.p.yaxis.visible = False

        self.info.css_classes = ["hidden"]
        self.y_at_zero.css_classes = ["hidden"]

class ScatterPlot(GenPlot):

    def __init__(self, source=None, y_at_zero_callback=None):
        super(ScatterPlot, self).__init__(source=source, y_at_zero_callback=y_at_zero_callback)

        # Customization for this specific class
        self.info.text = """<p>Scatter plot shows all selected measurements at moments in time when they were collected.</p>
        <p>This plot helps visually inspect the distribution of selected measurements and detect outliers and potential skew.
        Additionally, it can help find intervals of time with rare or no measurements.</p>"""

    def df2source(self, df):
        # Specifically for this plot, manipulate data from "raw" dataframe to create a minimalistic Bokeh's data source

        source_df = df[["timestamp", "mean", "nodeid"]].copy()
        cmap = get_cmap(df, "nodeid", palette="hls")
        source_df["color"] = source_df["nodeid"].apply(lambda x: col2RGB(cmap[x]))
        return ColumnDataSource(source_df)

    def plot(self, df):
        self.is_plotted = True

        self.p = figure(title="Selected Measurements",
                        plot_width=550, plot_height=400, tools=self.tools, toolbar_location = self.toolbar_location,
                        min_border_left=0, min_border_bottom=100)

        # Additional styling
        self.p.xaxis.major_label_orientation = 3.14/2
        if "units" in df.columns:
            self.p.yaxis[0].axis_label = str(column2val(df, "units"))
        self.p.yaxis[0].formatter.use_scientific = False
        self.p.xaxis.formatter=DatetimeTickFormatter(
                hours=["%d %B %Y"],
                days=["%d %B %Y"],
                months=["%d %B %Y"],
                years=["%d %B %Y"],
            )

        # Create reduced dataframe (that includes only necessary columns) and convert it to Bokeh data object for plotting
        self.source = self.df2source(df)

        # Plot prepared data object
        if len(df.nodeid.unique()) <= 10:
            self.p.scatter(x='timestamp', y='mean', legend='nodeid', source=self.source,
                      size=8, fill_color='color', fill_alpha=1.0, line_color=None)

            # There should be a single legend object; get its elements and hide it
            curr_items = self.p.legend[0].items
            self.p.legend[0].visible = False

            # Create a new legend outside the plot
            self.p.add_layout(Legend(items=curr_items, location=(0, 0)), 'right')
        else:
            self.p.scatter(x='timestamp', y='mean', source=self.source,
                      size=8, fill_color='color', fill_alpha=1.0, line_color=None)

            if "Selected dataset includes measurements" not in self.info.text:
                self.info.text += """<p>Selected dataset includes measurements for %d nodes (too many to label them individually in a legend).</p>"""% len(df.nodeid.unique())

        self.y_start = df["mean"].min()
        self.y_end = df["mean"].max()
        y_range = self.y_end - self.y_start
        self.y_start -= y_range * self.y_padding
        self.y_end += y_range * self.y_padding


class StaticNodeCIPlot(GenPlot):

    def __init__(self, source=None, y_at_zero_callback=None):
        super(StaticNodeCIPlot, self).__init__(source=source, y_at_zero_callback=y_at_zero_callback)

        # Customization for this specific class
        self.info.text = """<p>Confidence intervals (CIs) help perform node-to-node comparisons.
        <p>This plot visualizes confidence intervals and medians for nodes that include <u>at least 10 measurements</u>
        (confidence intervals for smaller subsets of measurements can't be considered reliable).</p>
        <p>Asymmetric 95% confidence intervals are constructed using nonparametric statistics and shown as vertical bars.</p>
        <p>Yellow diamonds inside these bars depict node-specific median values.</p>"""

    def load_or_compute(self, plot_df, curr_selectors, saved_analysis):

        uh = unique_dataset_hash(curr_selectors, additional_params={"type": "ci_static"})
        if uh in saved_analysis.keys():
            print "Reusing previously saved dataset for state hash:", uh
            df = saved_analysis[uh]
        else:
            print "Can't find previously saved dataset for state hash:", uh

            # Create new dataset from scratch
            cmap = get_cmap(plot_df, "nodeid", palette="hls")
            df = []
            for nodeid, grp in plot_df.groupby("nodeid"):
                c = col2RGB(cmap[nodeid])
                s_median, lo_val, hi_val, errc = median_and_ci(grp["mean"])
                if not errc:
                    df.append([nodeid, s_median, lo_val, hi_val, c])
            df = pd.DataFrame(df, columns = ["node", "med", "lo", "hi", "col"]).sort_values(by=["node"])

            # Save dataset to both global dictionary with all available datasets and to a file
            saved_analysis[uh] = df
            dest = "saved-analysis/%s.csv" % uh
            print "Created new dataset and saved it to:", dest
            df.to_csv(dest, index=False)

        return df

    def plot(self, plot_df, curr_selectors, saved_analysis):
        self.is_plotted = True

        df = self.load_or_compute(plot_df, curr_selectors, saved_analysis)
        source = ColumnDataSource(df)

        if len(df) > 0:

            self.p = figure(title="95% Confidence Intervals (CIs)",
                               x_range = df["node"].unique(),
                               plot_width=550, plot_height=400, tools=self.tools, toolbar_location=self.toolbar_location,
                               min_border_left=0, min_border_bottom=100)

            # Additional styling
            if "units" in plot_df.columns:
                self.p.yaxis[0].axis_label = str(column2val(plot_df, "units"))
            self.p.yaxis[0].formatter.use_scientific = False
            self.p.xaxis.major_label_orientation = 3.14/2

            self.p.vbar(x="node", bottom="lo", top="hi", source=source,
                        width=0.25, color="col", alpha=1.0)
            self.p.scatter(x="node", y="med", source=source,
                           marker="diamond", fill_color="#FFDC00", size=15, color="black")

            self.style()

            self.y_start = df["lo"].min()
            self.y_end = df["hi"].max()
            y_range = self.y_end - self.y_start
            self.y_start -= y_range * self.y_padding
            self.y_end += y_range * self.y_padding
        else:
            self.empty_plot("95% Confidence Intervals (CIs)",
                            "Not enough measurements to estimate.\nSelect different configuration or different node(s).")

class DynamicNodeCIPlot(GenPlot):

    def __init__(self, source=None, y_at_zero_callback=None):
        super(DynamicNodeCIPlot, self).__init__(source=source, y_at_zero_callback=y_at_zero_callback)

        # Customization for this specific class
        self.info.text = """<p>This plot shows how confidence intervals change as the number of measurements grows for each node.
        We <i>resample</i> measurements for each node as follows:
        we randomly select partial subsets of node measurements of a given size >=10 (x axis values)
        and compute confidence intervals and medians for those subsets. We repeat this estimation on <u>20 \"trials\"</u> and
        show average medians and average confidence interval bounds for each node.
        We continue this analysis until the size of the subsets reaches the total number of available measurements for each node.
        In this process, we learn how medians and confidence intervals converge as more measurements become available.</p>
        <p>Similar to the plot above, this plot shows the change in confidence intervals and medians for nodes that include <u>at least 10 measurements</u>.</p>"""

    def load_or_compute(self, plot_df, curr_selectors, saved_analysis, allowed_err):

        uh = unique_dataset_hash(curr_selectors, additional_params={"type": "ci_dynamic"})
        if uh in saved_analysis.keys():
            print "Reusing previously saved dataset for state hash:", uh
            df_avg_all = saved_analysis[uh]
        else:
            print "Can't find previously saved dataset for state hash:", uh

            # Create new dataset from scratch
            cmap = get_cmap(plot_df, "nodeid", palette="hls")
            df_indiv = pd.DataFrame()
            for nodeid, grp in plot_df.groupby("nodeid"):
                c = col2RGB(cmap[nodeid])
                #ci_node = ci_reduction(grp["mean"])
                ci_node = ci_reduction_parallel(grp["mean"])
                ci_node["nodeid"] = nodeid
                ci_node["color"] = c
                df_indiv = df_indiv.append(ci_node)

            # Average across different instantiations
            df_avg_all = pd.DataFrame()
            for (nodeid, c), grp in df_indiv.groupby(["nodeid", "color"]):
                df_avg_node = grp.groupby(['sample_size']).mean()
                df_avg_node["sample_size"] = df_avg_node.index
                df_avg_node["desired_lb"] = df_avg_node.med - allowed_err * df_avg_node.med
                df_avg_node["desired_ub"] = df_avg_node.med + allowed_err * df_avg_node.med
                df_avg_node["nodeid"] = nodeid
                df_avg_node["color"] = c
                df_avg_all = df_avg_all.append(df_avg_node)

            # Save dataset to both global dictionary with all available datasets and to a file
            saved_analysis[uh] = df_avg_all
            dest = "saved-analysis/%s.csv" % uh
            print "Created new dataset and saved it to:", dest
            df_avg_all.to_csv(dest, index=False)

        return df_avg_all

    def plot(self, plot_df, curr_selectors, saved_analysis, allowed_err=0.01):
        self.is_plotted = True

        #print "DynamicNodeCIPlot object, plot()"

        df = self.load_or_compute(plot_df, curr_selectors, saved_analysis, allowed_err)
        source = ColumnDataSource(df)

        if len(df) > 0:

            self.p = figure(title="Change of Individual CIs",
                       plot_width=550, plot_height=400, tools=self.tools, toolbar_location=self.toolbar_location,
                       min_border_left=0, min_border_bottom=100)
            # Additional styling
            if "units" in plot_df.columns:
                self.p.yaxis[0].axis_label = str(column2val(plot_df, "units"))
            self.p.yaxis[0].formatter.use_scientific = False
            self.p.xaxis[0].axis_label = "Number of Samples"

            custom_legend = []
            for (nodeid, col), grp in df.groupby(["nodeid", "color"]):
                band_x = np.append(grp["sample_size"], grp["sample_size"][::-1])
                band_y = np.append(grp["ci_lb"], grp["ci_ub"][::-1])
                r = self.p.patch(band_x, band_y, color=col, fill_alpha=0.4)
                self.p.line(grp["sample_size"], grp["med"], color=col, line_width=4)
                custom_legend.append((nodeid, [r]))

            if len(df.nodeid.unique()) > 10:
                if "Selected dataset includes measurements" not in self.info.text:
                    self.info.text += """<p>Selected dataset includes measurements for %d nodes (too many to label them individually in a legend).</p>"""% len(df.nodeid.unique())
            else:
                legend = Legend(items=custom_legend, location=(0, 0))
                self.p.add_layout(legend, 'right')

            self.y_start = df["ci_lb"].min()
            self.y_end = df["ci_ub"].max()
            y_range = self.y_end - self.y_start
            self.y_start -= y_range * self.y_padding
            self.y_end += y_range * self.y_padding

            self.style()
        else:
            self.empty_plot("Change of Individual CIs",
                            "Not enough measurements to estimate.\nSelect different configuration or different node(s).")

class DynamicAggregateCIPlot(GenPlot):

    def __init__(self, source=None, y_at_zero_callback=None, allowed_err_callback=None, trial_count_callback=None, resample_callback=None):

        self.allowed_err_label = Div(text="Allowed error:", css_classes=["increased_vspace"])
        self.allowed_err_selector = RadioGroup(labels=["1%", "5%", "10%"], active=0)

        self.trial_count_label = Div(text="Number of trials:", css_classes=["increased_vspace"])
        self.trial_count_selector = RadioGroup(labels=["20 (coarse, fast)", "100 (finer, slower)", "200 (finest, slowest)"], active=0)

        self.resample_button = Button(label="Resample", button_type="success")

        super(DynamicAggregateCIPlot, self).__init__(source=source, y_at_zero_callback=y_at_zero_callback)

        if allowed_err_callback:
            self.allowed_err_selector.on_change("active", partial(allowed_err_callback, plot_obj=self))
        if trial_count_callback:
            self.trial_count_selector.on_change("active", partial(trial_count_callback, plot_obj=self))
        if resample_callback:
            self.resample_button.on_click(resample_callback)

        # Customization for this specific class
        self.info.text = """<p>This plot shows the change in the aggregate confidence interval constructed
        for measurements for all selected nodes. Similar to the estimation for individual nodes described above,
        we <i>resample</i> selected measurements and track average median values and average confidence interval bounds.
        In the analysis of the stopping condition, we look for the point where the confidence interval fits within
        the tunable allowed error bound. Additionally, the number of averaged \"trials\" can be adjusted to
        produce more reliable estimates. Note that we start this analysis at <u>10 samples</u>, considering that
        smaller sets are insufficient for reliable estimation of confidence intervals.
        Use the Resample button to repeat this analysis on a new set of random trials.</p>"""

    def allowed_err_value(self):
        return float(self.allowed_err_selector.labels[self.allowed_err_selector.active].rstrip("%")) / 100

    def trial_count_value(self):
        return int(self.trial_count_selector.labels[self.trial_count_selector.active].split()[0])

    def widgets(self):
        """ Return row or column of elements/widgets to be added to the main document """

        self.spinner.css_classes=["plot-loader"]
        return row(column(self.y_at_zero, self.allowed_err_label, self.allowed_err_selector, self.trial_count_label, self.trial_count_selector, self.resample_button, width=150),
                   self.p, column(self.spinner, width=50), self.info)

    def remove_summary_statement(self):
        # Statement is found based on <h3> tag
        tag = "<h3>"

        t = self.info.text
        if tag in t:
            idx = t.index(tag)
            t = t[:idx]
            self.info.text = t
        else:
            # Doing nothing
            pass

    def load_or_compute(self, plot_df, curr_selectors, saved_analysis, force_resample=False):

        allowed_err = self.allowed_err_value()
        trial_count = self.trial_count_value()

        uh = unique_dataset_hash(curr_selectors, additional_params={"type": "ci_aggregate", "allowed_err": allowed_err, "trial_count": trial_count})
        if (uh in saved_analysis.keys()) and (not force_resample):
            print "Reusing previously saved dataset for state hash:", uh
            df_avg = saved_analysis[uh]
        else:
            print "Can't find previously saved dataset for state hash:", uh

            # Create new dataset from scratch

            #df_indiv = ci_reduction(plot_df["mean"], max_rep_count=trial_count)
            df_indiv = ci_reduction_parallel(plot_df["mean"], max_rep_count=trial_count)

            if len(df_indiv) > 0:
                df_avg = df_indiv.groupby(['sample_size']).mean()
                df_avg["desired_lb"] = df_avg.med - allowed_err * df_avg.med
                df_avg["desired_ub"] = df_avg.med + allowed_err * df_avg.med
                df_avg["stopping_ub"] = df_avg["ci_ub"] <= df_avg["desired_ub"]
                df_avg["stopping_lb"] = df_avg["ci_lb"] >= df_avg["desired_lb"]
                df_avg["stopping"] = df_avg["stopping_lb"] & df_avg["stopping_ub"]
                df_avg["sample_size"] = df_avg.index.values
            else:
                df_avg = pd.DataFrame()

            # Save dataset to both global dictionary with all available datasets and to a file
            saved_analysis[uh] = df_avg
            dest = "saved-analysis/%s.csv" % uh
            print "Created new dataset and saved it to:", dest
            df_avg.to_csv(dest, index=False)

            print "load_or_compute() in DynamicAggregateCIPlot: about to return"

        return df_avg

    def plot(self, plot_df, curr_selectors, saved_analysis, force_resample=False):
        self.is_plotted = True

        allowed_err = self.allowed_err_value()
        trial_count = self.trial_count_value()

        print "DynamicAggregateCIPlot object, plot()"
        df = self.load_or_compute(plot_df, curr_selectors, saved_analysis, force_resample)
        source = ColumnDataSource(df)

        if len(df) > 0:

            self.y_start = min(df["ci_lb"].min(), df["desired_lb"].min())
            self.y_end = max(df["ci_ub"].max(), df["desired_ub"].max())
            y_range = self.y_end - self.y_start
            self.y_start -= y_range * self.y_padding
            self.y_end += y_range * self.y_padding

            self.p = figure(title="Change of Overall CIs",
                      plot_width=550, plot_height=400, tools=self.tools, toolbar_location=self.toolbar_location,
                      min_border_left=0, min_border_bottom=100)
            if "units" in plot_df.columns:
                self.p.yaxis[0].axis_label = str(column2val(plot_df, "units"))
            self.p.xaxis[0].axis_label = "Number of Samples"
            self.p.yaxis[0].formatter.use_scientific = False

            custom_legend = []
            # Filled area plot example: https://github.com/bokeh/bokeh/blob/master/examples/plotting/file/bollinger.py
            band_x = np.append(df["sample_size"], df["sample_size"][::-1])
            band_y = np.append(df["ci_lb"], df["ci_ub"][::-1])
            r1 = self.p.line(df["sample_size"], df.med, line_width=4)
            custom_legend.append(("Median", [r1]))

            r2 = self.p.patch(band_x, band_y, color='#7570B3', fill_alpha=0.4)
            custom_legend.append(("95% CI", [r2]))

            r3 = self.p.line(df["sample_size"], df["desired_lb"], color='#85144b', line_width=2, line_dash="dashed")
            self.p.line(df["sample_size"], df["desired_ub"], color='#85144b', line_width=2, line_dash="dashed")
            custom_legend.append(("%.1f%% Error" % (allowed_err * 100), [r3]))

            ci_can_stop = df[df["stopping"] == True]
            if len(ci_can_stop):
                stop_at_sample_size_idx = ci_can_stop.index.min()
                stop_at_sample_size = df.get_value(stop_at_sample_size_idx, "sample_size")
            else:
                stop_at_sample_size = None

            if stop_at_sample_size:
                # Line works better than ray here
                r4 = self.p.line(x=[stop_at_sample_size, stop_at_sample_size], y=[self.y_start, self.y_end],
                                 color="red", line_dash="dotted", line_width=4)

                custom_legend.append(("Stopping Condition Is Met", [r4]))

                self.remove_summary_statement()
                self.info.text += """<h3>Recommended number of measurements:
                </h3><h1 style=\"text-align:center;\"><font color=\"red\">%d</font></h1><h3>At that point,
                95%% confidence interval fits within %.1f%% interval around the median value.</h3>""" % (stop_at_sample_size, allowed_err * 100)
            else:
                self.remove_summary_statement()
                self.info.text += """<h3>Selected %d measurements <u>do not allow</u> reaching the stopping condition
                where the 95%% confidence interval fits within %.1f%% interval around the median value.
                Consider selecting more measurements.</h3>""" % (len(plot_df), allowed_err * 100)

            #self.p.y_range = Range1d(min(df["desired_lb"].min(), df["ci_lb"].min())*0.995, max(df["desired_ub"].max(), df["ci_ub"].max())*1.005)
            self.p.y_range = Range1d(self.y_start, self.y_end)

            # Custom legend
            legend = Legend(items=custom_legend, location=(0, 0), orientation="horizontal", spacing=10)
            self.p.add_layout(legend, 'below')

            self.style()

            # Save created svg
            uh = unique_dataset_hash(curr_selectors,
                                     additional_params={"type": "ci_aggregate", "allowed_err": allowed_err, "trial_count": trial_count})
            self.p.output_backend = "svg"
            export_svgs(self.p, filename="saved-plots/%s.svg" % uh)
        else:
            self.empty_plot("Change of Overall CIs",
                            "Not enough measurements to estimate.\nSelect different configuration or different node(s).")

    def empty_plot(self, title, text):
        super(DynamicAggregateCIPlot, self).empty_plot(title=title, text=text)

        self.allowed_err_label.css_classes = ["hidden"]
        self.allowed_err_selector.css_classes = ["hidden"]
        self.trial_count_label.css_classes = ["hidden"]
        self.trial_count_selector.css_classes = ["hidden"]
        self.resample_button.css_classes = ["hidden"]

    def style(self):
        super(DynamicAggregateCIPlot, self).style()

        self.allowed_err_label.css_classes = ["increased_vspace"]
        self.trial_count_label.css_classes = ["increased_vspace"]
        self.allowed_err_selector.css_classes = []
        self.trial_count_selector.css_classes = []
        self.resample_button.css_classes = ["increased_vspace", "resample-button"]
