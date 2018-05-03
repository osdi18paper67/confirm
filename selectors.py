import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime as dt

import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure
from bokeh.models.widgets import CheckboxButtonGroup, CheckboxGroup
from bokeh.models.widgets import Div
from bokeh.layouts import widgetbox
from bokeh.models.widgets import RadioButtonGroup, RadioGroup
from bokeh.models.renderers import GlyphRenderer
from bokeh.models import Range1d
from bokeh.colors import RGB
from bokeh.models import Span
from bokeh.models.glyphs import ImageURL
from bokeh.models import DatetimeTickFormatter
from bokeh.models.widgets import Toggle
import time

from functools import partial
import inspect

def to_numeric(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

class MinimalSelector:
    """ Selector for switching between datasets.
    Limitations:
    - single value selector
    - includes only label (specified) and a list of options (specified, fixed)
    - uses a single callback (specified)
    - no post-selection or info label is included
     """

    def __init__(self, label, options, callback=None, default_width=600):
        self.label = label
        self.options = options
        self.callback = callback
        self.default_width = default_width

        self.generated = False
        self.wl = []

    def widgets(self):
        """ Return a list of widgets """

        if not self.generated:
            lw = Div(text=self.label, width=self.default_width)

            sw = RadioButtonGroup(labels=self.options, active=0, width=self.default_width,
                                  css_classes = ["green-buttons"])
            sw.on_change("active", self.callback)

            self.wl = [lw, sw]

        return self.wl


class Selector:
    def __init__(self, criterion,
                 prev_selector = None,
                 multiple_values=False, numeric=False, step_number=None, info=None,
                 selector_callback=None, toggle_callback=None, default_width=700):

        # Unique name of selection criterion matching a column in dataframe/dataset
        self.criterion = criterion

        # Preceeding selector object or None for the first selector in the sequence
        self.prev_selector = prev_selector

        # Allow multiple values being selected or not
        self.multiple_values = multiple_values

        # If true, selector values will be casted to int or float
        self.numeric = numeric

        # If assigned, show step number in the pre label as "Step X:"
        self.step_number = step_number

        # Additional info about selector, will be shown to right from selector options
        self.info = info

        # Callback function for selector change
        self.selector_callback = selector_callback

        # If not None, include button for selecting/unselecting all (and use the passed function as a callback)
        self.toggle_callback = toggle_callback

        self.default_width = default_width

        # The most up-to-date widgets
        self.wl = []

        # Flag indicating that all widgets have been generated
        self.generated = False

        # For convenience, give widgets specific names (in addition to having a list of them, self.wl)
        self.w_info = None
        self.w_pre = None
        self.w_post = None
        self.w_selector = None
        self.w_toggle = None
        self.w_spinner = None

        # Always disable selectors by default (enable and selecte when appropriate actions occur)
        self.disable()

    # Convenience methods for checking state
    def is_selected(self):
        return self.state == "selected"

    def is_enabled(self):
        return self.state == "enabled"

    def is_disabled(self):
        return self.state == "disabled"

    def disable(self):

        #print "\t\tdisable(), begin: criterion: %s" % (self.criterion)

        # key value that defines selector's behavior: disabled, enabled, or selected
        self.state = "disabled"

        # Single value or list of values; set only if state == selected
        self.value = None

        # Index or list of index values corresponding to value
        if self.multiple_values:
            self.value_idx = []
        else:
            self.value_idx = None

        # Available options (set only if state is enabled or selected)
        self.options = []

        # Dataframe before selection (set to non-empty dataframe if state != disabled)
        # Should be obtained from the post_df dataframe available from the previous selector (or complete dataframe for the first selector)
        self.pre_df = pd.DataFrame()

        # Dataframe after selection (set to non-empty dataframe if state == selected)
        self.post_df = pd.DataFrame()

        if self.generated:
            # Update widgets
            self.w_selector.labels = [] if self.multiple_values else None
            self.w_selector.active = [] if self.multiple_values else None
            self.w_post.text = ""

            # Update widget styles -- hide all
            #for w in self.wl:
            #    w.css_classes = ["hidden"]

        #print "\t\tdisable(), end: criterion: %s, state: %s, curr_value_idx: %s" % (self.criterion, self.state, str(self.value_idx))

    def enable(self, pre_df):
        #print "\t\tenable(), begin: criterion: %s, state: %s, curr_value_idx: %s" % (self.criterion, self.state, str(self.value_idx))

        self.state = "enabled"
        self.pre_df = pre_df

        if self.numeric:
            self.options = [str(x) for x in sorted(self.pre_df[self.criterion].unique())]
        else:
            self.options = sorted(self.pre_df[self.criterion].unique())

        # Index or list of index values corresponding to value
        if self.multiple_values:
            self.value_idx = []
        else:
            # FIXME
            self.value_idx = None

        if self.generated:
            # Update styling first
            self.w_info.css_classes = ["infolabel"]
            self.w_pre.css_classes = ["prelabel"]
            self.w_post.css_classes = ["postlabel"]
            self.w_selector.css_classes = ["green-buttons"]

            if self.toggle_callback:
                self.w_toggle.css_classes = []
            else:
                self.w_toggle.css_classes = ["hidden"]

            # Update widgets
            self.w_selector.labels = self.options

            # FIXME -- TRY without explicit active update
            self.w_selector.active = self.value_idx

            self.w_post.text = ""

        #print "\t\tenable(), end: criterion: %s, state: %s, curr_value_idx: %s" % (self.criterion, self.state, str(self.value_idx))

    def select(self, new_idx):
        #print "\t\tselect(), begin: criterion: %s, state: %s, curr_value_idx: %s, options: %s" % (self.criterion, self.state, str(self.value_idx), str(self.options))

        self.state = "selected"
        self.value_idx = new_idx

        # Handle both single values and lists being selected; also handle empty lists
        if isinstance(self.value_idx, list):
            if self.value_idx:
                #print "\t\tselect(): processing non-empty list:", self.value_idx

                if self.numeric:
                    self.value = [to_numeric(self.options[vidx]) for vidx in self.value_idx]
                else:
                    self.value = [self.options[vidx] for vidx in self.value_idx]
                self.post_df = self.pre_df[self.pre_df[self.criterion].isin(self.value)]
            else:
                #print "\t\tselect() ----> enable(), empty list is selected"

                self.value = []
                self.post_df = pd.DataFrame()

                # FIXME
                self.enable(self.pre_df)
                return
        else:
            ## Replace None with 0 to solve errors -- not needed
            # print "\t\tselect(): replacing value_idx of None with []"
            # if self.value_idx == None:
            #     if self.multiple_values:
            #         self.value_idx = []
            #     else:
            #         self.value_idx = None

            if self.numeric:
                self.value = to_numeric(self.options[self.value_idx])
            else:
                self.value = self.options[self.value_idx]
                #print "\t\tselect(): Updated value to:", self.value

            self.post_df = self.pre_df[self.pre_df[self.criterion] == self.value]
            #print "\t\tselect(): lenght of post_df with new value:", len(self.post_df)

        if self.generated:
            # Update styling first
            self.w_info.css_classes = ["infolabel"]
            self.w_pre.css_classes = ["prelabel"]
            self.w_post.css_classes = ["postlabel"]
            self.w_selector.css_classes = ["green-buttons"]

            if self.toggle_callback:
                self.w_toggle.css_classes = []
            else:
                self.w_toggle.css_classes = ["hidden"]

            self.w_selector.labels = self.options

            # FIXME -- TRY without explicit active update
            self.w_selector.active = self.value_idx

            # Since return is added above, after self.enable(), no need to check if state is indeed selected here
            self.w_post.text = "&#10004; Selected measurements: %s" % (format(len(self.post_df), ','))

        #print "\t\tselect(), end: criterion: %s, state: %s, curr_value_idx: %s, options: %s" % (self.criterion, self.state, str(self.value_idx), str(self.options))

    def update(self):
        """ Called when sequence is updated; updates current selector based on the state of the previous one """

        #print "\t\tupdate(): staring for:", self.criterion

        if self.prev_selector.is_disabled():
            #print "\t\tupdate(): previous selector is disabled. About to disable selector for:", self.criterion
            self.disable()
        elif self.prev_selector.is_enabled():
            #print "\t\tupdate(): previous selector is enabled. About to disable selector for:", self.criterion
            self.disable()

        elif self.prev_selector.is_selected():

            #print "\t\tupdate(): previous selector is selected. About to enable selector for:", self.criterion

            # Preserve current selection
            old_labels = self.w_selector.labels
            old_active = self.w_selector.active
            try_reselecting = self.is_selected()

            self.enable(pre_df=self.prev_selector.post_df)

            if len(self.options) == 1:
                #print "\t\tRun select() right after enable()"
                if self.multiple_values:
                    self.select([0])
                else:
                    self.select(0)
            elif try_reselecting:
                #print "\t\tupdate(): trying to reselect previously selected option"
                if self.w_selector.labels == old_labels:
                    self.select(old_active)

            #self.w_spinner.css_classes = ["hidden"]
        else:
            print "ERROR: invalid state found in prev_selector:", self.prev_selector.state

        # Return updated state
        return self.state

    def widgets(self):
        """ Return widgets; if called for the first time, create new widgets"""

        if not self.generated:
            self.create_widgets()

        return row(column(self.w_pre, self.w_selector, self.w_post, self.w_spinner, width=self.default_width),
                   column(self.w_info, self.w_toggle, width=1200-self.default_width),
                   css_classes=["hidden"] if self.is_disabled() else [])

        # if self.state != "disabled":
        #     return row(column(self.w_pre, self.w_selector, self.w_post, width=self.default_width),
        #               column(self.w_info, self.w_toggle, width=1000-self.default_width))
        # else:
        #     return row(column(self.w_pre, self.w_selector, self.w_post, width=self.default_width),
        #               column(self.w_info, self.w_toggle, width=1000-self.default_width),
        #               css_classes=["hidden"])

    def create_widgets(self):
        """ Create new widgets """

        spinner = Div(text="", css_classes=["hidden"])

        info = Div(text=self.info if self.info else "", css_classes=["infolabel"])

        # Create div with pre-label, provided of generated
        if self.multiple_values:
            pre_str = "Select <i><b>%s</b></i> (one or multiple)" % self.criterion
        else:
            pre_str = "Select <i><b>%s</b></i>" % self.criterion
        if self.step_number:
            pre_str = "Step %d: " % self.step_number + pre_str

        pre = Div(text=pre_str, width=self.default_width, css_classes=["prelabel"])

        toggle = Toggle(label="Select/Unselect All", button_type="success")
        if self.toggle_callback:
            toggle.css_classes = []
        else:
            toggle.css_classes = ["hidden"]

        if self.toggle_callback:
            toggle.on_click(partial(self.toggle_callback, sel_obj=self))

        # Create selector with available options
        if self.multiple_values:
            selector = CheckboxButtonGroup(labels=self.options, active=self.value_idx, width=self.default_width,
                                           css_classes = ["green-buttons"])
            #selector = CheckboxGroup(labels=self.options, active=self.value_idx, width=self.default_width)
        else:
            selector = RadioButtonGroup(labels=self.options, active=self.value_idx, width=self.default_width,
                                        css_classes = ["green-buttons"])
            #selector = RadioGroup(labels=self.options, active=self.value_idx, width=self.default_width)
        if self.selector_callback:
            selector.on_change("active", self.selector_callback)

        # Create div with post-label
        post = Div(text="", width=self.default_width, css_classes=["postlabel"])
        #post = Div(text="", width=self.default_width)


        self.w_info = info
        self.w_pre = pre
        self.w_post = post
        self.w_selector = selector
        self.w_toggle = toggle
        self.w_spinner = spinner

        self.wl = [info, pre, toggle, selector, post, spinner]
        self.generated = True
