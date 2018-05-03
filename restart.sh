#!/bin/bash

# Kill running app
kill -9 `ps aux | grep bokeh | head -1 | awk '{print $2}'`

# Start an updated app
/opt/conda/envs/py27/bin/bokeh serve . --show --allow-websocket-origin='*' &
