# Databricks notebook source
# 
# This file contains graph visualization features used for the 
#   2022 Machine Learning Workshop (part of the Software Symposium)!
#   https://FORWARD_SITE/mlworkshop2022 
#      OR https://INFO_SITE/cdo/events/internal-events/4354c5db-3d3d-4481-97c4-8ad8f12686f1
#
# You can (and should) change them for your own experiments, but they are uniquely defined
# here for constants that we will use together.


# COMMAND ----------

# MAGIC %md
# MAGIC # Extra Credit: Dynamic Visualizations 
# MAGIC Maybe you need to generate a data flow or graph representation in code (from data)?  
# MAGIC Aside from the classic list of visualizations (check [here for an expansive example set](https://www.python-graph-gallery.com/)), 
# MAGIC you may need simple graphical illustrations and don't want to spend time manually casting to a good
# MAGIC interface like [mermaid](https://mermaid.live) or something graphics-based 
# MAGIC like [PowerPoint's Smart Art](https://support.microsoft.com/en-us/office/create-a-smartart-graphic-from-scratch-fac94c93-500b-4a0a-97af-124040594842).
# MAGIC 
# MAGIC * Below is a very simple example that is executable in Databricks
# MAGIC * At time of writing you need to include the library `networkx` and `decorator==5.0.9` (there's [a bug with other versions](https://stackoverflow.com/questions/66920533/networkx-shows-random-state-index-is-incorrect))
# MAGIC 
# MAGIC Check out [this walkthrough](https://towardsdatascience.com/python-interactive-network-visualization-using-networkx-plotly-and-dash-e44749161ed7) for more examples.

# COMMAND ----------

# https://stackoverflow.com/questions/66920533/networkx-shows-random-state-index-is-incorrect
from matplotlib import pyplot as plt
import networkx as nx
g1 = nx.DiGraph()
g1.add_edges_from([("root", "NYC Taxi Data"), ("NYC Taxi Data", "Data Filtering"), 
                   ("Zip Codes", "Data Filtering"), ("TLC Data", "Data Filtering"), 
                   ("Data Filtering", "Customer Insights"), 
                   ("Demographics", "Customer Insights"), ("Account Info", "Customer Insights")
])
plt.tight_layout()
nx.draw_networkx(g1) # , arrows=True)
plt.show()
# plt.savefig("g1.png", format="PNG")
# tell matplotlib you're done with the plot: https://stackoverflow.com/questions/741877/how-do-i-tell-matplotlib-that-i-am-done-with-a-plot
plt.clf()
