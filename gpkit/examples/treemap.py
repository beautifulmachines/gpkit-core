"Treemap example"

# import plotly
from gpkit.interactive.plotting import treemap

from .uav import UAV

M = UAV()

fig = treemap(M)
# plotly.offline.plot(fig, filename="treemap.html")  # uncomment to show

fig = treemap(M, itemize="constraints", sizebycount=True)
# plotly.offline.plot(fig, filename="sizedtreemap.html")  # uncomment to show
