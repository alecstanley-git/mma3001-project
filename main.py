import matplotlib.pyplot as plt
import scienceplots

# Figure width in inches, matched to report.typ so figure text renders at the
# same physical point size as the report body text.
#   report text width = 6.2992 in  (A4, typst default margins)
#   plotWidth         = 70%        -> 4.4094 in
# For a figure inside a #columns(2)[...] block use twoColPlotWidth instead:
#   column width = (6.2992 - 0.2520 gutter) / 2 = 3.0236 in
#   twoColPlotWidth = 80%                       -> 2.4189 in
FIG_WIDTH = 4.4094

plt.style.use("science")
plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_WIDTH * 2.625 / 3.5)
