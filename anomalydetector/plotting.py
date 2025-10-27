import matplotlib.lines as mlines
import corner
import numpy as np

def make_corner_plot(id_dist, ood_dist, ranges, titles, n_bins=20):
    figure = corner.corner(
        id_dist, 
        range  = ranges, 
        titles = titles,
        title_fmt = None,
        hist_kwargs={"color": 'tab:green', "alpha": 0.3, "fill": True, },
        color = "tab:green",
        weights = np.ones(id_dist.shape[0]) / id_dist.shape[0],
        show_titles = True,
        plot_contours = True,
        plot_datapoints = False,
        fill_contours=True,
        plot_density=False,
        density = True,
        hist2d_kwargs = {"no_fill_contours": True, "plot_datapoints": False},
        hist1d_kwargs = {"density": True},
        bins = n_bins,
        quiet = True,
    )

    corner.corner(
        ood_dist, 
        range  = ranges,
        titles = titles,
        title_fmt = None,
        hist_kwargs={"color": 'tab:orange', "alpha": 0.3, "fill": True, },
        color="tab:orange",
        weights = np.ones(ood_dist.shape[0]) / ood_dist.shape[0],
        show_titles = True,
        plot_contours = True,
        fill_contours = True,
        plot_datapoints = False,
        plot_density=False,
        density=True,
        hist2d_kwargs = {"no_fill_contours": True, "plot_datapoints": False, "new_fig": False},
        hist1d_kwargs = {"density": True},
        fig = figure,
        bins = n_bins, 
        quiet = True,
    )

    id_line = mlines.Line2D([], [], color="tab:green", label="In Distribution (ID)")
    ood_line = mlines.Line2D([], [], color="tab:orange", label='Out of Distribution (OOD)')

    figure.legend(handles=[id_line, ood_line], bbox_to_anchor=(0., 0.8, 0.9, .0), loc=4, fontsize=24)

    return figure