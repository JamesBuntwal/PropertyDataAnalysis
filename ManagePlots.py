"""ManagePlots.py

This script houses convenience functions for some of the plots in the analysis notebook.

"""

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from datetime import timedelta

#-------------------------------------------------------------------------------
# Constants and Config
#-------------------------------------------------------------------------------
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'

formatter = ScalarFormatter()
formatter.set_scientific(False)

#-------------------------------------------------------------------------------
# Sales Per Week
#-------------------------------------------------------------------------------
def sales_plots(ax, sales_per_week):
    ax.plot(sales_per_week['week_start'], sales_per_week['count'], label = "Number of Sales per Week")
    ax.set_xlabel('Week Commencing Date')
    ax.set_ylabel('Number of Sales')
    return ax

def merge_legends(*axlist):
    lines, labels = list(zip(*[(x, x.get_label()) for ax in axlist for x in ax.get_lines() if not ax.get_label().startswith('_')]))
    return lines, labels

def plot_sales_per_week(sales_per_week):
    fig, ax = plt.subplots(figsize = (16,9))
    ax = sales_plots(ax, sales_per_week)
    ax.legend(loc = 'best')
    plt.show()

def plot_sales_per_week_with_last_weeks(    sales_per_week,
                                            last_weeks,
                                            last_b2_weeks):
    fig, ax = plt.subplots(figsize = (16,9))
    ax = sales_plots(ax, sales_per_week)
    ax.scatter( last_b2_weeks['week_start'],
                last_b2_weeks['count'],
                s=150,
                marker = 'o',
                color = 'k',
                label = "Last-but-two Week of the Year"
                )
    ax.scatter( last_weeks['week_start'],
                last_weeks['count'],
                s=150,
                marker = 'x',
                color = 'k',
                label = "Last Week of the Year"
                )
    ax.legend(loc='best')
    plt.show()


def plot_sales_per_week_with_boe_cpih(sales_per_week, BoE, CPIH):
    fig, ax = plt.subplots(figsize = (16,9))
    ax = sales_plots(ax, sales_per_week)
    ax2 = ax.twinx()
    ax2.step(BoE['Date'], BoE['Rate'], where = 'post', label = "Bank of England Base Rate", color = 'r')
    ax2.plot(CPIH['Date'], CPIH['Rate'], label = "CPIH", color = 'g')
    ax.legend(*merge_legends(ax, ax2), loc = 'best')
    plt.show()


politcal_events = [
    # date, label, height, fill_end
    ('2020-01-31', 'UK Left EU', 80000, None),
    ('2020-03-23', "First UK National Lockdown", 77000, '2020-06-23'),
    ('2020-06-01', 'Stamp Duty Holiday £500,000', 74000, '2021-06-30'),
    ('2020-11-05', 'Second UK National Lockdown', 71000, '2020-12-02'),
    ('2021-01-06', 'Third UK National Lockdown', 68000, '2021-03-08'),
    ('2021-07-01', 'Stamp Duty Holiday £250,000', 65000, '2021-09-30'),
    ('2022-02-04', 'Russia Invades Ukraine', 80000, None),
    ('2022-09-06', 'Boris Out - Truss In', 75000, None),
    ('2022-09-08', 'Death of Queen Elizabeth II', 70000, None),
    ('2022-09-23', 'Truss Minibudget', 65000, None),
    ('2022-10-25', 'Truss Out - Sunak In', 60000, None)
]

def plot_sales_per_week_with_politics(sales_per_week):
    fig, ax = plt.subplots(figsize = (16,9))
    ax = sales_plots(ax, sales_per_week)

    for event_date, event_label, h, event_end in politcal_events:
        event_date = pd.to_datetime(event_date)
        ax.vlines(event_date, 0, h, color = 'k')
        ax.annotate(event_label, (event_date+timedelta(days = 5), h-2000), )
        if event_end is not None:
            event_end = pd.to_datetime(event_end)
            ax.fill_between((event_date, event_end),
                            0,
                            h,
                            alpha = 0.2)
    ax.legend(loc = 'best')
    plt.show()


def plot_sales_per_week_summary(sales_per_week, last_weeks, last_b2_weeks):
    spikes = sales_per_week.loc[sales_per_week.week_start.isin(pd.to_datetime(['2021-06-28', '2021-09-27']))]

    fig, ax = plt.subplots(figsize = (16,9))
    ax = sales_plots(ax, sales_per_week)

    ax.plot([pd.to_datetime('2020-01-01'), pd.to_datetime('2020-05-01')], [25000, 5000], label = "Downtrend due to Covid and/or Brexit")
    ax.plot([pd.to_datetime('2020-04-01'), pd.to_datetime('2021-04-08')], [5000, 35000], label = "Uptrend due to stamp duty holiday")

    ax.scatter( last_b2_weeks['week_start'],
                last_b2_weeks['count'],
                s=150,
                marker = 'o',
                color = 'k',
                label = "Pre-holiday period spikes"
                )
    ax.scatter( last_weeks['week_start'],
                last_weeks['count'],
                s=150,
                marker = 'x',
                color = 'k',
                label = "Holiday period troughs"
                )

    ax.scatter( spikes['week_start'],
                spikes['count'],
                s = 150,
                color = 'r',
                label = "Stamp duty holiday end dates spikes")

    ax.plot(    [   pd.to_datetime('2021-02-01'),
                    pd.to_datetime('2024-07-01')],
                [32000, 8000],
                label = "Downtrend due to Cost of Living Crisis")
    
    ax.legend(loc='best')
    plt.show()


#-------------------------------------------------------------------------------
# Histogram of Sales
#-------------------------------------------------------------------------------

def plot_hist_sales_prices(empirical_data, N_bins = 250):

    # Create figure and axis to add plots to
    fig, ax = plt.subplots(figsize = (16,9))

    # plot the histogram
    ax.hist(    empirical_data,
                bins = N_bins,
                density = False)
    
    # reformat axis ticks to not use scientific notation
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Label the axis and title
    ax.set_xlabel("Price (£)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Sale Prices")
    plt.show()

def plot_hist_curve_fit(hist_data, dist):

    # calculate centers and widths of the bin
    bin_centers = hist_data.index.to_numpy()
    bin_width = (bin_centers[-1] - bin_centers[0]) / len(bin_centers)

    # create figure and axis to add plots to
    fig, (ax_left, ax_right) = plt.subplots(1,2, figsize = (12,6), sharex = True)

    # histogram side
    ax_left.bar(    x = bin_centers,
                    height = hist_data['count'],
                    width = bin_width,
                    color = BLUE
                    )
    ax_left.plot(   bin_centers,
                    hist_data['fit_data'],
                    color = ORANGE
                    )
    # reformat axis to not use scientific notation
    ax_left.xaxis.set_major_formatter(formatter)
    ax_left.yaxis.set_major_formatter(formatter)
    # add labels and title
    ax_left.set_xlabel("Price (£)")
    ax_left.set_ylabel("Frequency")
    ax_left.set_title('Histogram and Curve Fit')

    # CDF side
    ax_right.plot(  bin_centers,
                    hist_data['ecdf'],
                    color = BLUE,
                    label = "Empirical Data"
                    )
    ax_right.plot(  bin_centers,
                    hist_data['cdf'],
                    color = ORANGE,
                    label = "Fitted Data"
                    )
    # add labels and title
    ax_right.set_xlabel("Price (£)")
    ax_right.set_ylabel("Probability")
    ax_right.set_title('Cumulative Probability Distributions')

    # add legend
    fig.legend(*merge_legends(ax_right), ncols = 2, loc = 'upper center', title = dist.name)
    plt.show()


#-------------------------------------------------------------------------------
# Heat Maps
#-------------------------------------------------------------------------------

def plot_heat_maps(agg_data):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))

    agg_data.plot(
        column      = 'NumberOfSales',
        ax          = ax_left,
        legend      = True,
        legend_kwds = { 'label': "Number of Sales",
                        'orientation': "horizontal"},
        cmap='OrRd'
        )

    agg_data.plot(
        column      = 'AverageSalePrice',
        ax          = ax_right,
        legend      = True,
        legend_kwds = { 'label': "Average Sale Price (£)",
                        'orientation': "horizontal"},
        cmap='OrRd'
        )

    # Set plot title and labels
    ax_left.set_title('6.) Heat Map of Number of Sales per 10Km Grid Square')
    ax_right.set_title('7.) Heat Map of Average Sale Price per 10Km Grid Square')
    ax_left.xaxis.set_major_formatter(formatter)
    ax_left.yaxis.set_major_formatter(formatter)
    ax_right.xaxis.set_major_formatter(formatter)
    ax_right.yaxis.set_major_formatter(formatter)

    # Show the plot
    plt.show()






