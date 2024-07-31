"""ManageData.py

This script is used to take the external datasets and create pyspark dataframes for use in the analysis notebook. It also houses some data manipulation related convenience functions.


"""

#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
import pyspark as ps
from pyspark.sql import functions as psf
import pandas as pd
import geopandas as gpd

#-------------------------------------------------------------------------------
# Paid for Price Data
#-------------------------------------------------------------------------------
def get_PPD(spark):
    """
    Column names and descriptions available in PPD_columns_descriptions.xlsx
    """
    output = (
        spark
        .read
        .csv(   "Data/pp-complete.csv",
                header=False,
                inferSchema=True
                )
        .toDF(  "transaction_id",
                "price",
                "date_of_transfer",
                "postcode",
                "property_type",
                "old_new",
                "duration",
                "PAON",
                "SAON",
                "street",
                "locality",
                "town_city",
                "district",
                "county",
                "PPD_category_type",
                "record_status_monthly_file_only"
                )
    )
    return output

#-------------------------------------------------------------------------------
# National Statistics Postcode lookup
#-------------------------------------------------------------------------------
def get_NSPL(spark):
    """
    Load in the NSPL data from CSV.
    """
    file_path = 'Data/NSPL_2021_MAY_2024/Data/NSPL21_MAY_2024_UK.csv'
    output = (
        spark
        .read
        .csv(   file_path,
                header = True,
                inferSchema = True
                )
        )
    return output

#-------------------------------------------------------------------------------
# Region Names and Codes
#-------------------------------------------------------------------------------
def get_region_names_codes(spark):
    """
    Load in the Region names and codes data from CSV.
    """
    file_path = 'Data/NSPL_2021_MAY_2024/Documents/Region names and codes EN as at 12_20 (RGN).csv'
    output = (
        spark
        .read
        .csv(   file_path,
                header = True,
                inferSchema = True
                )
        )
    return output

#-------------------------------------------------------------------------------
# Postcode-Region Lookup Table
#-------------------------------------------------------------------------------
def get_postcode_region_lookup(spark):
    """
    This table will be used to join region names to postcodes.
    The column 'pcds' had a better match rate on the PPD dataset than 'pcd'.
    The final lookup table is formed from left joining the region names and codes
    data on to the NSPL data via postcode.
    """

    nspl = (
        get_NSPL(           spark
                            )
        .select(            'rgn',
                            'pcds'
                            )
        .withColumnRenamed( 'rgn',
                            'region_code'
                            )
        .withColumnRenamed( 'pcds',
                            'postcode'
                            )
    )

    regions = (
        get_region_names_codes( spark
                                )
        .select(                'RGN20CD',
                                'RGN20NM'
                                )
        .withColumnRenamed(     'RGN20CD',
                                'region_code'
                                )
        .withColumnRenamed(     'RGN20NM',
                                'region'
                                )
    )

    postcode_region_lookup = (
        nspl
        .join(          regions,
                        'region_code',
                        'left'
                        )
    )

    return postcode_region_lookup



#-------------------------------------------------------------------------------
# Historic Bank of England Base Rates
#-------------------------------------------------------------------------------
def get_bank_of_england():
    """
    https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp
    """
    # Read from csv
    BoE = pd.read_csv('Data/Bank Rate history and data  Bank of England Database.csv')

    # Reformat the date column
    BoE['Date'] = pd.to_datetime(BoE['Date Changed'], format='%d %b %y')

    # Filter for 2020 onwards
    BoE = BoE.loc[BoE.Date >= pd.to_datetime('2020-01-01'), :]

    # repeat final value for recent date.
    # this is because the dates are when the rate changed so a final end date
    # is necessary when plotting
    BoE.loc[BoE.index.max()+1, :] = [None, 5.25, '2024-07-01']

    # Sort by date
    BoE.sort_values('Date', inplace = True)

    return BoE

#-------------------------------------------------------------------------------
# Historic Consumer Price Index Rates
#-------------------------------------------------------------------------------
def get_consumer_price_indices():
    """
    https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/l55o/mm23
    """
    # Read from CSV
    CPIH = pd.read_csv('Data/CPIH.csv')

    # Reformat Date field
    CPIH['Date'] = pd.to_datetime(CPIH['Date'], format='%Y %b')

    # Filter for 2020 onwards
    CPIH = CPIH.loc[CPIH.Date >= pd.to_datetime('2020-01-01'), :]

    return CPIH

#-------------------------------------------------------------------------------
# Last-But-N Weeks
#-------------------------------------------------------------------------------
def last_but_n_weeks(sales_per_week, n, recurse = True):
    """
    This function returns the last-but-n dates from the sales_per_week data.
    It requires a single recursion as the initial list of dates contains n observations
    per year and so to isolate just the nth observations the previous are calculated
    and removed.
    """
    date_list = pd.to_datetime(
        sales_per_week
        .loc[sales_per_week.year != 2024, :]
        .groupby('year')
        ['week_start']
        .nlargest(n+1)
        .reset_index(level = 1, drop = True)
        .tolist()
        )
    if recurse:
        rec_list = set(last_but_n_weeks(sales_per_week, n-1, recurse = False))
        date_list = set(date_list) - rec_list
        bools = sales_per_week.week_start.isin(date_list)
        output = sales_per_week.loc[bools, :].copy(deep = True)
        return output
    else:
        return date_list

#-------------------------------------------------------------------------------
# Paid Price with Geo data
#-------------------------------------------------------------------------------

def get_PPD_with_geo(spark, PPD):
    NSPL = (
        get_NSPL(spark)
        .select(            'pcds',
                            'oseast1m',
                            'osnrth1m'
                            )
        .withColumnRenamed( 'pcds',
                            'postcode'
                            )
        .withColumnRenamed( 'oseast1m',
                            'Easting'
                            )
        .withColumnRenamed( 'osnrth1m',
                            'Northing'
                            )
    )
    PPD_with_geo = (
        PPD
        .join(      NSPL,
                    'postcode',
                    'left'
                    )
        .select(    'transaction_id',
                    'date_of_transfer',
                    'Easting',
                    'Northing',
                    'price')
    )
    return PPD_with_geo

#-------------------------------------------------------------------------------
# Paid Price with Geo data
#-------------------------------------------------------------------------------

def PPDgeo_per_year(year, PPD_with_geo, shape_file_10km):
    """
    Filter PPD for a given year and join on the geopandas data
    """
    ppd_pd = (
        PPD_with_geo
        .filter(psf.year(psf.col('date_of_transfer')) == year)
        .toPandas()
    )
    ppd_geo = gpd.GeoDataFrame(
        ppd_pd, 
        geometry = gpd.points_from_xy(ppd_pd.Easting, ppd_pd.Northing),
        crs='EPSG:27700'  # British National Grid
    )

    # Perform spatial join
    output = (
        gpd.sjoin(  ppd_geo,
                    shape_file_10km,
                    how='left'
                    )
        .groupby(   'TILE_NAME'
                    )
        .agg({      "transaction_id" : 'count',
                    'price' : 'mean'}
                    )
        .rename(    columns = { "transaction_id" : "NumberOfSales",
                                "price" : "AverageSalePrice"}
                                )
        .reset_index(drop = False)
    )
    output['weighted_value'] = output['NumberOfSales']*output["AverageSalePrice"]

    return output[["TILE_NAME", "NumberOfSales", "weighted_value"]]
