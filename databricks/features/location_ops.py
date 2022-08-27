# Databricks notebook source
# MAGIC %run ../utilities/settings

# COMMAND ----------

import geopandas as gpd
import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
import math

# COMMAND ----------

def shape_resolution(gdf_src, resolution_target):
    pass
    
    


#helper to demonstrate the different resolutions
def shape_plot_example(gdf_example=None, res_compare=[7, 9, 11], col_viz=None):
    from h3ronpy import vector, util
    # containment example - https://observablehq.com/@nrabinowitz/h3-hierarchical-non-containment?collection=@nrabinowitz/h3
    if gdf_example is None:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        gdf_example = world[world["continent"] == "Africa"]
        res_compare = [2, 3, 4]
    
    # original source - https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
    figure, axis = plt.subplots(2, 2, figsize=(10, 10))
    gdf_copy = gpd.GeoDataFrame(gdf_example)
    txt_title = ''
    if col_viz is not None:
        gdf_copy.plot(ax=axis[1, 1], column=col_viz, cmap='winter', legend=True)
        txt_title = f", value:{col_viz}"
    else:
        gdf_copy.plot(ax=axis[1, 1])
    axis[1,1].set_title(f'original{txt_title}')
    fn_log(f"[shape_plot_example] Input columns: {gdf_copy.columns}")

    # convert to h3 cells and build a new geodataframe
    for res_idx in range(min(3, len(res_compare))):
        ax_target = axis[res_idx % 2, math.floor(res_idx / 2)]
        try:
            gdf_plot = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(gdf_copy, res_compare[res_idx]))
        except Exception as e: 
            fn_log("[shape_plot_example]: failed to convert entire section to h3 cells, skipping")
            continue
        del gdf_copy['to_h3_idx']
        if col_viz is not None:
            gdf_plot.plot(ax=ax_target, column=col_viz, cmap='winter', legend=True)
        else:
            gdf_plot.plot(ax=ax_target)
        ax_target.set_title(f"resolution: {res_compare[res_idx]}{txt_title}")

        

    
# https://github.com/nmandery/h3ronpy
# consider other shape sources

# plotting different colors, like heatmap -- https://www.kaggle.com/code/imdevskp/geopandas
# shape_plot_example(col_viz='pop_est')


# COMMAND ----------

# helper to demonstrate simple plot of  amap...
def shape_plot_map(gdf_data, col_viz=None):
    from h3ronpy import vector, util

    # original source - https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
    figure, axis = plt.subplots(1, 1, figsize=(14, 8))
    gdf_copy = gpd.GeoDataFrame(gdf_data)
    txt_title = ''
    if col_viz is not None:
        gdf_copy.plot(ax=axis, column=col_viz, cmap='winter', legend=True)
        txt_title = f", value:{col_viz}"
    else:
        gdf_copy.plot(ax=axis)
    axis.set_title(f'original{txt_title}')
    fn_log(f"[shape_plot_example] Input columns: {gdf_copy.columns}")

#     # convert to h3 cells and build a new geodataframe
#     for res_idx in range(min(3, len(res_compare))):
#         ax_target = axis[res_idx % 2, math.floor(res_idx / 2)]
#         try:
#             gdf_plot = util.h3index_column_to_geodataframe(vector.geodataframe_to_h3(gdf_copy, res_compare[res_idx]))
#         except Exception as e: 
#             fn_log("[shape_plot_example]: failed to convert entire section to h3 cells, skipping")
#             continue
#         del gdf_copy['to_h3_idx']
#         if col_viz is not None:
#             gdf_plot.plot(ax=ax_target, column=col_viz, cmap='winter', legend=True)
#         else:
#             gdf_plot.plot(ax=ax_target)
#         ax_target.set_title(f"resolution: {res_compare[res_idx]}{txt_title}")


# COMMAND ----------

# helper to dump to CSV as nn
def shape_export(path_shp, path_csv):
    gdf_new = gpd.read_file(path_shp)
    df_new = pd.DataFrame(gdf_new)
    df_new.to_csv(path_csv)

# COMMAND ----------

def shape_import(path_csv=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/raw/tl_2021_us_zcta520.csv",
                 path_df=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/geometry/ztca",
                 col_zip="ZCTA5CE20", col_geometry='geometry'):
    """Method to load raw zip shapefiles and join against known zip code.  To skip zipcode join, do not specify a zip column (`col_zip`)."""
    import zipcodes
    import re
    import json

    fn_log(f"[zip_import] Reading CSV from '{path_csv}'....")
    df_with_schema = (
        spark.read.format("csv")
        .option("inferSchema" , True)
        .option("header", True)
        .load(path_csv)
        .drop('_c0')
        .withColumn('geometry', F.col(col_geometry).cast('string'))
        # .withColumn('geo_processed', convertUDF(F.col('geometry')))  --- wait until final component
    )
    if col_zip is not None:
        df_with_schema = df_with_schema.withColumnRenamed(col_zip, 'zip')
    fn_log(f"[shape_import] Imported CSV '{path_csv}' with columns {df_with_schema.columns}...")
    
    # rename from MTFCC20 -> mtfcc
    re_sub = re.compile(r"[\d]+$")
    for x in df_with_schema.columns:
        if re_sub.search(x) is not None:
            df_with_schema = df_with_schema.withColumnRenamed(x, re_sub.sub('', x.lower()))
        else:
            df_with_schema = df_with_schema.withColumnRenamed(x, x.lower())
    
    # now join for zip code information...
    def zip_lookup(z):
        if not zipcodes.is_real(z):
            return None
        exact_zip = zipcodes.matching(z)
        if len(exact_zip):
            exact_zip = exact_zip[0]
            return json.dumps(exact_zip)
        return None

    if col_zip is None:
        df_final = df_with_schema
    else:
        # zip binding and keep columns - https://pypi.org/project/zipcodes/
        fn_log(f"[shape_import] Binding to zip locations...")
        zip_expand = F.udf(zip_lookup)
        df_final = (df_with_schema
            .withColumn('lookup', F.from_json(zip_expand(F.col('zip').cast('string')), "MAP<STRING,STRING>"))
            # list_json_cols = ['city', 'country', 'state', 'timezone', 'zip_code_type', 'area_codes']
            .withColumn('city', F.col('lookup.city'))
            .withColumn('country', F.col('lookup.country'))
            .withColumn('timezone', F.col('lookup.timezone'))
            .withColumn('state', F.col('lookup.state'))
            .withColumn('zip_code_type', F.col('lookup.zip_code_type'))
            .withColumn('area_codes', F.col('lookup.area_codes'))
            .withColumn('acceptable_cities', F.col('lookup.acceptable_cities'))
            .withColumn('unacceptable_cities', F.col('lookup.unacceptable_cities'))
            .drop('lookup')
        )
    # write out to final destination
    fn_log(f"[shape_import] Write to final delta location '{path_df}'...")
    df_final.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(path_df)
    df_final = spark.read.format('delta').load(path_df)
    return df_final

# df_zip_shape = zip_import()
# display(df_zip_shape)

if False:  # testing and singleton imports
#     df = shape_import(path_csv=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/raw/dma.csv",
#                  path_df=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/geometry/dma", col_zip=None)
    df = shape_import(path_csv=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/raw/taxi_zones.csv",
                 path_df=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/geometry/nyctaxi_zones", col_zip=None, col_geometry='the_geom')
    display(df)

# COMMAND ----------

import fsspec
import pandas
import geopandas as gpd

# NOTE: this method (using a local mount and 'ffspec') did not work for a direct load, please use 
#       the `shape_export` and `shape_import` functions instead

# method to read shp directly from ADLS (unfortunately, it didn't work in the end)
def shape_import_adls(path_file=f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/tl_2021_us_zcta520.shp", 
                      mode='rb', sas_token=None):
    import re
    # Example shape data location
    # tl_2021_us_zcta520.shp.ea.iso.xml "https://dsairgeneraleastus2sa.blob.core.windows.net/mlworkshop2022/shapefiles

    # extra_configs = {
    #   "fs.azure.account.auth.type": "CustomAccessToken",
    #   "fs.azure.account.custom.token.provider.class": spark.conf.get("spark.databricks.passthrough.adls.gen2.tokenProviderClassName"),
    #   "token_credential": spark.conf.get("spark.databricks.passthrough.adls.gen2.tokenProviderClassName")
    # }
    if sas_token is None:
        sas_token = "sp=rl&st=2022-08-21T21:09:53Z&se=2022-08-23T05:09:53Z&spr=https&sv=2021-06-08&sr=d&sig=TVium1rK%2B6I89hSk6kj%2Fv0%2BMkS14YIFCA9g0OA2iq4E%3D&sdd=1"

    adls_account_name = 'dsairgeneraleastus2sa'

    adls_path = f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/{path_file}"
    fsspec_handle = fsspec.open(adls_path, account_name=adls_account_name, anon=False, account_key=adls_account_key, mode=mode) # account_name = adls_account_name, sas_token=sas_key)

#     with fsspec_handle.open() as f:
#         textf = f.read()
#     print(textf)

    # tl_2021_read(path_file='tl_2021_us_zcta520.shp.iso.xml').read()
    return fsspec_handle

# print(tl_2021_read('tl_2021_us_zcta520.shp.iso.xml', 'rt').open().read())

# https://towardsdatascience.com/uber-h3-for-data-analysis-with-python-1e54acdcc908
# warning, this will read a large amount of memory into place...
# with tl_2021_read(mode='rb').open() as file:
#     united_states = gpd.read_file(file)

# COMMAND ----------

# # attempt to map zip code to a source DF using lat/lon and h3 indexing
# def h3_map_zip(df_src, col_src_lat, col_src_lon, df_zip, h3_resolution=11):
#     # confused about resolution? check out this example - https://observablehq.com/@nrabinowitz/h3-hierarchical-non-containment?collection=@nrabinowitz/h3
    
#     # encode source into h3 resolution
#     # load a zip shape file, filter by intersected h3
#     # 
#     # 
    
#     .withColumn('h3', udf_h3(F.col('intptlat'), F.col('intptlon'), F.lit(8)))
    
    

# COMMAND ----------

# testing to plot different resolutions of zip code ata
if False: 
    import geopandas as gpd
    from shapely import wkt
    df_zip_shape = spark.read.format('delta').load(f"{CREDENTIALS['paths']['databricks_dataset']}/tl_2021_us_zcta520/geometry/ztca")
    df_sub = (df_zip_shape
        .filter((F.col('city') == F.lit('Austin')) & (F.col('state')==F.lit('TX')))
    #     .filter(F.col('state')==F.lit('TX'))
    )
    
    # https://pypi.org/project/h3-pyspark/
    from h3_pyspark.indexing import index_shape 

    from shapely import wkt, geometry
    import json

    # create encoding function from raw zip code geometry to h3
#     pdf_geo_decode = lambda x: json.dumps(geometry.mapping(wkt.loads(x)))

    pdf_sub = (df_sub
#         .withColumn('geometry', udf_geo_decode(F.col('geometry_wkt')))
#         .withColumn(f"h3", index_shape(F.col('geometry_json'), F.lit(resolution_h3)))    
        .toPandas()     
    )

    pdf_sub['geometry'] = pdf_sub['geometry'].apply(lambda x: wkt.loads(x))
    shape_plot_example(pdf_sub, res_compare=[7, 8, 9], col_viz='awater')
    # shape_plot_example(pdf_sub, res_compare=[5,6,7], col_viz='awater')
    # gdf2_us = gpd.GeoDataFrame(pdf_sub)
    # gdf2_us.plot()


    # geo_df['sqft_log'] = np.log(geo_df['sqft_living'])fig, ax = plt.subplots(figsize = (10,10))
    # kings_county_map.to_crs(epsg=4326).plot(ax=ax, color='lightgrey')
    # geo_df.plot(column = 'sqft_log', ax=ax, cmap = 'winter',
    #             legend = True, legend_kwds={'shrink': 0.3},
    #             alpha = .5)
    # ax.set_title('Sqft Heatmap')



# COMMAND ----------

