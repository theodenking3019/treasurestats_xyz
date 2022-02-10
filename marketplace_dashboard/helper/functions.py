import datetime as dt
import json
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# connect to database
def db_connect(func):
    def with_connection_(*args,**kwargs):
        sql_credential = os.path.join("static", "credentials", "mysql_credential.json")
        with open(sql_credential) as f:
            mysql_credentials = json.loads(f.read())
        engine = create_engine(
            "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            user=mysql_credentials['username'], 
            pw=mysql_credentials['pw'], 
            host=mysql_credentials['host'], 
            db=mysql_credentials['database']
            )
        )
        connection = engine.connect()
        try:
            return_value = func(connection, *args,**kwargs)
        except:
            raise Exception("Error connecting to database!")
        finally:
            connection.close()
        return return_value
    return with_connection_

# define functions
def q75(x):
    return np.percentile(x, 75)

def q25(x):
    return np.percentile(x, 25)

@db_connect
def get_sales(connection, collection, lookback_window, display_currency):
    # read in sales data
    min_datetime = dt.datetime.now() - dt.timedelta(days = lookback_window)
    sales_query  = 'SELECT tx_hash, datetime, sale_amt_magic, nft_collection, nft_id, nft_subcategory, quantity FROM treasure.marketplace_sales WHERE datetime >= %(min_datetime)s'
    if collection != 'all':
        sales_query = sales_query + 'AND nft_collection = %(collection)s'
    marketplace_sales_list = []
    marketplace_sales_query = connection.execute(sales_query, {'min_datetime': min_datetime, 'collection': collection})
    for row in marketplace_sales_query:
        marketplace_sales_list.append(row)
    marketplace_sales = pd.DataFrame(marketplace_sales_list)
    if len(marketplace_sales)==0:
        return pd.DataFrame()
    marketplace_sales.columns=list(marketplace_sales_query.keys())
    marketplace_sales['date'] = marketplace_sales['datetime'].dt.date

    # split out multiple-quantity sales into individual rows
    multi_sales = marketplace_sales.loc[marketplace_sales['quantity']>1].copy()
    marketplace_sales = marketplace_sales.loc[marketplace_sales['quantity']==1].copy()
    multi_sales['sale_amt_magic'] = multi_sales['sale_amt_magic'] / multi_sales['quantity']
    multi_sales = multi_sales.loc[multi_sales.index.repeat(multi_sales['quantity'])].reset_index(drop=True)
    multi_sales['quantity'] = 1
    marketplace_sales = pd.concat([marketplace_sales, multi_sales])

    # read in token prices
    if display_currency != 'MAGIC':
        prices_query = 'SELECT * FROM treasure.token_prices WHERE datetime >= %(min_datetime)s'
        token_prices_list = []
        token_prices_query = connection.execute(prices_query, {'min_datetime': min_datetime})
        for row in token_prices_query:
            token_prices_list.append(row)
        token_prices = pd.DataFrame(token_prices_list)
        token_prices.columns=list(token_prices_query.keys())

        token_prices['date'] = token_prices['datetime'].dt.date
        token_prices.rename(columns={'datetime':'token_price_datetime'}, inplace=True)

        marketplace_sales = marketplace_sales.merge(token_prices, how='left', on='date')
        marketplace_sales['token_price_sale_datetime_diff'] = marketplace_sales['datetime'] - marketplace_sales['token_price_datetime']
        most_recent_token_prices = marketplace_sales.groupby('tx_hash',as_index=False).agg({'token_price_sale_datetime_diff':'min'})
        marketplace_sales = marketplace_sales.merge(most_recent_token_prices, how='inner', on=['tx_hash', 'token_price_sale_datetime_diff'])
        marketplace_sales['sale_amt_usd'] = marketplace_sales['sale_amt_magic'] * marketplace_sales['price_magic_usd']
        marketplace_sales['sale_amt_eth'] = (marketplace_sales['sale_amt_magic'] * marketplace_sales['price_magic_usd']) / marketplace_sales['price_eth_usd']

    return marketplace_sales

@db_connect
def get_attribute_values(connection, df, attributes):
    attributes_dfs = {}
    for key, value in attributes.items():
        if (pd.isnull(value[0])):
            continue
        elif (len(value) > 1):
            tmp_attributes_lst = []
            tmp_attributes_query = connection.execute(f'SELECT * FROM treasure.attributes_{key}')
            for row in tmp_attributes_query:
                tmp_attributes_lst.append(row)
            tmp_attributes = pd.DataFrame(tmp_attributes_lst)
            tmp_attributes.columns=list(tmp_attributes_query.keys())
            tmp_attributes = tmp_attributes.loc[:, value + ['id']]
            attributes_dfs[key] = tmp_attributes
        else:
            tmp_attributes = df.loc[df['nft_collection']==key, value + ['nft_id']].drop_duplicates()
            tmp_attributes.rename(columns={'nft_id':'id'}, inplace=True)
            attributes_dfs[key] = tmp_attributes

        return attributes_dfs