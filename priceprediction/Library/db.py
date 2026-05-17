import os
from hdbcli import dbapi
import pandas as pd


def create_hana_connection():
    user = os.environ.get("HANA_USER")
    pwd = os.environ.get("HANA_PASSWORD")

    if not user or not pwd:
        raise ValueError("HANA_USER or HANA_PASSWORD not set")

    conn = dbapi.connect(
        address="8a27452f-1c34-4351-8048-b533c3e6ea5e.hana.prod-us10.hanacloud.ondemand.com",
        port=443,
        user=user,
        password=pwd
    )

    print("Connected to SAP HANA successfully")
    return conn


def load_table(conn, table_name, schema_name):
    query = f'SELECT * FROM "{schema_name}"."{table_name}"'
    df = pd.read_sql(query, conn)

    print(f"Data loaded with {len(df)} rows and {len(df.columns)} columns")
    return df
