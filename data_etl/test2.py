import pandas as pd

# fpath = "D:/special/coding/git/bigdata/parquet_data/temperature_org/temperature_20250116.parquet"
# fpath = "D:/special/coding/git/bigdata/parquet_data/cattle_org/cattle_20250116.parquet"
# csvpath = "D:/special/coding/git/bigdata/parquet_data/cattle_org.csv"
fpath = "D:/special/coding/git/bigdata/parquet_data/cattle/cattle_20250116.parquet"
csvpath = "D:/special/coding/git/bigdata/parquet_data/cattle.csv"
df = pd.read_parquet(fpath)
df.to_csv(csvpath, index=False, encoding='utf-8-sig')
# print(len(df))
# print(df['LKNTS_NM'].value_counts())
# print(df['OCCRRNC_DE'].value_counts())
