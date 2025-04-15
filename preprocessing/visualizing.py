import polars as pl

def show_table(csv_path):
    df = pl.read_csv(csv_path, separator=";", skip_rows=17)


    print("COSMO-E Dataset Preview:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    print("\nFirst 5 rows:")
    print(df.head(5))

    print("\nColumn data types:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")