import polars as pl


def preprocess_csv(file_path):
    # Load CSV skipping metadata (first 15 rows contain metadata and headers)
    df = pl.read_csv(file_path, separator=";", skip_rows=15, has_header=False)

    # Manually set headers from the first row (indicator row)
    headers = df[0].to_list()
    df = df[1:]  # Drop the first row after assigning it as headers
    df.columns = headers

    # Drop redundant rows (like "unit", "member" rows, if applicable)
    # You can drop them if you know their specific contents
    df = df.filter(~df.select_at_idx(0).is_in(["unit", "member"]))

    return df


# Example usage
file_path = "D:/ML2025/PCPP/data/cosmo-e.csv"
clean_df = preprocess_csv(file_path)

print("Clean Dataset Preview:")
print(clean_df)
