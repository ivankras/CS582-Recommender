
def missing_counts_by_cols(data):
    missing_val_count_by_column = data.isnull().sum()
    return missing_val_count_by_column[missing_val_count_by_column > 0]


def missing_values_cols(data):
    return [col for col in data.columns if data[col].isnull().any()]
