
# Extract the annotation column (class labels)
class_labels = df["annotation"].unique()

# Define function to count high, low, and none values per class
def count_value_types_per_class(df, class_column, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    results = []
    
    for label in class_labels:
        subset = df[df[class_column] == label]
        
        high_count = (subset.applymap(lambda x: str(x).strip().lower() == 'high')).sum().sum()
        low_count = (subset.applymap(lambda x: str(x).strip().lower() == 'low')).sum().sum()
        none_count = (subset.applymap(lambda x: str(x).strip().lower() == 'none')).sum().sum()
        
        results.append({
            "class": label,
            "high_count": high_count,
            "low_count": low_count,
            "none_count": none_count
        })
    
    return pd.DataFrame(results)

# Compute counts for each class
class_counts_df = count_value_types_per_class(df, "annotation")

# Define function to get the measurements corresponding to high, low, and none values per class
def get_measurements_per_class(df, class_column):
    measurement_results = {}

    for label in class_labels:
        subset = df[df[class_column] == label]

        high_measurements = subset.applymap(lambda x: str(x).strip().lower() == 'high')
        low_measurements = subset.applymap(lambda x: str(x).strip().lower() == 'low')
        none_measurements = subset.applymap(lambda x: str(x).strip().lower() == 'none')

        high_cols = high_measurements.any().index[high_measurements.any().values]
        low_cols = low_measurements.any().index[low_measurements.any().values]
        none_cols = none_measurements.any().index[none_measurements.any().values]

        measurement_results[label] = {
            "high_measurements": list(high_cols),
            "low_measurements": list(low_cols),
            "none_measurements": list(none_cols)
        }

    return measurement_results

# Compute measurements for each class
measurements_per_class = get_measurements_per_class(df, "annotation")

# Convert to DataFrame for better visualization
measurements_df = pd.DataFrame.from_dict(measurements_per_class, orient='index')

# Display the results
tools.display_dataframe_to_user(name="Measurements per Class", dataframe=measurements_df)
