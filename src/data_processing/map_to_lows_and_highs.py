import pandas as pd

from utils.file_handler import read_csv_to_dataframe, save_dataframe_to_csv
from utils.get_env import get_path_from_env

base_path = get_path_from_env('BASE_PATH')

# Load the new CSV file
file_in_path = base_path / 'Synchronized_annotated' / 'summary_classes_synced_merged_selected.csv'
file_out_path = base_path / 'Synchronized_annotated' / 'summary_classes_synced_merged_selected_mapped.csv'
df = read_csv_to_dataframe(file_in_path)

# Extracting annotation column and numerical data
annotations_new = df.iloc[:, 0]
data_new = df.iloc[:, 1:]

# For each column, determine low, medium, and high values
characterization_new = {}

for col in data_new.columns:
    sorted_values = data_new[col].sort_values().tolist()
    #low_values = {sorted_values[0]}  # Single lowest value
    #high_values = {sorted_values[-1]}  # Single highest value
    low_values = set(sorted_values[:3])  # Lowest 3 values
    high_values = set(sorted_values[-3:])  # Highest 3 values

    category = []
    for val in data_new[col]:
        if pd.isna(val):  # Check for NaN
            category.append('none')
        elif val in low_values:
            category.append('low')
        elif val in high_values:
            category.append('high')
        else:
            category.append('medium')

    # Store results in dictionary
    characterization_new[col] = category

# Convert to DataFrame for better visualization
characterization_df = pd.DataFrame(characterization_new)
characterization_df.insert(0, 'annotation', annotations_new)

# Save the characterization results
save_dataframe_to_csv(characterization_df, file_out_path)