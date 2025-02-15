import pandas as pd

from get_env import get_base_path
from handle_csv import read_csv_to_pandas_dataframe

# Load the dataset
base_path = get_base_path()
file_path = base_path /'summary_classes_raw_mapped.csv'
df = read_csv_to_pandas_dataframe(file_path)

# Identify relevant columns to retain:
# 1. Keep 'annotation'
# 2. Keep motion magnitude columns ('motion_magnitude')
# 3. Keep hygrometer, thermometer, and luxmeter data for d1, d2, and d3
relevant_columns = ["annotation"]
relevant_columns += [col for col in df.columns if "motion_magnitude" in col]
relevant_columns += [col for col in df.columns if any(sensor in col for sensor in ["hygrometer", "thermometer", "luxmeter", "airQuality", "co2", "sound"]) and any(device in col for device in ["d1", "d2", "d3", "d4", "d5"])]

# Filter dataset to include only relevant columns
df_filtered = df[relevant_columns].copy()

# Replace "medium" values with NaN instead of dropping entire rows
df_filtered.replace("medium", pd.NA, inplace=True)

# Generate the summary again
summary = {}
for col in df_filtered.columns:
    if col != "annotation":
        filtered_group = df_filtered.dropna(subset=[col])  # Remove NaN values but keep other data
        if not filtered_group.empty:
            summary[col] = (
                filtered_group.groupby(col)["annotation"]
                .unique()
                .apply(lambda x: ", ".join(x))
                .to_dict()
            )

# Convert to structured summary format
structured_summary = []
for sensor, values in summary.items():
    for level, activities in values.items():
        structured_summary.append(f"- {sensor} {level} {activities}")

# Save the structured summary to a text file
summary_file_path = base_path / 'summary_classes_raw_mapping_list.txt'

with open(summary_file_path, 'w') as file:
    file.write('\n'.join(structured_summary))

# Output the file path for download
print(f'Summary saved successfully: {summary_file_path}')
