"""
This script merges synchronized data files and generates a heatmap showing the correlation between 
numerical features.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTPUTS_PATH` in your `.env` file to specify input data directory and heatmap output location.
- Input files must be CSVs with a shared time index to support correct merging across features.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
from pathlib import Path

from utils.file_handler import check_if_output_directory_exists
from utils.get_env import get_path_from_env

from data_processing.merge import merge_files_on_time
from data_analysis.visualize.utils import generate_heatmap


def correlate_features_and_plot(input_dir: Path, output_dir: Path, heatmap_title: str) -> None:
    """
    Merges time-aligned CSV files and generates a heatmap of feature correlations.

    Args:
        input_dir (Path): Directory containing input CSV files to be merged on time.
        output_dir (Path): Directory where the heatmap image will be saved.
        heatmap_title (str): Title used for the heatmap and the saved image filename.

    Returns:
        None
    """
    df = merge_files_on_time(input_dir, skip_unannotated=False)

    if df is not None and not df.empty:
        success = generate_heatmap(df, output_dir, heatmap_title)
        if success:
            print(f'Saved a heatmap to {output_dir}.')
        else:
            print(f'WARNING: Could not save a heatmap.')
    else:
        print('WARNING: DataFrame empty, skipping.')


if __name__ == '__main__':
    heatmap_title = 'Correlation Heatmap of All Features Synchronized to 1s'

    input_dir = get_path_from_env('INPUTS_PATH')
    output_dir = get_path_from_env('OUTPUTS_PATH')
    check_if_output_directory_exists(output_dir)