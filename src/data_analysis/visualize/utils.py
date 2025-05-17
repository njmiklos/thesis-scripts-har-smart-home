"""
This script provides utilities to generate and save visualizations from time series and tabular data. It includes 
functions for creating histograms, scatter plots, time series plots, heatmaps, and confusion matrices, as well as 
tools for combining images into a grid.

Environment Configuration:
- Set `INPUTS_PATH` and `OUTUTS_PATH` in your `.env` file to specify input and output directories.
- Timestamps are assumed to be in milliseconds and are converted to localized datetime objects.
- Refer to `README.md` for full setup, usage instructions, and formatting requirements.
"""
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
import math

from pathlib import Path
from typing import List

from data_processing.convert_timestamps import convert_timestamps_from_miliseconds_to_localized_datetime_srs
from utils.get_env import get_path_from_env


def create_safe_title(title: str) -> str:
    """
    Create a safe filename from a title by removing all characters 
    except letters (a-z, A-Z) and whitespace, replacing spaces with underscores,
    and changing to lower case.

    Args:
        title (str): The original title string to sanitize.

    Returns:
        str: A filename-safe version of the title.
    """
    cleaned = re.sub(r'[^a-zA-Z\s]', '', title)
    safe_title = cleaned.replace(' ', '_')
    return safe_title.lower()

def generate_histogram(data_srs: pd.Series, title: str, frequency_axis_label: str, 
                  value_axis_label: str, output_dir_path: Path, color='green',  bins: int = 15) -> bool:
    """
    Generates a histogram to visualize the distribution of values in a specified column.

    Args:
        data_srs (pd.Series): The column in the DataFrame with the values to visualize in the histogram.
        title (str): The title of the histogram, e.g., 'Value distribution of temperature'.
        value_axis_label (str): Label for the x-axis, e.g., 'Temperature [°C]'.
        frequency_axis_label (str): Label for the y-axis, representing the frequency of occurrences, e.g., 'Frequency'.
        output_dir_path (Path): The directory path where the image will be saved.
        color (str, optional): The color of the plotted line, e.g., 'blue', '#ff5733'.
        bins (int, optional): Number of bins/intervals to divide data into. Default is 15.

    Returns:
        bool: True if plot was saved sucessfully, False otherwise.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data_srs, bins=bins, color=color, edgecolor='black', alpha=0.7)

    title = title.replace('_', ' ')
    title = title.title()
    plt.title(title)

    value_axis_label = value_axis_label.title()
    plt.xlabel(value_axis_label)

    frequency_axis_label = frequency_axis_label.title()
    plt.ylabel(frequency_axis_label)

    plt.grid(True)

    #plt.show()
    raw_title = f'{title}_histogram'
    safe_filename = create_safe_title(raw_title) + '.png'
    output_path = output_dir_path / safe_filename
    try:
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        plt.close()
        print(e)
        return False

def generate_comparative_scatterplots(time_srs1: pd.Series, time_srs2: pd.Series, data_srs1: pd.Series, data_srs2: pd.Series, 
                               modification: str, plot_title: str, time_axis_label: str, value_axis_label: str, 
                               output_dir_path: Path, srs1_color='orange', srs2_color='green',
                               srs1_marker='o', srs2_marker='x', alpha=0.6) -> bool:
    """
    Generates and saves a scatter plot comparing the same data type in two columns from two different DataFrames.

    Args:
        time_srs1, time_srs2 (pd.Series): The columns in the DataFrames with the time to visualize in the histogram.
        data_srs1, data_srs2 (pd.DataFrame): Series containing data, old and modified, respectively.
        plot_title (str): The title of the plot.
        time_axis_label (str): Label for the x-axis (time), e.g., 'Time [s]'.
        value_axis_label (str): Label for the y-axis, e.g., 'Acceleration Magnitude'.
        output_dir_path (Path): The directory path where the plot image will be saved.
        srs1_color (str, optional): Color of the original data points.
        srs2_color (str, optional): Color of the new data points.
        srs1_marker (str, optional): Marker style for original data.
        srs2_marker (str, optional): Marker style for new data..
        alpha (float, optional): Transparency level for the scatter points.

    Returns:
        bool: True if plot was saved sucessfully, False otherwise.
    """
    time_srs1 = convert_timestamps_from_miliseconds_to_localized_datetime_srs(time_srs1)
    time_srs2 = convert_timestamps_from_miliseconds_to_localized_datetime_srs(time_srs2)

    # Create scatter plot
    modification = modification.title()
    plt.figure(figsize=(12, 6))
    plt.scatter(time_srs1, data_srs1, label='Original Data', color=srs1_color, marker=srs1_marker, alpha=alpha)
    plt.scatter(time_srs2, data_srs2, label=f'{modification} Data', color=srs2_color, marker=srs2_marker, alpha=alpha)

    # Add labels and titles
    time_axis_label = time_axis_label.title()
    plt.xlabel(time_axis_label)

    value_axis_label = value_axis_label.title()
    plt.ylabel(value_axis_label)

    plot_title = plot_title.replace('_', ' ')
    plot_title = plot_title.title()
    plt.title(plot_title)
    
    # Additional settings
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    raw_title = f'{plot_title}_comparative_scatter_plot'
    safe_filename = create_safe_title(raw_title) + '.png'
    output_path = output_dir_path / safe_filename
    try:
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        plt.close()
        print(e)
        return False
    
def generate_scatter_plot(time_srs: pd.Series, data_srs: pd.Series, plot_title: str, 
                     time_axis_label: str, value_axis_label: str, output_dir_path: Path, 
                     color='green', marker='o', alpha=0.6) -> bool:
    """
    Generates and saves a scatter plot of a value column from a DataFrame.

    Args:
        time_srs (pd.Series): The column in the DataFrame with the time to visualize in the histogram.
        data_srs (pd.Series): The column in the DataFrame with the values to visualize in the histogram.
        plot_title (str): The title of the plot.
        time_axis_label (str): Label for the x-axis (time), e.g., 'Time [s]'.
        value_axis_label (str): Label for the y-axis, e.g., 'Acceleration Magnitude'.
        output_dir_path (Path): The directory path where the plot image will be saved.
        color (str, optional): Color of the data points.
        marker (str, optional): Marker style for data.
        alpha (float, optional): Transparency level for the scatter points.

    Returns:
        bool: True if plot was saved sucessfully, False otherwise.
    """
    time_srs = convert_timestamps_from_miliseconds_to_localized_datetime_srs(time_srs)
    value_axis_label = value_axis_label.title()

    # Create scatter plot
    plt.figure(figsize=(12, 6))
    plt.scatter(time_srs, data_srs, label=value_axis_label, color=color, marker=marker, alpha=alpha)

    # Add labels and titles
    time_axis_label = time_axis_label.title()
    plt.xlabel(time_axis_label)

    plt.ylabel(value_axis_label)

    plot_title = plot_title.replace('_', ' ')
    plot_title = plot_title.title()
    plt.title(plot_title)
    
    # Additional settings
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    raw_title = f'{plot_title}_scatter_plot'
    safe_filename = create_safe_title(raw_title) + '.png'
    output_path = output_dir_path / safe_filename
    try:
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False
    
def generate_comparative_timeseries_plot(time_srs: pd.Series, data_srs1: pd.Series, data_srs2: pd.Series, 
                                         data_srs1_label: str, data_srs2_label: str, 
                                         plot_title: str, time_axis_label: str, value_axis_label: str, 
                                         output_dir_path: Path, srs1_color='orange', srs2_color='green',
                                         linestyle_srs1='--', linestyle_srs2='-.',
                                         linewidth=2.0, alpha=0.8, marker='o', markersize=5) -> bool:
    """
    Generates and saves a time series line plot comparing two data series.

    Args:
        time_srs (pd.Series): The column in the DataFrame with the time to visualize in the plot.
        data_srs1, data_srs2 (pd.Series): Series containing data.
        data_srs1_label, data_srs2_label (str): The labels for the data, e.g., 'Humidity'. 
        plot_title (str): The title of the plot.
        time_axis_label (str): Label for the x-axis (time), e.g., 'Time [s]'.
        value_axis_label (str): Label for the y-axis, e.g., 'Humidity [%]'.
        output_dir_path (Path): The directory path where the plot image will be saved.
        srs1_color (str, optional): Color of the original data line.
        srs2_color (str, optional): Color of the modified data line.
        linestyle_srs1 (str, optional): Line style for the data_srs1 (e.g., '-', '--', '-.', ':').
        linestyle_srs2 (str, optional): Line style for the data_srs2 (e.g., '-', '--', '-.', ':').
        linewidth (float, optional): Line width for the plot.
        alpha (float, optional): Transparency level for the line.
        marker (str, optional): Marker style for data points.
        markersize (float, optional): Size of the markers.

    Returns:
        bool: True if plot was saved successfully, False otherwise.
    """
    # Create time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_srs, data_srs1, label=data_srs1_label, color=srs1_color, linestyle=linestyle_srs1, 
             linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)
    plt.plot(time_srs, data_srs2, label=data_srs2_label, color=srs2_color, linestyle=linestyle_srs2, 
             linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)

    # Add labels and title
    plt.xlabel(time_axis_label.title())
    plt.ylabel(value_axis_label.title())
    plt.title(plot_title.replace('_', ' ').title())

    # Additional settings
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    raw_title = f'{plot_title}_comparative_time_series_plot'
    safe_filename = create_safe_title(raw_title) + '.png'
    output_path = output_dir_path / safe_filename
    try:
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        plt.close()
        print(e)
        return False

def generate_timeseries_plot(time_srs: pd.Series, data_srs: pd.Series, plot_title: str, 
                         time_axis_label: str, value_axis_label: str, output_dir_path: Path, 
                         color='green', linestyle='-', linewidth=2.0, alpha=0.8, 
                         marker='o', markersize=5, avg_line=True, avg_line_color='orange', 
                         avg_line_style='--', avg_line_width=2.0) -> bool:
    """
    Generates and saves a time series line plot with markers (dots) at each data point and an optional average line.

    Args:
        time_srs (pd.Series): The column in the DataFrame with the time to visualize in the plot.
        data_srs (pd.Series): The column in the DataFrame with the values to visualize.
        plot_title (str): The title of the plot.
        time_axis_label (str): Label for the x-axis (time), e.g., 'Time [s]'.
        value_axis_label (str): Label for the y-axis, e.g., 'Acceleration Magnitude'.
        output_dir_path (Path): The directory path where the plot image will be saved.
        color (str, optional): Color of the line plot.
        linestyle (str, optional): Line style for the plot (e.g., '-', '--', '-.', ':').
        linewidth (float, optional): Line width for the plot.
        alpha (float, optional): Transparency level for the line.
        marker (str, optional): Marker style for the dots (e.g., 'o', 's', 'D', 'x').
        markersize (int, optional): Size of the markers (dots).
        avg_line (bool, optional): Whether to plot a horizontal line for the average value.
        avg_line_color (str, optional): Color of the average line.
        avg_line_style (str, optional): Style of the average line (e.g., '--', '-.', ':').
        avg_line_width (float, optional): Width of the average line.

    Returns:
        bool: True if plot was saved successfully, False otherwise.
    """
    time_srs = convert_timestamps_from_miliseconds_to_localized_datetime_srs(time_srs)
    value_axis_label = value_axis_label.title()

    # Compute average value
    avg_value = data_srs.mean()

    # Create line plot with dots
    plt.figure(figsize=(12, 6))
    plt.plot(time_srs, data_srs, label=value_axis_label, color=color, linestyle=linestyle, 
             linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)

    # Add an average line
    if avg_line:
        plt.axhline(y=avg_value, color=avg_line_color, linestyle=avg_line_style, 
                    linewidth=avg_line_width, label=f'Avg: {avg_value:.2f}')

    # Add labels and titles
    time_axis_label = time_axis_label.title()
    plt.xlabel(time_axis_label)
    plt.ylabel(value_axis_label)

    plot_title = plot_title.replace('_', ' ').title()
    plt.title(plot_title)
    
    # Additional settings
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    raw_title = f'{plot_title}_time_series_plot'
    safe_filename = create_safe_title(raw_title) + '.png'
    output_path = output_dir_path / safe_filename
    try:
        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        return True
    except Exception as e:
        plt.close()
        return False

def generate_heatmap(df: pd.DataFrame, output_dir_path: Path, title: str = 'Feature Correlation Heatmap') -> bool:
    """
    Generates and saves a heatmap to visualize the correlation between numerical columns, excluding specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        title (str): The title of the heatmap.
        output_dir_path (Path): The directory path where the heatmap image will be saved.

    Returns:
        bool: True if plot was saved successfully, False otherwise.
    """
    # Drop non-numeric or unwanted columns
    cols_to_ignore = ['time', 'annotation']
    df_filtered = df.drop(columns=[col for col in cols_to_ignore if col in df.columns])

    # Calculate the correlation matrix
    corr_matrix = df_filtered.corr()

    # Generate the heatmap
    plt.figure(figsize=(20, 16))
    plt.title(title.title(), fontsize=20, pad=20)

    cax = plt.matshow(corr_matrix, cmap='coolwarm')
    plt.colorbar(cax)

    # Add labels for x and y axes
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns, fontsize=8)

    # Add gridlines
    plt.gca().xaxis.tick_bottom()  # Move x-axis labels to bottom
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)

    # Prepare the output path
    safe_filename = create_safe_title(title) + '.png'
    output_path = output_dir_path / safe_filename

    # Save the plot
    try:
        plt.savefig(output_path, format='png', dpi=400, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f'Failed to save heatmap: {e}')
        plt.close()
        return False
    
def generate_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_dir_path: Path) -> bool:
    """
    Generate and save a confusion matrix as a PNG image with a white background.

    Args:
        matrix (np.ndarray): Confusion matrix in percentage form.
        class_names (list[str]): List of class labels.
        output_dir_path (Path): Path to save the confusion matrix.

    Returns:
        bool: True if successfully saved, False otherwise.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot Confusion Matrix
    cax = ax.matshow(matrix, cmap='Greens', vmin=0, vmax=100)

    # Add color bar
    plt.colorbar(cax)

    # Add labels for x and y axes
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Format values
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.1f}%', ha='center', va='center', color='black', fontsize=8)

    plt.xlabel('Predicted Labels', fontsize=16)
    plt.gca().xaxis.tick_bottom()  # Move x-axis labels to bottom

    plt.ylabel('True Labels', fontsize=16)

    title = 'Confusion Matrix in Percentage'
    plt.title(title, fontsize=20)

    # Prepare the output path
    output_file_name = 'confusion_matrix_percentage.png'
    output_path = output_dir_path / output_file_name

    try:
        plt.savefig(output_path, format='png', dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved confusion matrix to {output_path}')
        return True
    except Exception as e:
        plt.close()
        print(f'Error saving confusion matrix: {e}')
        return False

def resize_image(img: Image.Image, scale: float = 0.25) -> Image.Image:
    """
    Resizes an image according to the given scale.

    Args:
        Image.Image: Input image object to be risized.   
        scale (float): Factor by which to scale down the images.

    Returns:
        Image.Image: Resized image object.    
    """
    return img.resize((int(img.width * scale), int(img.height * scale)))

def load_and_resize_images(image_paths: List[Path], scale: float = 0.25) -> List[Image.Image]:
    """
    Loads and resizes a list of image files.

    Args:
        image_paths (List[Path]): List of image file paths to load.
        scale (float): Factor by which to scale down the images.

    Returns:
        List[Image.Image]: List of resized image objects.
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        img = resize_image(img, scale)
        images.append(img)
    return images

def arrange_images_into_grid(images: List[Image.Image], no_columns: int = 5) -> Image.Image:
    """
    Arranges images into a grid.

    Args:
        images (List[Image.Image]): List of images to arrange.
        no_columns (int): Number of columns in the grid.

    Returns:
        Image.Image: Combined grid image.
    """
    rows = math.ceil(len(images) / no_columns)
    width, height = images[0].size
    grid_img = Image.new('RGB', (no_columns * width, rows * height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % no_columns) * width
        y = (idx // no_columns) * height
        grid_img.paste(img, (x, y))

    return grid_img

def combine_images_into_grid(input_dir: Path, output_dir: Path, output_filename: str,
                             no_columns: int, scale: float) -> None:
    """
    Processes all image files in the input directory by resizing them and
    combining them into a grid.

    Args:
        input_dir (Path): Directory containing input image files.
        output_dir (Path): Directory to write the output combined image.
        output_filename (str): Name for the output file of the grid.
        no_columns (int): Number of columns in the grid.
        scale (float): Factor by which to scale down the images to save memory.
    """
    supported_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in supported_extensions])[:19]

    if not image_paths:
        raise FileNotFoundError('No image files found in input directory.')

    images = load_and_resize_images(image_paths, scale)
    grid = arrange_images_into_grid(images, no_columns)

    output_file = output_dir / output_filename
    grid.save(output_file)
    print(f'Saved combined image grid to: {output_file}')


if __name__ == '__main__':
    output_filename = 'combined_grid.jpg'
    no_columns = 5
    scale = 0.25

    output_dir = get_path_from_env('OUTUTS_PATH')
    input_dir = get_path_from_env('INPUTS_PATH')
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory {input_dir} does not exist.')

    combine_images_into_grid(input_dir, output_dir, output_filename, no_columns, scale)