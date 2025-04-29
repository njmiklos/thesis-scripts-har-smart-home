# 🏡📊 MA Thesis Project: Human Activity Recognition with Smart Home Data
## 📝 Description
This repository contains the code I am developing for my MA thesis on Human Activity Recognition (HAR) using smart home data.
## 🚧 Status: Work in Progress (WIP)
Better organization of files and descriptions are coming soon!
## 📂 What's Inside
This repository covers working with timeseries sensor data in preparation for a model training, including:
- 📡 Data Acquisition: Communicating with a local InfluxDB instance to collect and save sensor data to .csv files.
- 🧹 Data Processing: Transforming, resampling, denoising, synchronizing, and merging timeseries sensor data.
- 📊 Data Analysis & Visualization: Generating statistical summaries based on annotated episodes, creating scatter plots, timeseries graphs, and histograms from sensor readings.
- 🏷️ Data Annotation: Annotating sensor data using predefined episodes.
- 📁 File Management: Splitting timeseries data into daily or episodic segments.
# 📝 Project Structure
```
📂 thesis-scripts-har-smart-home/
├── 📁 src/               # Python scripts
│   └── ...      
├── 📁 inputs/            # Inputs to be processed or helping in processing (.gitignored)
├── 📁 outputs/           # Placed for processed files (.gitignored)
├── 📄 .env               # Environment variables (.gitignored)
├── 📄 LICENSE            # Terms of use
├── 📄 README.md          # Project documentation (you are here! Hi!)
└── 📄 requirements.txt   # List of dependencies
```
# 🛠️ Usage Instructions
## 1. Install Dependencies
There are a couple of libraries I have used. You will need them too in order to run the scripts.
1. Optional, recommended: Create and activate a virtual environment of your choice.
2. Optional, recommended: Upgrade pip with `pip install --upgrade pip`.
3. Navigate to the project's root directory: `cd thesis-scripts-har-smart-home`.
4. Install all required packages: `pip install -r requirements.txt`.
5. Optional, recommended: Verify the installation with `pip list`.
## 2. Create an .env File
To run the scripts, you must create a .env file with environment variables for paths and database credentials. The .env file is not included in this repository to protect privacy and security.
### .env File Example
```
# Directory Paths
BASE_PATH=/module_path/
DATA_PATH=/path_to_data/
ANNOTATIONS_FILE_PATH=/path_to/annotations.csv
OUTPUT_PATH=/output_path/
LOGGING_PATH=/path_to/logs/

# Database Connection
HOST='localhost'
PORT=8086   # typically
DATABASE_NAME='name_of_your_database'

# Coommunication with the Chat Academic Cloud API
CHAT_AC_API_KEY='yourKey'
CHAT_AC_ENDPOINT='https://chat-ai.academiccloud.de/yourEndpoint'
```
### Explanation of Variables in the .env File
- `BASE_PATH`: Directory containing all Python files for this project.
- `DATA_PATH`: Directory containing database files (e.g., .csv files with annotations or sensor data).
- `ANNOTATIONS_FILE_PATH`: Path to a .csv file with annotated episodes. Every episode is a row. It must include the following columns:
    - `start`: Episode start time (e.g., 1733353200000 – UTC milliseconds, Europe/Berlin timezone).
    - `end`: Episode end time (see the format above).
    - `annotation`: Activity class (e.g., 'sleeping', 'airing').
- `OUTPUT_PATH`: Directory for storing all output files.
- `LOGGING_PATH`: Directory for structured logs.
- My database engine is InfluxDB (version 1.x). The API requires `HOST`, `PORT`, and `DATABASE_NAME`.
- I used the [Chat Academic Cloud API](https://docs.hpc.gwdg.de/services/saia/index.html) when working with Foundation Models. I needed an API key `CHAT_AC_API_KEY` and an endpoint address `CHAT_AC_ENDPOINT`.
## 2. Run Scripts
Once you have an .env file:
1. Set your working directory to `thesis-scripts-har-smart-home/src`
Open your terminal in `thesis-scripts-har-smart-home/src` or run `cd /path/to/thesis-scripts-har-smart-home/src`.
2. Invoke the script as a module
Run: `python3 -m {subdirectory}.{module}`. 
- Omit the `.py` extension.
- For top-level scripts (placed directly in `src/`), there is no subdirectory, so simply run: `python3 -m {module}`.

Example 1: If I wanted to run `src/get_env.py`, I would run: `python3 -m get_env`. 
Example 2: If I wanted to run `src/ei_tools/classify_eim.py`, I would run: `python3 -m ei_tools.classify_eim`.
# 🚀 Example Pipeline (WIP, TODO)
1. Data Collection:
    - Create an Annotation File: List all annotated episodes and parse them using `parse_annotation_file.py` to ensure compatibility with other scripts.
    - Download Sensor Data: Retrieve data from your InfluxDB instance and export it to .csv files using scripts in the `databank_communication` directory.
2. Initial Data Exploration: Inspect your annotated data using `explore_data_pandas.py`, `visualize_data.py`, `summarize_classes.py`. Look for errors (e.g., sensor malfunctions), duplicates (e.g., multiple logs of the same event), missing or infinite values.
3. Data Cleaning: 
    - Resample Data: Standardize sampling rates with `resample_df.py`. Sensors may have missed intervals (e.g., due to connection drops).
    - Denoise Data: Remove outliers from faulty sensor readings using `denoise_data.py`.
    - Synchronize Data: Align start and end times of all sensor recordings using `synchronize_data.py`.
4. Data Exploration: Re-examine your data to ensure cleaning hasn't introduced errors with `explore_data_pandas.py`, `visualize_data.py`, `summarize_classes.py`. If you see that the fridge opened at 3 AM, question your assumptions about the culprit. Is this a data issue or a sneaky partner? (Probably the data!)
5. Data Correlation: Identify relationships between sensor readings to uncover patterns. Use: `visualize_data.py`.
6. Data Filtering: Reduce redundancy by removing highly similar or irrelevant data to retain only meaningful features for model training. Use `filter_df.py`.
7. Data Segmentation & Balancing with `separate_into_episodes.py`
    - Segment Data: Split timeseries into episodes, e.g., daily or activity-based segments.
    - Balance Classes: Adjust for imbalanced activity distributions (e.g., too many ‘sleeping' episodes, not enough ‘cooking').
# 💡 Notes & Philosophy
- This code is structured based on my thesis needs (e.g., sensor types, locations, sampling rates), so it may not be plug-and-play for others. Full project details will be in my thesis.
# 🛡️ License
The repository is licensed under the MIT License. In short, this means:
- I retain ownership of the code, but you can use it freely under the MIT terms, and mine.  
- This code is nothing you can't find online, just better documented and less optimized. Feel free to reuse it, modify it, or even train your AI on it.  
- If you find it helpful and use it in your work, I'd appreciate a shoutout. I like positive ✨ attention ✨
# ❤️ Acknowledgments
- Technical University of Chemnitz – Professorship of Media Informatics
- The computing time granted by the Resource Allocation Board and provided on the supercomputer Emmy/Grete at NHR-Nord@Göttingen as part of the NHR infrastructure. My thesis used their hosted FM models with [ChatAI](https://docs.hpc.gwdg.de/services/chat-ai/index.html) and [API](https://docs.hpc.gwdg.de/services/saia/index.html).