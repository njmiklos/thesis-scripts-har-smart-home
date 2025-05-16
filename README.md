# 🏡📊 MA Thesis Project: Human Activity Recognition with Smart Home Data
## 📝 Description
This repository contains the code I am developing for my MA thesis on Human Activity Recognition (HAR) using smart home data.
## 🚧 Status: Work in Progress (WIP)
Better organization of files and descriptions are coming soon!
## 📝 Project Structure
```
📂 thesis-scripts-har-smart-home/
├── 📁 src/               # Python scripts. Their purpose is explained in the docstring at the top of every file.
|   ├── 📁 data_acqusition/
|   ├── 📁 data_analysis/
|   ├── 📁 data_processing/
|   ├── 📁 inference/
│   └── 📁 utils/
├── 📁 inputs/            # Inputs to be processed or helping in processing (.gitignored)
├── 📁 outputs/           # Placed for processed files (.gitignored)
├── 📄 .env               # Environment variables (.gitignored)
├── 📄 LICENSE            # Terms of use
├── 📄 README.md          # Project documentation (you are here! Hi!)
└── 📄 requirements.txt   # List of dependencies
```
## 📂 What's Inside `src` Directories
This repository covers working with timeseries sensor data in preparation for a model training, including:
- 📡 Data Acquisition: Communicating with a local InfluxDB instance to collect and save sensor data to .csv files, parsing a file listing annotated acitvity episodes.
- 📊 Data Analysis: Generating statistical summaries based on annotated episodes, creating scatter plots, timeseries graphs, and histograms from sensor readings, correlating features, combining plots into grids.
- 🧹 Data Processing: Transforming, resampling, denoising, segmenting, synchronizing, and merging timeseries sensor data, annotating sensor data.
- 🤖 Inference: Predicting annotations for sensor data with deep learning and foundation models, and evaluating their quality against true annotations.
- 🗃️ Utils: Generic functions for handling files, logging, setting up the working evironment for the project.
# 🛠️ Usage Instructions
## 1. Download this Repository
- Follow the [GitHub documentation](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github)
## 2. Download `runner.py` to Run Edge Impulse Models
If you want to use Edge Impulse models locally:
- Download the [file `runner.py` from Edge Impulse's GitHub repository `linux-sdk-python`](https://github.com/edgeimpulse/linux-sdk-python/blob/master/edge_impulse_linux/runner.py).
- Move the file to the directory `src/inference`.
- Rename the file to `edge_impulse_runner.py`. Your `src/inference/` directory should look like this:
```
.
├── 📄 classify_with_ei_model.py
├── 📄 edge_impulse_runner.py       # Renamed `runner.py`
├── 📁 evaluate
│   └── ...
├── 📄 __init__.py
├── 📄 process_windows_with_fm.py
└── 📄 query_fm_api.py
```
This allows the scripts to interact with local Edge Impulse models.
## 3. Install Dependencies
There are a couple of libraries used in this project. You'll need them to run the scripts. 
1. Make sure you have Python installed with `python --version` or `python3 --version` for higher versions. If no specific version is shown, e.g., `Python 3.10.12`, you need to install Python.
2. a. Optional, recommended: Create a virtual environment.  
If you're not familiar with virtual environments, here's a quick guide: A virtual environment is like a container for your Python project. It keeps the packages you install (typically with `pip`) isolated from the rest of your system (or more precisely: your system's global Python environment). This helps prevent conflicts (you might need different versions of the same package across projects) and avoids clutter (when you delete the environment, all its packages are removed too).
- Choose where to store your environments. I like having a dedicated `venvs` directory in my home folder, which you can also create: `mkdir ~/venvs`.
- Create a virtual environment in the directory called `shproject` (you can choose a different name, it is only meant to mean something to you because you will be using it): `python3 -m venv ~/venvs/shproject`. To work in it, you need to activate it: `source ~/venvs/shproject/bin/activate`. You can see if the environment is active if its name is in the parentheses in front of you username in the terminal, e.g., `(shproject) user@hostname:~$`. When you are done working in the environment, deactivate it with `deactivate` (`(shproject)` will vanish). 
2. b. Activate the virtual environment.
3. Optional, recommended: Upgrade pip with `pip install --upgrade pip`.
4. Navigate to the project's root directory: `cd thesis-scripts-har-smart-home`.
5. Install all required packages: `pip install -r requirements.txt`.
6. Optional, recommended: Verify the installation by listing all packages within the virtual environment: `pip list`. The list should match the content of `requirements.txt`.
## 4. Create an `.env` File
To run the scripts, you must create a `.env` file with environment variables for paths and database credentials. The `.env` file is not included in this repository to protect privacy and security.
### `.env` File Example
```
# Directory Paths
BASE_PATH=/your-path-to-repository/thesis-scripts-har-smart-home/src/
INPUTS_PATH=/your-path-to-repository/thesis-scripts-har-smart-home/inputs
ANNOTATIONS_FILE_PATH=/your-path-to-repository/thesis-scripts-har-smart-home/inputs/file-with-annotations.csv
OUTPUTS_PATH=/your-path-to-repository/thesis-scripts-har-smart-home/outputs/
MODEL_PATH=/your-path-to-repository/thesis-scripts-har-smart-home/inputs/model-file-name.eim

# Database Connection
HOST='localhost'
PORT=8086   # typically
DATABASE_NAME='name_of_your_database'

# Communication with the Chat Academic Cloud API
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
- `MODEL_PATH`: Path to an Edge Impulse model file (`*.eim`).
- My database engine is InfluxDB (version 1.x). The API requires `HOST`, `PORT`, and `DATABASE_NAME`.
- I used the [Chat Academic Cloud API](https://docs.hpc.gwdg.de/services/saia/index.html) when working with Foundation Models. I needed an API key `CHAT_AC_API_KEY` and an endpoint address `CHAT_AC_ENDPOINT`.
## 5. Run Scripts
>⚠️ **Note:** If you work with a virtual environment, make sure that it is active.
1. Write your own `main.py` and import necessary scripts or adjust an existing main code accordingly. You can add or remove variables in the *.env file and use them instead of fixed paths in the scripts.
2. Set your working directory to `thesis-scripts-har-smart-home/src`
- Open your terminal in `thesis-scripts-har-smart-home/src` or run `cd /path-to/thesis-scripts-har-smart-home/src`.
3. Invoke the script as a module
Run: `python3 -m subdirectory.module-name`. 
- Omit the `.py` extension.
- Mind `.` instead of `/` between the directory and module name.
- For top-level scripts (placed directly in `src/`), there is no subdirectory, so simply run: `python3 -m module-name`.
- If you want to run a script that uses an Edge Impulse model, you need to make the model file executable first: `chmod +x '/your-path/to-model/model-file-name-os-architecture-version.eim'`.  
_Example_: If I wanted to run `src/inference/classify_with_ei_model.py`, I would run: `python3 -m inference.classify_with_ei_model`.
# 🚀 Example Pipeline
1. Data Collection:
    - Collect Sensor Data: Install sensors in a household and collect data over time. Record the timestamps and descriptions of activities of interest.
    - Create an Annotation File: List all annotated activity episodes and validate the format using `data_acquisition/validate_annotation_file.py` to ensure compatibility of the file with other scripts.
    - Download Sensor Data: Retrieve data from your InfluxDB instance and export it to `.csv` files using `data_acquisition/query_db.py`.
2. Follow the section "Usage Instructions".
3. Initial Data Exploration: Use scripts from the `data_analysis` directory to inspect your annotated data. Generate summaries and plots to check for sensor errors (e.g., malfunctions or dropouts), duplicates, missing or infinite values. Address issues early before further processing.
4. Data Cleaning: 
    - Resample Data: Standardize sampling rates using `data_processing/resample.py`, especially useful if sensors missed intervals due to connectivity issues.
    - Denoise Data: Remove outliers and faulty sensor readings using `data_processing/denoise.py`.
    - Synchronize Data: Align start and end times of all sensor recordings using `data_processing/synchronize.py`.
    - Annotate Data: Add annotations as a new column using `data_processing/annotate.py`.
5. Data Exploration: Re-analyze your data post-cleaning to ensure no new issues have been introduced. Use scripts in `data_analysis` to visualize or summarize.
6. Data Correlation:Discover relationships between sensor readings with `data_analysis/correlate_features.py`, and visualize them using tools in `data_analysis/visualize`.
7. Data Filtering: Reduce redundancy by removing irrelevant or highly similar features. Use `data_processing/filter.py` to retain only meaningful data for training.
8. Data Segmentation & Balancing with `data_processing/segment.py`:
    - Segment Data: Split time series into meaningful segments (e.g., per day or per activity).
    - Balance Classes: Adjust for imbalanced distributions (e.g., too many 'sleeping', not enough 'cooking' samples).
9. Data Representation
    - Add hand-crafted time-based features using `data_processing/add_time_features.py`.
    - Transform your data completely by generating summary representations using `data_processing/compress.py`.
10. Test Deep Learning or Foundation Models: Use scripts in `inference` to:
    - Run inference with a local deep learning model or a foundation model via API.
    - Measure processing time and memory usage.
    - Save predictions for evaluation.
11. Evaluate Results: Compare model predictions with true annotations using scripts in `data_analysis/visualize.py`. Generate visualizations to assess model quality.
# 💡 Notes & Philosophy
This code is structured based on my thesis needs (e.g., sensor types, locations, sampling rates), so it may not be plug-and-play for others. Full project details will be in my thesis.
# 🛡️ License
The repository is licensed under the MIT License. In short, this means:
- I retain ownership of the code, but you can use it freely under the MIT terms, and mine.  
- This code is nothing you can't find online, just better documented and less optimized. Feel free to reuse it, modify it, or even train your AI on it.  
- If you find it helpful and use it in your work, I'd appreciate a shoutout. I like positive ✨ attention ✨
# ❤️ Acknowledgments
- Technical University of Chemnitz – Professorship of Media Informatics
- The computing time granted by the Resource Allocation Board and provided on the supercomputer Emmy/Grete at NHR-Nord@Göttingen as part of the NHR infrastructure. My thesis used their hosted FM models with [ChatAI](https://docs.hpc.gwdg.de/services/chat-ai/index.html) and [API](https://docs.hpc.gwdg.de/services/saia/index.html).