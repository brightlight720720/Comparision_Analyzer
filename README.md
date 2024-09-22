# Multi-format Analysis App

## Overview
The Multi-format Analysis App is a Streamlit-based web application that allows users to upload CSV, Excel, or JSON files and perform various statistical analyses, including C-Index computation, Integrated Discrimination Improvement (IDI), and Net Reclassification Improvement (NRI) calculations.

## Features
1. File upload support for CSV, Excel, and JSON formats
2. Data preview and summary statistics
3. C-Index computation and comparison between old and new models
4. IDI and NRI calculations
5. Visualizations for C-Index comparison
6. Export functionality for analysis results and plots

## Pages
1. **Home**: Upload file, preview data, split dataset, and select columns
2. **C-Index**: Compute and compare C-index for old and new models
3. **IDI/NRI**: Calculate Integrated Discrimination Improvement and Net Reclassification Improvement

## Installation
1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run main.py
   ```

## Usage
1. Navigate to the Home page and upload your CSV, Excel, or JSON file
2. Select the required columns for analysis
3. Go to the C-Index page to compute and compare C-index for old and new models
4. Visit the IDI/NRI page to calculate IDI and NRI metrics

## Author
YuHsuan

## Contact
Email: brightlight720@gmail.com

## License
This project is open-source and available under the MIT License.
