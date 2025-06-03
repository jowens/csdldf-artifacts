import pandas as pd
import altair as alt
import argparse
import os
import re

# Set up argument parser
parser = argparse.ArgumentParser(description='Plot runtime data from CSV files.')
parser.add_argument('--dirname', type=str, required=True,
                    help='Directory containing CSV files (e.g., batchsize-1_CSDLDF_Timed.csv)')
parser.add_argument('--runnumber', type=int,
                    help='Maximum run number to consider (optional).')

args = parser.parse_args()

# Dictionary to store dataframes, with batchsize as key
dfs = {}

# Regex to match the filename pattern and extract the batchsize number
filename_pattern = re.compile(r'batchsize-(\d+)_CSDLDF_Timed\.csv')

# Read CSV files dynamically
try:
    for filename in os.listdir(args.dirname):
        match = filename_pattern.match(filename)
        if match:
            batchsize = int(match.group(1))
            filepath = os.path.join(args.dirname, filename)
            df = pd.read_csv(filepath)

            # Apply runnumber filter if specified
            if args.runnumber is not None and args.runnumber > 0:
                df = df.head(args.runnumber)

            dfs[batchsize] = df
except FileNotFoundError:
    print(f"Error: Directory '{args.dirname}' not found.")
    exit()
except Exception as e:
    print(f"An error occurred while reading files: {e}")
    exit()

# List to hold processed dataframes for concatenation
processed_dfs = []

# Add 'Run Number' and 'Batchsize' columns to each DataFrame
for batchsize, df in dfs.items():
    df['Run Number'] = df.index + 1
    df['Batchsize'] = batchsize
    processed_dfs.append(df)

if not processed_dfs:
    print("No relevant CSV files found in the specified directory.")
    exit()

# Concatenate all DataFrames
df_combined = pd.concat(processed_dfs)

# Plot the data with log scale on y-axis, distinct colors, and partial transparency
column_name = 'time' # Column name for runtime

chart = alt.Chart(df_combined).mark_point(opacity=0.5,filled=True,size=1).encode( # Set opacity here
    x=alt.X('Run Number:Q', title='Run Number'),
    y=alt.Y(f'{column_name}:Q', title='Runtime (ns)', scale=alt.Scale(type="log")), # Set y-axis to log scale
    color=alt.Color('Batchsize:N', legend=alt.Legend(title="Batchsize"), title='Batchsize'), # Rely on default distinct colors
    tooltip=['Run Number', f'{column_name}', 'Batchsize']
).properties(
    title='Pass Runtime vs. Run Number for Different Batchsizes)'
).interactive() # Make the chart interactive for zooming/panning

# Save the chart as a PDF and PNG in the specified directory
basename = args.dirname
if (args.runnumber):
    basename = basename + f'-{args.runnumber}'
output_pdf_path = os.path.join(args.dirname, f'{basename}.pdf')
output_png_path = os.path.join(args.dirname, f'{basename}.png')
try:
    # chart.save(output_pdf_path)
    chart.save(output_png_path, ppi=300)
    print(f"Plot saved to {output_png_path}")
except Exception as e:
    print(f"An error occurred while saving the plots: {e}")