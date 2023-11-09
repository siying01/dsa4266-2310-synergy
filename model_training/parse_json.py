'''This script takes in an unzipped json file and parses it into a csv file
For more information, refer to the help page: python parse_json.py --help
Example usage: python parse_json.py ../data/dataset0.json train --file_containing_labels ../data/data.info
Example usage: python parse_json.py ../data/dataset1.json test'''

# Import dependencies
import numpy as np
import pandas as pd
import argparse
import os
import json
from pathlib import Path


# Set-up help page for this script
parser = argparse.ArgumentParser(description='Parses an unzipped json file into a csv file.')
parser.add_argument('file_to_parse', type=str, help='Path to the file to be parsed. Can be relative path, relative to the folder containing this script.')
parser.add_argument('use_type', type=str, choices=['train', 'test'], help='Usage type. Either train or test.')
parser.add_argument('--file_containing_labels', type=str, help='When use_type is "train", the path to the file containing the labels is required.')
args = parser.parse_args()


# Access the command-line arguments
file_to_parse = args.file_to_parse
use_type = args.use_type
file_containing_labels = args.file_containing_labels
base_path = Path(__file__).parent
file_path = (base_path / file_to_parse).resolve()


# If the json data is used for training, ensure that we have the file containing labels 
if use_type == 'train' and file_containing_labels is None:
    parser.error('The --file_containing_labels is required when use_type is train.')


# Parse json into csv format while applying bag-level summarisation 
id, pos, seq = [],[],[]
time1_mean, time1_median, time1_std, stddev1_mean, stddev1_median, stddev1_std, current1_mean, current1_median, current1_std, current1_range = [],[],[],[],[],[],[],[],[],[]
time2_mean, time2_median, time2_std, stddev2_mean, stddev2_median, stddev2_std, current2_mean, current2_median, current2_std, current2_range = [],[],[],[],[],[],[],[],[],[]
time3_mean, time3_median, time3_std, stddev3_mean, stddev3_median, stddev3_std, current3_mean, current3_median, current3_std, current3_range = [],[],[],[],[],[],[],[],[],[]

id, pos, seq, time1, stddev1, current1, time2, stddev2, current2, time3, stddev3, current3 = [],[],[],[],[],[],[],[],[],[],[],[]
with open(file_path) as data_file:
    for line in data_file:   # For each json object: eg. {'ENST00000000233': {'244': {'AAGACCA': [[0.00299, 2.06, 125.0, 0.0177, 10.4, 122.0, 0.0093, 10.9, 84.1]]}}}  
        json_data = json.loads(line)
        for k1, v1 in json_data.items():   # k1 = 'ENST00000000233'
            id.append(k1)
            for k2, v2 in v1.items():   # k2 = '244'
                pos.append(int(k2))
                for k3, v3 in v2.items():   # k3 = 'AAGACCA'
                    reads = v3
                    seq.append(k3)
                    f1, f2, f3, f4, f5, f6, f7, f8, f9 = [], [], [], [], [], [], [], [], []
                    numreads = 0
                    for read in reads:   # Each read is a new row
                        numreads += 1
                        f1.append(read[0])
                        f2.append(read[1])
                        f3.append(read[2])
                        f4.append(read[3])
                        f5.append(read[4])
                        f6.append(read[5])
                        f7.append(read[6])
                        f8.append(read[7])
                        f9.append(read[8])
                    # Consolidate read values for each bag
                    time1_mean.append(np.mean(f1))   # Mean of dwelling
                    time1_median.append(np.median(f1))   # Median of dwelling
                    time1_std.append(np.std(f1))   # Standard deviation of dwelling
                    stddev1_mean.append(np.mean(f2))   # Mean of standard deviation
                    stddev1_median.append(np.median(f2))   # Median of standard deviation
                    stddev1_std.append(np.std(f2))   # Standard deviation of standard deviation
                    current1_mean.append(np.mean(f3))   # Mean of current
                    current1_median.append(np.median(f3))   # Median of current
                    current1_std.append(np.std(f3))   # Standard deviation of current
                    current1_range.append(np.max(f3) - np.min(f3))   # Range of current
                    time2_mean.append(np.mean(f4))
                    time2_median.append(np.mean(f4))
                    time2_std.append(np.std(f4))
                    stddev2_mean.append(np.mean(f5))
                    stddev2_median.append(np.median(f5))
                    stddev2_std.append(np.std(f5))
                    current2_mean.append(np.mean(f6))
                    current2_median.append(np.mean(f6))
                    current2_std.append(np.std(f6))
                    current2_range.append(np.max(f6) - np.min(f6))
                    time3_mean.append(np.mean(f7))
                    time3_median.append(np.median(f7))
                    time3_std.append(np.std(f8))
                    stddev3_mean.append(np.mean(f8))
                    stddev3_median.append(np.median(f8))
                    stddev3_std.append(np.std(f8))
                    current3_mean.append(np.mean(f9))
                    current3_median.append(np.median(f9))
                    current3_std.append(np.std(f9))
                    current3_range.append(np.max(f9) - np.min(f9))             

data = {'transcript_id':id,
        'transcript_position': pos,
        '6-seq': seq,
        'time_1_mean': time1_mean,
        'time_1_median': time1_median,
        'time_1_std': time1_std,
        'stddev_1_mean': stddev1_mean,
        'stddev_1_median': stddev1_median,
        'stddev_1_std': stddev1_std,
        'current_1_mean': current1_mean,
        'current_1_median': current1_median,
        'current_1_std': current1_std,
        'current_1_range': current1_range,
        'time_2_mean': time2_mean,
        'time_2_median': time2_median,
        'time_2_std': time2_std,
        'stddev_2_mean': stddev2_mean,
        'stddev_2_median': stddev2_median,
        'stddev_2_std': stddev2_std,
        'current_2_mean': current2_mean,
        'current_2_median': current2_median,
        'current_2_std': current2_std,
        'current_2_range': current2_range,
        'time_3_mean': time3_mean,
        'time_3_median': time3_median,
        'time_3_std': time3_std,
        'stddev_3_mean': stddev3_mean,
        'stddev_3_median': stddev3_median,
        'stddev_3_std': stddev3_std,
        'current_3_mean': current3_mean,
        'current_3_median': current3_median,
        'current_3_std': current3_std,
        'current_3_range': current3_range}

df = pd.DataFrame(data)


# Merge parsed data with labels (if the data is used for training)
if use_type == 'train':
    labels_file = (base_path / file_containing_labels).resolve()
    data_info = pd.read_csv(labels_file)
    df = pd.merge(df, data_info, on = ['transcript_id', 'transcript_position'], how = 'left')
    
    
# To figure out the filename to save the dataframe as
root, extension = os.path.splitext(file_to_parse)
new_filename = root + '.csv'   # Saves the output as the original filename, but with a .csv extension instead of .json extension
save_path = (base_path / new_filename).resolve()
df.to_csv(save_path)