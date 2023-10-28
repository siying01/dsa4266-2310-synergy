'''This script takes in an unzipped json file and parse it into a csv file
For more information, refer to the help page: python parse_json.py --help
Example usage: python parse_json_generalised.py ../../data/dataset1.json test'''

import argparse
import os
import pandas as pd
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Parses an unzipped json file into a csv file.')
parser.add_argument('file_to_parse', type=str, help='Path to the file to be parsed. Can be relative path, relative to the folder containing this script.')
parser.add_argument('use_type', type=str, choices=['train', 'test'], help='Usage type. Either train or test.')
parser.add_argument('--file_containing_labels', type=str, help='When use_type is "train", the path to the file containing the labels is required.')
args = parser.parse_args()

# access the arguments
file_to_parse = args.file_to_parse
use_type = args.use_type
file_containing_labels = args.file_containing_labels

# check if the third argument is required and provided
if use_type == 'train' and file_containing_labels is None:
    parser.error("The --file_containing_labels is required when use_type is 'train'.")


base_path = Path(__file__).parent
file_path = (base_path / file_to_parse).resolve()
print(file_path)

print(f"File To Parse: {file_path}")
print(f"To Be Used For: {use_type}")
if file_containing_labels is not None:
    print(f"File Containing Labels: {file_containing_labels}")
    
    
id, pos, seq, time1, stddev1, current1, time2, stddev2, current2, time3, stddev3, current3 = [],[],[],[],[],[],[],[],[],[],[],[]
with open(file_path) as data_file:
    #for each json object: eg. {'ENST00000000233': {'244': {'AAGACCA': [[0.00299, 2.06, 125.0, 0.0177, 10.4, 122.0, 0.0093, 10.9, 84.1]]}}}
    for line in data_file: 
        json_data = json.loads(line)
        for k1, v1 in json_data.items(): #k1: 'ENST00000000233'
            for k2, v2 in v1.items(): #k2 = '244
                for k3, v3 in v2.items(): #k3 = 'AAGACCA'
                    reads = v3
                    for read in reads: #each read is a new row
                        id.append(k1)
                        pos.append(int(k2))
                        seq.append(k3)
                        time1.append(read[0])
                        stddev1.append(read[1])
                        current1.append(read[2])
                        time2.append(read[3])
                        stddev2.append(read[4])
                        current2.append(read[5])
                        time3.append(read[6])
                        stddev3.append(read[7])
                        current3.append(read[8])
                    
data = {'transcript_id':id,
        'transcript_position': pos,
        '6-seq': seq,
        'time_1': time1,
        'stddev_1': stddev1,
        'mean_current_1': current1,
        'time_2': time2,
        'stddev_2': stddev2,
        'mean_current_2': current2,
        'time_3': time3,
        'stddev_3': stddev3,
        'mean_current_3': current3}

df = pd.DataFrame(data)

if use_type == "train":
    file2_path = (base_path / file_containing_labels).resolve()
    data_info = pd.read_csv(file2_path)

    # left join with data info file
    df = pd.merge(df, data_info, on = ['transcript_id', 'transcript_position'], how = "left")
    
    
# to figure out the filename to save the dataframe as
root, extension = os.path.splitext(file_to_parse)
new_filename = root + '.csv'
save_path = (base_path / new_filename).resolve()
df.to_csv(save_path)