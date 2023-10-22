'''This script takes in an unzipped json file and parse it into a csv file
The input file has to be at the same level as this script
It is left join-ed with data.info file to obtain the labels for training the model'''
import pandas as pd
import numpy as np
import json
from pathlib import Path

base_path = Path(__file__).parent.parent.parent.parent
file_path = (base_path / "data/dataset0.json").resolve()

id, pos, seq = [],[],[]
time1_mean, time1_median, time1_std, stddev1_mean, stddev1_median, stddev1_std, current1_mean, current1_median, current1_std, current1_range = [],[],[],[],[],[],[],[],[],[]
time2_mean, time2_median, time2_std, stddev2_mean, stddev2_median, stddev2_std, current2_mean, current2_median, current2_std, current2_range = [],[],[],[],[],[],[],[],[],[]
time3_mean, time3_median, time3_std, stddev3_mean, stddev3_median, stddev3_std, current3_mean, current3_median, current3_std, current3_range = [],[],[],[],[],[],[],[],[],[]

with open(file_path) as data_file:
    #for each json object: eg. {'ENST00000000233': {'244': {'AAGACCA': [[0.00299, 2.06, 125.0, 0.0177, 10.4, 122.0, 0.0093, 10.9, 84.1]]}}}
    for line in data_file: 
        json_data = json.loads(line)
        for k1, v1 in json_data.items(): #k1: 'ENST00000000233'
            id.append(k1)
            for k2, v2 in v1.items(): #k2 = '244
                pos.append(int(k2))
                for k3, v3 in v2.items(): #k3 = 'AAGACCA'
                    reads = v3
                    seq.append(k3)
                    f1, f2, f3, f4, f5, f6, f7, f8, f9 = [], [], [], [], [], [], [], [], []
                    numreads = 0
                    for read in reads: #each read is a new row
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
                    #consolidate read values for each bag
                    time1_mean.append(np.mean(f1)) #mean of dwelling
                    time1_median.append(np.median(f1)) #median of dwelling
                    time1_std.append(np.std(f1)) #standard deviation of dwelling
                    stddev1_mean.append(np.mean(f2)) #mean of standard deviation
                    stddev1_median.append(np.median(f2)) #median of standard deviation
                    stddev1_std.append(np.std(f2)) #standard deviation of standard deviation
                    current1_mean.append(np.mean(f3)) #mean of current
                    current1_median.append(np.median(f3)) #median of current
                    current1_std.append(np.std(f3)) #standard deviation of current
                    current1_range.append(np.max(f3) - np.min(f3)) #range of current
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

file2_path = (base_path / "data/data.info").resolve()
data_info = pd.read_csv(file2_path)

#left join with data info file
df_full = pd.merge(df, data_info, on = ['transcript_id', 'transcript_position'], how = "left")

save_path = (base_path / 'data/dataset0.csv').resolve()
df_full.to_csv(save_path)