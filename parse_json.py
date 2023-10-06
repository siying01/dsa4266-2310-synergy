'''This script takes in an unzipped json file and parse it into a csv file
The input file has to be at the same level as this script'''
import pandas as pd
import json
from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / "dataset0.json").resolve()
print(file_path)


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

file2_path = (base_path / "data.info").resolve()
data_info = pd.read_csv(file2_path)

#left join with data info file
df_full = pd.merge(df, data_info, on = ['transcript_id', 'transcript_position'], how = "left")
#print(df_full.head())
save_path = (base_path / 'dataset0.csv').resolve()
df_full.to_csv(save_path)
print(len(df_full['mean_current_3']))