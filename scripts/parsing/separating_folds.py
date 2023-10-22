"""This script separates the samples into 5-different folds for cross-validation
Separation is done by gene id such that transcripts with the same gene ids will all be in the same fold"""
import pandas as pd
from pathlib import Path
import random

base_path = Path(__file__).parent.parent
file_path = (base_path / "data" / "data.info").resolve()
data_info = pd.read_csv(file_path)

file_path2 = (base_path / "data" / "full_data.csv").resolve()
df = pd.read_csv(file_path2)

gene = data_info['gene_id'].unique() #Number of unique gene ids: 3852
random.shuffle(gene) #Shuffle the gene id randomly

folds = [1,2,3,4,5] * 770
folds.extend([1,2]) #Separate all 3852 gene ids into 5 folds evenly

data_dict = {'gene_id': gene,
             'folds': folds}

gene_folds = pd.DataFrame(data_dict)

#join with the main dataframe so that we know which samples goes to which fold during cross-validation
full_dataset = pd.merge(df, gene_folds, on = 'gene_id', how = "left")

save_path = (base_path / "data" / "data_folded_k5.csv").resolve()
full_dataset.to_csv(save_path)