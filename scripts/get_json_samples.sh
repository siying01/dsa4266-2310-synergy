#!/bin/bash

## How to use this file
## 1. start in an empty directory you want to download the samples to
## 2. move this file to the empty directory
## 3. change permissions by running on cli: chmod +x get_json_samples.sh
## 4. run this file on cli by: ./get_json_samples.sh
## 5. result: all samples in .json.gz in the directory

# download json files from S3
 aws s3 cp --no-sign-request s3://sg-nex-data/data/processed_data/m6Anet/ . --recursive --exclude "*" --include "*.json"

# extract json from folders and rename json to sample's name
for dir in */; do
    dir=${dir%/}
    mv "$dir/data.json" "$dir.json"
done

# remove empty folders
find ./ -type d -exec rmdir {} \;

# compress json files
for file in *.json; do
   # to keep the uncompressed file: gzip -c "$file" > "$file.gz"
   gzip $file
done







