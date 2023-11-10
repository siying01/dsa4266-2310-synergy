# Prediction of m6A RNA modifications from direct RNA-Seq data
DSA4266 Project 2: Team synergy

A Random Forest classifier was trained with Python to predict m6A RNA modifications from direct RNA-Seq data. 

## Installation Guidelines and Instructions to Run the Method (with example)
1. Download the `for_student_evaluation.zip` GitHub zipped file onto your local laptop. Unzip the file with the following command:
`unzip path/to/downloaded/zipped/file`  
`# eg. unzip ~/DSA4266/Project2/for_student_evaluation.zip`
2. Create an Ubuntu instance using Research Gateway.
3. Copy the `for_student_evaluation` folder from local laptop into the home directory of the Ubuntu instance, by running the following on your local terminal:  
`scp -r -i path/to/local/ssh/key path/to/downloaded/github/folder <server_user@server_ip_address>:~`  
`# eg. scp -r -i ../AWS/dsa4266-synergy-sharedkey.pem project2/for_student_evaluation ubuntu@122.248.227.219:~`

4. Login to your Ubuntu instance and run the following installations:  
`sudo apt -y install python3-pip`   
`pip install pandas`    
`pip install numpy pandas scikit-learn==1.1.2`  

5. To generate the m6A predictions, run the following commands within the Ubuntu instance:  
`cd ~/for_student_evaluation/scripts`  
`python3 generate_predictions.py ../data`  
6. To view the prediction results:  
`cat ~/for_student_evaluation/scripts/result/data_predict.csv`
