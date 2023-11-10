# Prediction of m6A RNA modifications from direct RNA-Seq data
DSA4266 Project 2: Team synergy

A Random Forest classifier was trained with Python to predict m6A RNA modifications from direct RNA-Seq data. 

## Installation Guidelines and Instructions to Run the Method (with example)
1. Download the `for_student_evaluation.zip` onto your local laptop. 
2. Create an Ubuntu instance using Research Gateway.
3. Run the following on your local terminal, to copy the `for_student_evaluation.zip` from local laptop into the home directory of the Ubuntu instance:  
`scp -i path/to/local/ssh/key path/to/downloaded/zipped/file server_user@server_ip_address:~`   
`# eg. scp -i ~/DSA4266/AWS/dsa4266-synergy-sharedkey.pem ~/DSA4266/Project2/for_student_evaluation.zip ubuntu@122.248.227.219:~`  

4. Login to your Ubuntu instance and run the following installations:  
`sudo apt -y install python3-pip`
'sudo apt install unzip'   
`pip install pandas`    
`pip install numpy pandas scikit-learn==1.1.2`  

6. Within your Ubuntu instance, unzip the `for_student_evaluation.zip` using this command:  
`unzip ~/for_student_evaluation.zip`  
   
7. To generate the m6A predictions, run the following commands within the Ubuntu instance:  
`cd ~/for_student_evaluation/scripts`  
`python3 generate_predictions.py ../data`  
8. To view the prediction results:  
`cat ~/for_student_evaluation/scripts/result/data_predict.csv`
