# Prediction of m6A RNA modifications from direct RNA-Seq data
DSA4266 Project 2: Team synergy

This project aims to train a machine learning classifier to predict m6A RNA modifications from direct RNA-Seq data. 

## For student evaluations, here are the steps to generate the predictions
1. Download the `for_student_evaluation` GitHub folder onto your local laptop.
2. Download the trained model from this link: https://drive.google.com/drive/folders/11kax7xqOQZgsfUbIwsQqtR6IN2KcFSFT?usp=drive_link
3. Copy the downloaded `randomforest.pkl` file into the folder `for_student_evaluation/scripts/`
4. Create an Ubuntu instance using Research Gateway.
5. Copy the `for_student_evaluation` folder from local laptop into the home directory of the Ubuntu instance, by running the following on your local terminal:  
`scp -r -i path/to/local/ssh/key path/to/downloaded/github/folder <server_user@server_ip_address>:~`  
`# eg. scp -r -i ../AWS/dsa4266-synergy-sharedkey.pem project2/for_student_evaluation ubuntu@122.248.227.219:~`

6. Login to your Ubuntu instance and run the following installations:  
`sudo apt -y install python3-pip`   
`pip install pandas`    
`pip install numpy pandas scikit-learn==1.1.2`  

7. To generate the m6A predictions, run the following commands within the Ubuntu instance:  
`cd ~/for_student_evaluation/scripts`  
`python3 generate_predictions.py ../data`  
8. To view the prediction results:  
`cat ~/for_student_evaluation/scripts/result/data_predict.csv`
