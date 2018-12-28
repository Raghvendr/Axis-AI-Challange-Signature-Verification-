# Axis-AI-Challange-Signature-Verification
Offline signature verification using the siamese network


1)Please use this link to download the model files
  https://drive.google.com/open?id=1ezGYq8OOWUp3GSQkuMI-FihTw29uoROT
  
  a) download the model, unzip it and keep the folder in the same folder where my codes(mentioned in step 2 and 3) are there.

2) preprocessing.py file is to preprocess the image  ----> you don't need to do anything but keep the file on the same folder(working directory)

3) test_python_file.py   ---> you need to run this file after making the changes mentioned in 3b and 3c points. This will generate a csv file named "Team_datascience_final_output.csv", contains the forgery flag.
    a) loads the model 
    b) needs the folder name with path where the genuine signatures of the users are stored. Indexing to find the user has been done in the following way filename[3:6] ---> register_user_folder =""  (python code)
    c) needs the folder name with path where test images are stored  ----> test_folder =" "
    
    
    
Let me know incase of any query. call me on 7506245814
