import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
# Import socket module
import socket 
import json
import csv
import ast
# import pyspark      
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName('sparkdf').getOrCreate()
TCP_IP = "localhost"
TCP_PORT = 6100
s = socket.socket()   
s.connect((TCP_IP, TCP_PORT))

def recieve_data(TCP_IP,TCP_PORT):
    d=""
    u=""
    temp=""
  
    
	
    # dt=json.loads(d)
    # # df=spark.read.json(dt)
    # # df.show()
    # receive data from the server and decoding to get the string.
    
    while(True):
            d=s.recv(1024).decode()
            length = len(d)
            if(length==0):
                break
            temp=d.split("\n",1)
            u+=temp[0]
            if len(temp)==2:
                var=json.loads(u)
                fi=list(var.values())
                df=pd.DataFrame.from_dict(fi)
                u=temp[1]
    s.close()
#     return
recieve_data(TCP_IP,TCP_PORT) 
