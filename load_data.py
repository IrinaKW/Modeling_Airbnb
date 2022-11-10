import requests
import pandas as pd
import os
import zipfile

link='https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip'

def upload_ziplink(link):
    req=requests.get(link)
    filename=link.split('/')[-1]
    with open(filename,'wb') as output_file:
        output_file.write(req.content)

    with zipfile.ZipFile(filename,"r") as zip_ref:     
        zip_ref.extractall()
    os.remove(filename)

upload_ziplink(link)
