import requests
import os
import zipfile


def download_file_ziplink(link, file):
    req=requests.get(link)
    filename=link.split('/')[-1]
    with open(filename,'wb') as output_file:
        output_file.write(req.content)
    with zipfile.ZipFile(filename,"r") as zip_ref:     
        zip_ref.extract(file)
    os.remove(filename)


if __name__ == "__main__":
    link='https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip'
    file='tabular_data/listing.csv'
    download_file_ziplink(link, file)