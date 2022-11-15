#%%
import pandas as pd
import glob
from load_data import download_file_ziplink
import requests
import zipfile
import os
import cv2

def download_images(link):
    path='tabular_data/clean_tabular_data.csv'
    df=pd.read_csv(path)

    req=requests.get(link)
    filename=link.split('/')[-1]
    with open(filename,'wb') as output_file:
        output_file.write(req.content)
    with zipfile.ZipFile(filename,"r") as zip_ref:     
        png_file_names=zip_ref.namelist()

        for png_file in png_file_names:
            for i in df['ID']:
                if png_file.__contains__(i) & png_file.endswith('.png'):
                    zip_ref.extract(png_file)
                
                pass


def resize_images():
    path='tabular_data/clean_tabular_data.csv'
    df=pd.read_csv(path)
    h_smallest=156
    for i in df['ID']:
        path_png_dir='images/'+str(i)
        # This block provides the smallest height avaliable
        # try:
        #     files = os.listdir(path_png_dir)
        #     for file in files:
        #         im = cv2.imread(path_png_dir+'/'+file)
        #         h, w, c = im.shape

        #         if h_smallest>h:
        #             h_smallest=h
        #     print(h_smallest)
        # except:
        #     pass

    for i in df['ID']:
        path_png_dir='images/'+str(i)
        base_height=156
        try:
            files = os.listdir(path_png_dir)
            for file in files:
                im = cv2.imread(path_png_dir+'/'+file)
                h, w, c = im.shape
                ratio=h/w
                base_width=int(base_height/ratio)
                if c!=3:
                    os.remove(file)
                    print('deleted not RBG')
                else:
                    file_name='data/processed_images/'+file
                    img=cv2.resize(im,(base_width, base_height))
                    cv2.imwrite(file_name, img)
        except:
            pass
                    

if __name__ == "__main__":
    link='https://aicore-project-files.s3.eu-west-1.amazonaws.com/airbnb-property-listings.zip'
    download_images(link)
    resize_images()
    


