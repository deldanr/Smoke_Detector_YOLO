#############################################
# SMOKE DETECTOR V2.0 "SMOKY"
#
# Author: Daniel Eldan R.
# Date  : 12-2022
# Mail  : deldanr@gmail.com
# Name  : Scrapper
# Desc  : Download image from online cameras
############################################


#
# IMPORT BASE LIBRARIES
#
import numpy as np
import pandas as pd
import requests
import time
import subprocess

from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import detect as dt

#
# List of citys and cameras we are going to scrap
#
comunas = ["Rancagua","Valparaiso","Curacavi","Lago Rapel"]
links = ['https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCRG',
         #'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCIC',
         #'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCSN',
         'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCRD',
         #'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCVM',
         #'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCPD',
         'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCCV',
         'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCSO'
         #'https://aipchile.dgac.gob.cl/camara_ubicacion/show/designador/SCTL',
        ]

# Build a DataFrame with them
df = pd.DataFrame({"Comunas":comunas,"Links":links})

#
# Function to save the scrapped images
#
def save_images(url,comuna):
    respuesta = requests.get(
                url,
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'
                }
            )

    soup = BeautifulSoup(respuesta.text, 'html.parser')
    images = soup.find_all('img')
    
    i = 1
    for items in images:
        if "https" in items['src']:

            nombre = "static/test/"+comuna+"0"+str(i)+".jpg"
            i = i + 1
            img_data = requests.get(items['src']).content
            try:
                with open(nombre, 'wb') as handler:
                    handler.write(img_data)
            except:
                print("Error guardando imagenes de las camaras")

#
# Secondary function to save image of a single camera from another route
#            
def save_images2():
    comuna = ["Vichuquen"]
    link = ["https://images-webcams.windy.com/72/1632930172/current/full/1632930172.jpg"]
    df = pd.DataFrame({"Comuna": comuna, "link": link})
    
    for i in np.arange(len(df)):
        img_data = requests.get(df['link'][i]).content
        nombre = "static/test/"+df['Comuna'][i]+".jpg"
        try:
            with open(nombre, 'wb') as handler:
                handler.write(img_data)
        except:
            print("Error guardando imagenes alternativas")
            
#
# IMPORTANT function that update the scraped images and run inference on them - Run every 300 seconds
#
def actualiza_imagenes():
    while True:
        start = time.time()
        print("Actualizando fotos")
        
        for i in np.arange(len(df)):
            save_images(df['Links'][i],df['Comunas'][i])
        save_images2()
        
        subprocess.call(["python", "detect.py", "--exist-ok"])
        print(f"Actualización Terminada en {time.time()-start}")
        time.sleep(120)




