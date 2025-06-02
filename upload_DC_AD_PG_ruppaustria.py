import os
import requests
import time
from datetime import datetime
from class_Misc import Misc
from class_Settings import Settings

# Load configuration and set variables
CONFIG = Settings.load_yaml_config()  # You have to set the right init file in Settings.py!

server_address = CONFIG["server_address"]
directus_token = CONFIG["directus_token"]
url_image = f'{server_address}/files?access_token={directus_token}'  # upload path for file postings
url_cont_DC = f'{server_address}/items/temperatures?access_token={directus_token}'
url_cont_AD = f'{server_address}/items/contaminations?access_token={directus_token}'
work_dir = CONFIG["work_dir"]
dir_path = os.path.join(work_dir, 'upload')  # Folder of files to upload into S3 bucket through Directus

#set logging
mylogs_DC_AD_PG = Misc.start_logger ('upload_DC_AD_PG')

#start looping
upload_time_wait = 100  # time in seconds when program goes into directory and looks for files to upload
while True:
    try:
        # list to store files
        list_of_image_files = []
        # Iterate directory for files
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)) and (path.endswith('_AD.jpg') or path.endswith('_DC.jpg') or path.endswith('_PG.png')):
                if len(path) == 54:
                    list_of_image_files.append(path)
                else:
                    mylogs_DC_AD_PG.error(f"ERROR bad length of filename: {path}, length: {len(path)}")
        list_of_image_files.sort(key=lambda x: (not x.endswith('_DC.jpg'), x)) # lamda function puts all _DC files in front position of the list
        mylogs_DC_AD_PG.info(f'BAT upload_DC_AD_PG_ruppaustria is running; List of files to transfer: {list_of_image_files}')

        for path_image_file in list_of_image_files:
        # for path_image_file in (f for f in list_of_image_files if f.endswith("_DC.jpg")):  # LRRRRRRRRRRRRRRRRRRRRRRR only DC to get not AD into DIRECTUS 
            path_img = os.path.join(dir_path, path_image_file)
            try:
                with open(path_img, 'rb') as img:
                    name_img = os.path.basename(path_img)
                    files = {'image': (name_img, img, 'image/jpeg', {'Expires': '0'})}
                    with requests.Session() as s:
                        response = s.get(server_address)
                        mylogs_DC_AD_PG.info(f'server returncode: {response.status_code}')
                        
                        if response.status_code != 200:
                            mylogs_DC_AD_PG.error(f'Server returned error code: {response.status_code}')
                            continue  # Skip to the next file if server is not accessible 
                        if (path.endswith('_DC.jpg') or path.endswith('_AD.jpg')):   
                            r = s.post(url_image, files=files)                                            # write files to DIRECTUS
                        if r.status_code != 200:
                            mylogs_DC_AD_PG.error(f'Error uploading image: Status Code: {r.status_code}, Response: {r.text}')
                        else:
                            if (path.endswith('_DC.jpg') or path.endswith('_AD.jpg')): # label the written files fields with information
                                json_data = r.json()
                                image_id = json_data["data"]["id"]
                                temp_digits = path_image_file[42:44] + "." + path_image_file[45:47]
                                ts = f"{path_image_file[11:21]}T{path_image_file[24:26]}:{path_image_file[27:29]}:{path_image_file[30:32]}.{path_image_file[33:36]}+00:00"
                            
                                if path_img.endswith('_AD.jpg'):
                                    r_image = requests.post(url_cont_AD, json={
                                        "image": image_id,
                                        "temperature": temp_digits,
                                        "production_line": str(path_image_file[7]),
                                        "timestamp": ts
                                    })
                                elif path_img.endswith('_DC.jpg'):
                                    r_image = requests.post(url_cont_DC, json={
                                        "image": image_id,
                                        "temperature": temp_digits,
                                        "production_line": str(path_image_file[7]),
                                        "timestamp": ts
                                    })
                os.remove(path_img)
                mylogs_DC_AD_PG.info(f'OK remove file in BAT upload_DC_AD_PG_ruppaustria: {path_img[-23:]}')
               #      mylogs_DC_AD_PG_PG.error(f'IMAGE {path_image_file} not written: Status Code: {r_image.status_code}, Response: {r_image.text}')
            except Exception as e:
                mylogs_DC_AD_PG.error(f'Failed to process file {path_image_file}: {e}')
        
        mylogs_DC_AD_PG.info(f'BAT upload_DC_AD_PG_ruppaustria is running; next upload in {upload_time_wait}s')
    
    except Exception as e:
        mylogs_DC_AD_PG.error(f'Something went wrong in BAT upload_DC_AD_PG_ruppaustria.py: {e}')
    
    time.sleep(upload_time_wait)
