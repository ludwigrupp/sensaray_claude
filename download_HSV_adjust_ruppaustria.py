import os
import requests
import time
import numpy as np
import shutil
from datetime import datetime
import random
import cv2
import urllib3
from imageio import imread
from class_Misc import Misc
from class_HSVcalc import HSV_calc
from class_Settings import Settings
import logging

# mylog = Misc.start_logger('download_HSV_adjust_ruppaustria')

# Load configuration
CONFIG = Settings.load_yaml_config()
if not CONFIG:
    print("Failed to load configuration. Exiting program.")
    exit(1)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

server_address    = CONFIG["server_address"]
directus_token    = CONFIG["directus_token"]
work_dir          = CONFIG["work_dir"]
HSV_work_dir      = os.path.join(work_dir, "HSV_masks/")
HSV_file_name     = os.path.join(HSV_work_dir, f"HSV_mask_{CONFIG['line_name']}.npy")
HSV_file_name_raw = os.path.join(HSV_work_dir, f"HSV_mask_{CONFIG['line_name']}_raw.npy")

f2a_work_dir = os.path.join(work_dir, 'files2add_HSV/')
f2a_pics_HSV_added = os.path.join(f2a_work_dir, 'pics_HSV_added/')

flawless_ordner_ID =  '20ae7750-7e6f-4734-af6a-0ccdcd287dc4'
url_flawless       = f'https://ruppaustria_directus.sensaray.at/files?limit=-1&&filter[_and][0][_and][0][tags][_null]=true&filter[_and][1][_and][0][type][_nnull]=true&filter[_and][1][_and][1][folder][_eq]={flawless_ordner_ID}&access_token={directus_token}'

download_time_wait = 90  # besser aus CONFIG[download_time_wait]

print("BATCH download_HSV_adjust_ruppaustria.py has started!")

def download_flawless_files():
    try:
        with requests.Session() as session:
            response = session.get(server_address, verify=False)
            print(f"Server response: {response}")
            response = session.get(url_flawless, verify=False)
            response.raise_for_status()

            data = response.json().get("data", [])
            print(f"{len(data)} entries loaded")

            for item in data:
                file_id = item["id"]
                file_url = f"{server_address}/assets/{file_id}?access_token={directus_token}"
                response = session.get(file_url, verify=False, stream=True)
                response.raise_for_status()

                file_path = os.path.join(f2a_work_dir, item["filename_download"])
                with open(file_path, "wb") as file:
                    shutil.copyfileobj(response.raw, file)
                print(f"Downloaded {item['filename_download']} to {file_path}")
                # write file tag HSV processed, not to be downloaded again
                response = session.patch(
                    f"{server_address}/files/{file_id}?access_token={directus_token}",
                    json={"tags": "HSV_Processed"}
                )
    except requests.exceptions.RequestException as e:
        print(f"RequestException: {e}")
    except Exception as e:
        print(f"Exception: {e}", exc_info=True)

def process_HSV_files():
    try:
        if os.path.exists(HSV_file_name) and os.path.exists(HSV_file_name_raw ):
            HSV_mask           = np.load(HSV_file_name)
            HSV_mask_raw       = np.load(HSV_file_name_raw)
            old_HSV_mask       = HSV_mask.copy()
            old_HSV_mask_raw   = HSV_mask.copy()
            old_HSV_mask_count = np.sum(HSV_mask) 
        else:
            input("No HSV file found to update - go on or stop?")
            HSV_mask     = np.zeros((180, 256, 256), dtype=np.uint8)
            HSV_mask_raw = np.zeros((180, 256, 256), dtype=np.uint8)

        perc_of_pixels = round(np.sum(HSV_mask) / (180 * 256 * 256) * 100, 4)
        print(f"HSV_mask loaded!! Shape: {HSV_mask.shape}\nPercent of HSV values detected: {perc_of_pixels}%\n")     
        
        ### start loading the files and check them
        new_HSV_detected = False
        for filename in os.listdir(f2a_work_dir):
            file_path = os.path.join(f2a_work_dir, filename)
            if os.path.isfile(file_path) and (file_path.endswith('.png') or file_path.endswith('.jpg')):
                frame = cv2.imread(file_path)
                print(f'file to be processed: {file_path[-50:]}, frame.shape : {frame.shape}')
                if frame.shape[0] == 2000: frame = frame[:999,:,:]
                if frame is not None: 
                    HSV_frame                            = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    HSV_mask_raw, HSV_no_match_count_raw = HSV_calc.build_HSV_mask_from_HSV_frame(HSV_frame, HSV_mask_raw)
                    HSV_mask, HSV_no_match_count         = HSV_calc.build_HSV_mask_from_HSV_frame(HSV_frame, HSV_mask)
                    if HSV_no_match_count > 0:
                        new_HSV_detected = True
                    print(f'HSV_no_match_count  : {HSV_no_match_count} in processed file: {file_path[-50:]}') 
                    os.rename(file_path,  os.path.join(f2a_work_dir + 'pics_HSV_added', filename))

        count_of_HSV_values = old_HSV_mask_count
        count_of_HSV_values_new = np.sum(HSV_mask)
        print(f"HSV values added: {count_of_HSV_values_new - count_of_HSV_values}")
        
        if new_HSV_detected:
            print(f"Processing HSV_mask with neighbors! .............. ")
            HSV_mask_slices = HSV_calc.fill_lines_after_two_neighbors(HSV_mask)

            # prepare the names for archiving HSV mask
            HSV_work_dir_archive      = os.path.join(work_dir, "HSV_masks/archive/")
            timestamp         = datetime.now()
            timestamp_string  = timestamp.strftime('D_%Y-%m-%d_T_%H_%M_%S_%f')[:-7]
            HSV_file_name_archive     = os.path.join(HSV_work_dir_archive, f"HSV_mask_{CONFIG['line_name']}_{timestamp_string}.npy")
            HSV_file_name_archive_raw = os.path.join(HSV_work_dir_archive, f"HSV_mask_{CONFIG['line_name']}_{timestamp_string}_raw.npy")
            HSV_calc.write_HSV_mask_to_dir (HSV_file_name, HSV_mask_slices, HSV_file_name_archive, old_HSV_mask)
            HSV_calc.write_HSV_mask_to_dir (HSV_file_name_raw, HSV_mask_raw, HSV_file_name_archive_raw, old_HSV_mask_raw)
            count_of_HSV_values = old_HSV_mask_count
            count_of_HSV_values_new = np.sum(HSV_mask)
            print(f"HSV values added: {count_of_HSV_values_new - count_of_HSV_values} HSV_mask is updated/saved!")
    except Exception as e:
        print(f"Exception in HSV processing: {e}", exc_info=True)

while True:
    download_flawless_files()
    process_HSV_files()
    print(f"Running BATCH download_HSV_adjust_ruppaustria - next download run in {round(download_time_wait / 60, 2)} min")
    time.sleep(download_time_wait)
    
