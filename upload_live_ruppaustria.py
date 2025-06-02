import os
import requests
import time
import urllib3
import traceback
from datetime import datetime
from class_Misc import Misc
from class_Settings import Settings

# Disable warnings because of missing certificates of the server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load configuration
CONFIG = Settings.load_yaml_config() # You have to set the right init file in Settings.py!

work_dir = CONFIG["work_dir"]
dir_path = os.path.join(work_dir, 'upload')  # Folder of files to upload into S3 bucket through Directus

# URL of the PHP upload script
url_php_upload = CONFIG["url_php_upload"]

# Time in seconds between directory scans for new files to upload
upload_time_wait = 60

# Initialize logging
mylog_live = Misc.start_logger ('upload_live_ruppaustria')

print(f"{datetime.now()} BAT upload_live_ruppaustria.py has started!\n")

while True:
    try:
        # List of files to store in Directus
        list_of_jpg_files = []

        # Iterate directory for _LP files
        for path in os.listdir(dir_path):
            # Check if the current path is a file
            if os.path.isfile(os.path.join(dir_path, path)) and path.endswith('_LP.jpg'):
                list_of_jpg_files.append(path)
        
        print('List of files:', list_of_jpg_files)

        for dir_jpg_file in list_of_jpg_files:
            path_img = os.path.join(dir_path, dir_jpg_file)
            print('Path img:', path_img)
            
            with open(path_img, 'rb') as img:
                name_img = os.path.basename(path_img)
                files = {'file': (name_img, img)}
                r = requests.post(url_php_upload, files=files, verify=False)
            
            print('Response:', r, 'Response text:', r.text)
            
            if r.status_code == 200:
                os.remove(path_img)
                print("Live POST written:", dir_jpg_file)
            else:
                mylog_live.error(f'ERROR Code - {r.status_code}; Live Image NOT written: {dir_jpg_file}')
                print("Live POST NOT written:", dir_jpg_file)
        
        print(f"{datetime.now()} BAT upload_live_ruppaustria.py is running - next upload run in {upload_time_wait}s")

    except requests.exceptions.RequestException as e:
        mylog_live.error(f'RequestException: {e}')
        mylog_live.error('Traceback: ' + traceback.format_exc())
        print(f'RequestException: {e}')
        print('Traceback:', traceback.format_exc())

    except Exception as e:
        mylog_live.error(f'Exception: {e}')
        mylog_live.error('Traceback: ' + traceback.format_exc())
        print(f'Exception: {e}')
        print('Traceback:', traceback.format_exc())

    time.sleep(upload_time_wait)
