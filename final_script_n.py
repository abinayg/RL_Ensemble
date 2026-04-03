#from dummy import *
from test_helper import *
import sys
import os
#from google.auth.transport.requests import Request
#from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2 import service_account
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import datetime as dt

# ==============================================================================
# ⚠️ CRITICAL CONFIGURATION ⚠️
# Paste the ID of the Google Drive folder you manually shared with the Service Account.
# The bot will create the "Trading_Bot" folder INSIDE this shared directory.
# ==============================================================================
ROOT_SHARED_FOLDER_ID = "13aEoVnHq9s5WRffwjdz71Ztw4KOCz6Fw"

today = dt.date.today()
feature_cols  = ['close', 'volume', 'day', 'macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence', 'daily_return', 'Y_h1', 'Y_h3', 'Y_h5']       
           
tickers = [ 
    "JAMNAAUTO.NS",
    "VEDL.NS",
    "POWERGRID.NS",
    "SHRIRAMFIN.NS",
    "IRB.NS",
    "NATCOPHARM.NS",
    "ANDHRSUGAR.NS",
    "TTKPRESTIG.NS",
    "MOTHERSON.NS",
    "SHANTIGEAR.NS"
]

tickerconfig = TickerDownloadConfig(
                tickers = tickers,
                feature_cols = feature_cols,
                indicators = ['macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma'],
                horizons = (1,3,5),
                end_date = today + dt.timedelta(days=2)  #gives data up till yesterday
                ) 

tkr_decision_dict = {}
dec_str = ""
  
today = dt.date.today()

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_drive():
    """Authenticates using a Service Account from an environment variable."""
    creds_json_str = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    
    if not creds_json_str:
        raise ValueError("GCP_SERVICE_ACCOUNT_KEY environment variable is missing!")
    
    creds_dict = json.loads(creds_json_str)
    
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES)
    
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name, parent_id=None):
    """
    Creates a folder. If parent_id is provided, creates it INSIDE that folder.
    """
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    # ADDED: supportsAllDrives and includeItemsFromAllDrives to search shared locations
    results = service.files().list(
        q=query, 
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    items = results.get('files', [])

    if items:
        print(f"Found existing folder: '{folder_name}'")
        return items[0]['id']
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]

        # ADDED: supportsAllDrives to create
        folder = service.files().create(
            body=file_metadata, 
            fields='id',
            supportsAllDrives=True
        ).execute()
        
        print(f"Created new folder: '{folder_name}' (ID: {folder.get('id')})")
        return folder.get('id')

def upload_file(service, filename, folder_id, mime_type):
    """
    Uploads a file. If it exists, it OVERWRITES it. If not, it creates it.
    """
    name = os.path.basename(filename)
    
    query = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    
    # ADDED: supportsAllDrives to search
    results = service.files().list(
        q=query, 
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    files = results.get('files', [])
    
    media = MediaFileUpload(filename, mimetype=mime_type)

    if files:
        file_id = files[0]['id']
        
        # ADDED: supportsAllDrives to update
        service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True
        ).execute()
        print(f"Overwrote existing file: {name} (ID: {file_id})")
        
    else:
        file_metadata = {
            'name': name,
            'parents': [folder_id]
        }
        
        # ADDED: supportsAllDrives to create
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id',
            supportsAllDrives=True
        ).execute()
        print(f"Created new file: {name} (ID: {file.get('id')})")

# --- MAIN EXECUTION ---
if __name__ == '__main__':

    for tkr in tickers:
      print(f"tkr:{tkr}")
      predict_df = download_last_2_months(tkr, tickerconfig)
      predict_df = predict_df.tail(61)
      [dqn_model_seed_1000, env_seed_1000] = get_dqn_model_n_env(predict_df, mdl_rel_path="dqn_model_exp_seed_1000", episodic_length=61,  episodes=1, seed=100, grid_search=False )
      [dqn_model_seed_200, env_seed_200] = get_dqn_model_n_env(predict_df, mdl_rel_path="dqn_model_exp_seed_100", episodic_length=61,  episodes=1, seed=200, grid_search=False )
      [dqn_model_seed_999, env_seed_999] = get_dqn_model_n_env(predict_df, mdl_rel_path="dqn_model_exp_seed_999", episodic_length=61,  episodes=1, seed=300, grid_search=False )
  
      [a2c_model_seed_0, a2c_env_seed_0] = get_a2c_model_n_env(predict_df, mdl_rel_path="a2c_model_exp_seed_0", episodic_length=61,  episodes=1, seed=300, grid_search=False )
      [a2c_model_seed_1, a2c_env_seed_1] = get_a2c_model_n_env(predict_df, mdl_rel_path="a2c_model_exp_seed_1", episodic_length=61,  episodes=1, seed=300, grid_search=False )
      [a2c_model_seed_2, a2c_env_seed_2] = get_a2c_model_n_env(predict_df, mdl_rel_path="a2c_model_exp_seed_2", episodic_length=61,  episodes=1, seed=300, grid_search=False )
  
      [ppo_model_seed_100, ppo_env_seed_100] = get_ppo_model_n_env(predict_df, mdl_rel_path="ppo_model_exp_seed_100", episodic_length=61,  episodes=1, seed=300, grid_search=False )
      [ppo_model_seed_111, ppo_env_seed_111] = get_ppo_model_n_env(predict_df, mdl_rel_path="ppo_model_exp_seed_111", episodic_length=61,  episodes=1, seed=300, grid_search=False )
      [ppo_model_seed_222, ppo_env_seed_222] = get_ppo_model_n_env(predict_df, mdl_rel_path="ppo_model_exp_seed_222", episodic_length=61,  episodes =1, seed=300, grid_search=False )
  
      dqn_models = [dqn_model_seed_1000, dqn_model_seed_200, dqn_model_seed_999]
      dqn_envs = [env_seed_1000, env_seed_200, env_seed_999]
      dqn_votes, dqn_actions = ensemble_vote_dqn(dqn_models, dqn_envs, predict_df, episode_length=61)
      
      a2c_models = [a2c_model_seed_0, a2c_model_seed_1, a2c_model_seed_2]
      a2c_envs = [a2c_env_seed_0, a2c_env_seed_1, a2c_env_seed_2]
      a2c_votes, a2c_actions = ensemble_vote_dqn(a2c_models, a2c_envs, predict_df, episode_length=61)
      
      ppo_models = [ppo_model_seed_100, ppo_model_seed_111, ppo_model_seed_222]
      ppo_envs = [ppo_env_seed_100, ppo_env_seed_111, ppo_env_seed_222]
      ppo_votes, ppo_actions = ensemble_vote_dqn(ppo_models, ppo_envs, predict_df, episode_length=61)

      final_recommendations = plot_high_lev_ensemble_decisions(predict_df, 
                                     dqn_actions, 
                                     a2c_actions, 
                                     ppo_actions, 
                                     label_name=tkr,
                                     episode_length=61, 
                                     disable_dqn=True, 
                                     disable_ppo=False, 
                                     disable_a2c=False,
                                     date=today,
                                     plot_name="ensemble"
                                     )
      last_decision = final_recommendations[-1]
  
      if last_decision == 0:
        dec_str = "BUY"
      elif last_decision == 1:
        dec_str = "SELL"
      else:
        dec_str = "HOLD"
  
      tkr_decision_dict[tkr] = dec_str
  
    filename=f"ensemble_results_{today}.txt"
    with open(filename, 'w') as f:
      for key,value in tkr_decision_dict.items():
        wr_str = f"{key:<20} -> {value:>10}\n"
        f.write(wr_str)

    ################## CODE FOR THE GOOGLE DRIVE UPLOAD #####################

    # 1. Login
    service = authenticate_drive()

   print(f"----- Setting up Drive Structure -----")
    
    # 1. Hardcode your Trading_Bot folder ID here
    #TRADING_BOT_FOLDER_ID = "PASTE_YOUR_TRADING_BOT_FOLDER_ID_HERE"
    TRADING_BOT_FOLDER_ID = "13aEoVnHq9s5WRffwjdz71Ztw4KOCz6Fw"
    
    # 2. Create the daily folder directly inside Trading_Bot
    TARGET_FOLDER = f"{today}"
    sub_folder_id = get_or_create_folder(service, TARGET_FOLDER, parent_id=TRADING_BOT_FOLDER_ID) 
    # 3. Upload Text File
    if os.path.exists(f"ensemble_results_{today}.txt"):
        upload_file(service, f"ensemble_results_{today}.txt", sub_folder_id, "text/plain")

    # 4. Upload ALL PNG Images
    all_files = os.listdir('.')
    png_files = [f for f in all_files if f.lower().endswith('.png') and f"{today}" in f]
    
    if png_files:
        print(f"Found {len(png_files)} PNG images. Uploading...")
        for filename in png_files:
            print(f"Processing: {filename}")
            upload_file(service, filename, sub_folder_id, "image/png")
    else:
        print("No PNG files found in the current directory.")
