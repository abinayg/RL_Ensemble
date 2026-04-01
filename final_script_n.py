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

today = dt.date.today()
feature_cols  = ['close', 'volume', 'day', 'macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence', 'daily_return', 'Y_h1', 'Y_h3', 'Y_h5']       
           
#tickers = ['HDFCLIFE.NS',
#            'JSWSTEEL.NS',
#            'IRCTC.NS' ,
#            'ASHOKLEY.NS',
#            'SALZERELEC.NS',
#            'BIOCON.NS',
#            'CAMS.NS',
#            'TATACHEM.NS',
#            'POCL.NS',
#            'MANAPPURAM.NS'
#          ]

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
                #end_date = today  ,
                #end_date=today - dt.timedelta(days=1)
                ) 

tkr_decision_dict = {}
dec_str = ""
  


#today = dt.date.today() - dt.timedelta(days=1)  #yesterday's date
today = dt.date.today()
# If modifying these scopes, delete the file token.json.
#SCOPES = ['https://www.googleapis.com/auth/drive.file']
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_drive():
    """Authenticates using a Service Account from an environment variable."""
    # We will pass the JSON content via a GitHub Secret
    creds_json_str = os.environ.get('GCP_SERVICE_ACCOUNT_KEY')
    
    if not creds_json_str:
        raise ValueError("GCP_SERVICE_ACCOUNT_KEY environment variable is missing!")
    
    # Parse the JSON string back into a dictionary
    creds_dict = json.loads(creds_json_str)
    
    # Authenticate using the dictionary
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES)
    
    return build('drive', 'v3', credentials=creds)

#def authenticate_drive():
#    """Authenticates the user and returns the Drive service."""
#    creds = None
#    # The file token.json stores the user's access and refresh tokens.
#    if os.path.exists('token.json'):
#        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#    
#    # If there are no (valid) credentials available, let the user log in.
#    if not creds or not creds.valid:
#        if creds and creds.expired and creds.refresh_token:
#            creds.refresh(Request())
#        else:
#            flow = InstalledAppFlow.from_client_secrets_file(
#                'credentials.json', SCOPES)
#            creds = flow.run_local_server(port=0)
#        # Save the credentials for the next run
#        with open('token.json', 'w') as token:
#            token.write(creds.to_json())
#
#    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name, parent_id=None):
    """
    Creates a folder. If parent_id is provided, creates it INSIDE that folder.
    """
    # 1. Search to see if it already exists in the specific parent
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    if items:
        print(f"Found existing folder: '{folder_name}'")
        return items[0]['id']
    else:
        # 2. Create it if not found
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]  # <--- THIS MAKES IT A SUB-DIRECTORY

        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Created new folder: '{folder_name}' (ID: {folder.get('id')})")
        return folder.get('id')
def upload_file(service, filename, folder_id, mime_type):
    """
    Uploads a file. If it exists, it OVERWRITES it. If not, it creates it.
    """
    name = os.path.basename(filename)
    
    # 1. Search for existing file with the same name in the specific folder
    query = f"name = '{name}' and '{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id)").execute()
    files = results.get('files', [])
    
    media = MediaFileUpload(filename, mimetype=mime_type)

    if files:
        # --- OVERWRITE MODE ---
        # We found the file! Get its ID.
        file_id = files[0]['id']
        
        # use .update() instead of .create()
        service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"Overwrote existing file: {name} (ID: {file_id})")
        
    else:
        # --- CREATE MODE ---
        # File doesn't exist, create a new one.
        file_metadata = {
            'name': name,
            'parents': [folder_id]
        }
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Created new file: {name} (ID: {file.get('id')})")

# --- MAIN EXECUTION ---
if __name__ == '__main__':


    for tkr in tickers:
      #dummy_ticker_list = [tkr]
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

      #only dqn
#      final_recommendations = plot_high_lev_ensemble_decisions(predict_df, 
#                                     dqn_actions, 
#                                     a2c_actions, 
#                                     ppo_actions, 
#                                     label_name=tkr,
#                                     episode_length=61, 
#                                     disable_dqn=False, 
#                                     disable_ppo=True, 
#                                     disable_a2c=True,
#                                     date=today,
#                                     plot_name="dqn"
#                                     )
#
# #    only a2c
#      final_recommendations = plot_high_lev_ensemble_decisions(predict_df, 
#                                     dqn_actions, 
#                                     a2c_actions, 
#                                     ppo_actions, 
#                                     label_name=tkr,
#                                     episode_length=61, 
#                                     disable_dqn=True, 
#                                     disable_ppo=True, 
#                                     disable_a2c=False,
#                                     date=today,
#                                     plot_name="a2c"
#                                     )
# #    only ppo
#      final_recommendations = plot_high_lev_ensemble_decisions(predict_df, 
#                                    dqn_actions, 
#                                    a2c_actions, 
#                                    ppo_actions, 
#                                    label_name=tkr,
#                                    episode_length=61, 
#                                    disable_dqn=True, 
#                                    disable_ppo=False, 
#                                    disable_a2c=True,
#                                    date=today,
#                                    plot_name="ppo"
#
#                                     )




      #ensemble plot with only a2c and ppo 
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
  
  
  
  
    #store the results in a text file
    filename=f"ensemble_results_{today}.txt"
    with open(filename, 'w') as f:
      for key,value in tkr_decision_dict.items():
        wr_str = f"{key:<20} -> {value:>10}\n"
        f.write(wr_str)
        #f.write(key + "\t\t" + "->" + value + "\n")












    ################## CODE FOR THE GOOGLE DRIVE UPLOAD #####################


    # 1. Login
    service = authenticate_drive()

    print(f"----- Setting up Drive Structure -----")
    main_folder_id = get_or_create_folder(service, "Trading_Bot")
    
    # 2. Define target folder in Drive
    TARGET_FOLDER = f"{today}"
    sub_folder_id = get_or_create_folder(service, TARGET_FOLDER, parent_id=main_folder_id)
    
    # 3. Upload Text File (Optional, kept from before)
    if os.path.exists(f"ensemble_results_{today}.txt"):
        upload_file(service, f"ensemble_results_{today}.txt", sub_folder_id, "text/plain")

    # 4. Upload ALL PNG Images
    # Get list of all files in current directory
    all_files = os.listdir('.')
    
    # Filter for .png files (case insensitive)
    png_files = [f for f in all_files if f.lower().endswith('.png') and f"{today}" in f]
    
    if png_files:
        print(f"Found {len(png_files)} PNG images. Uploading...")
        for filename in png_files:
            print(f"Processing: {filename}")
            upload_file(service, filename, sub_folder_id, "image/png")
    else:
        print("No PNG files found in the current directory.")
