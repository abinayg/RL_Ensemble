
from dummy import *
import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta

#import gymansium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np
from scipy import stats

@dataclass
class TickerDownloadConfig:
  tickers: list[str]  = field(default_factory=list)
  feature_cols: list[str] = field(default_factory=list)
  indicators: list[str] = field(default_factory=list)
  start_date : str = '2019-01-01'
  end_date: str = '2026-01-01'
  interval: str = '1d'
  horizons: tuple[int] = field(default_factory=tuple)

def download_last_2_months(ticker: str, tickerconfig: TickerDownloadConfig) -> pd.DataFrame:
  #today = dt.date.today()
  end = tickerconfig.end_date
  #today = dt.date(2025,12,11)
  start = end - timedelta(days=500)
  #end = today # to include yesterday
  print(f" {start}, {end}")


  df_raw = YahooDownloader(start_date = start, 
                           end_date = end,
                           ticker_list = [ticker]
                           ).fetch_data()

  fe = FeatureEngineer(  use_technical_indicator = True,
                         tech_indicator_list = tickerconfig.indicators,
                         use_vix = True,
                         use_turbulence = True,
                         #use_turbulence = False,
                         user_defined_feature = True
                       )

  processed = fe.preprocess_data(df_raw)
  list_ticker = processed["tic"].unique().tolist()

  processed = build_targets(processed, tickerconfig, use_absolute=True)
  list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
  combination = list(itertools.product(list_date,list_ticker))
  
  processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
  processed_full = processed_full[processed_full['date'].isin(processed['date'])]
  processed_full = processed_full.sort_values(['date','tic'])
  
  processed_full = processed_full.fillna(0)

  return processed_full 


#def prediction_by_dqn(predict_df, mdl_rel_path,  episodic_length,  episodes=1, seed=100, grid_search=False ):
def get_dqn_model_n_env (predict_df, mdl_rel_path,  episodic_length,  episodes=1, seed=100, grid_search=False ):
    found_downtrend = False

    # Create a new test environment using test_data
    #test_env_fns = [make_env_fn(test_df.iloc[start:start+episodic_length], asset_price_cols, feature_cols_per_asset, seed=123)]

    feature_cols  = ['close', 'volume', 'day', 'macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence', 'daily_return', 'Y_h1', 'Y_h3', 'Y_h5']
    asset_price_cols = ["close"]
    feature_cols_per_asset = [feature_cols]
    val_env = [make_env_fn(predict_df.tail(episodic_length), asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=123)]
    val_env = DummyVecEnv(val_env)
    norm_env = VecNormalize.load(f"./{mdl_rel_path}/dqn_vec_normalize.pkl", val_env)


    #Load the model
    #model = DQN.load("./working_model_DQN/dqn_multiticker.zip", env=norm_env)
    model = DQN.load(f"./{mdl_rel_path}/best_model.zip", env=norm_env)

    perc_changes_per_episode = [] 
    for ep in range(episodes):
        obs = norm_env.reset()
        done = False
        ep_ret = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = norm_env.step(action)
            if grid_search == False:
                #print(f"info: {infos}")
                pass
            if dones.any():
                break
        if isinstance(infos, list) and 'pv_after' in infos[0]:
            initial_cash = norm_env.venv.envs[0].initial_cash
            print(f"Final PV (episode {ep+1}): {infos[0]['pv_after']:.2f}") if grid_search == False else None
            perc_change = ((infos[0]['pv_after'] - initial_cash) / initial_cash) * 100
            print(f"Percentage change from initial cash: {perc_change:.2f}%") if grid_search == False else None
            perc_changes_per_episode.append(perc_change)
        elif isinstance(infos, dict) and 'pv_after' in infos:
            print(f"DICT Final PV (episode {ep+1}): {infos['pv_after']:.2f}")
    
    #call the plot function of the underlying env
    #don't plot for now
    #if episodes == 1:
    #  env = norm_env.venv.envs[0]  # This is your MultiAssetTradingEnv instance
    #  env.plot(asset_index=0)

    print("Mean eval return:", np.mean(perc_changes_per_episode))
    #return np.mean(perc_changes_per_episode)
    return [model, norm_env]


def get_ppo_model_n_env (predict_df, mdl_rel_path,  episodic_length,  episodes=1, seed=100, grid_search=False ):
    found_downtrend = False

    # Create a new test environment using test_data
    #test_env_fns = [make_env_fn(test_df.iloc[start:start+episodic_length], asset_price_cols, feature_cols_per_asset, seed=123)]

    feature_cols  = ['close', 'volume', 'day', 'macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence', 'daily_return', 'Y_h1', 'Y_h3', 'Y_h5']
    asset_price_cols = ["close"]
    feature_cols_per_asset = [feature_cols]
    val_env = [make_env_fn(predict_df.tail(episodic_length), asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=123)]
    val_env = DummyVecEnv(val_env)
    norm_env = VecNormalize.load(f"./{mdl_rel_path}/ppo_vec_normalize.pkl", val_env)


    #Load the model
    #model = DQN.load("./working_model_DQN/dqn_multiticker.zip", env=norm_env)
    model = PPO.load(f"./{mdl_rel_path}/best_model.zip", env=norm_env)

    perc_changes_per_episode = [] 
    for ep in range(episodes):
        obs = norm_env.reset()
        done = False
        ep_ret = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = norm_env.step(action)
            if grid_search == False:
                #print(f"info: {infos}")
                pass
            if dones.any():
                break
        if isinstance(infos, list) and 'pv_after' in infos[0]:
            initial_cash = norm_env.venv.envs[0].initial_cash
            print(f"Final PV (episode {ep+1}): {infos[0]['pv_after']:.2f}") if grid_search == False else None
            perc_change = ((infos[0]['pv_after'] - initial_cash) / initial_cash) * 100
            print(f"Percentage change from initial cash: {perc_change:.2f}%") if grid_search == False else None
            perc_changes_per_episode.append(perc_change)
        elif isinstance(infos, dict) and 'pv_after' in infos:
            print(f"DICT Final PV (episode {ep+1}): {infos['pv_after']:.2f}")
    
    #call the plot function of the underlying env
    #don't plot for now
    #if episodes == 1:
    #  env = norm_env.venv.envs[0]  # This is your MultiAssetTradingEnv instance
    #  env.plot(asset_index=0)

    print("Mean eval return:", np.mean(perc_changes_per_episode))
    #return np.mean(perc_changes_per_episode)
    return [model, norm_env]



def get_a2c_model_n_env (predict_df, mdl_rel_path,  episodic_length,  episodes=1, seed=100, grid_search=False ):
    found_downtrend = False

    # Create a new test environment using test_data
    #test_env_fns = [make_env_fn(test_df.iloc[start:start+episodic_length], asset_price_cols, feature_cols_per_asset, seed=123)]

    feature_cols  = ['close', 'volume', 'day', 'macd', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma', 'vix', 'turbulence', 'daily_return', 'Y_h1', 'Y_h3', 'Y_h5']
    asset_price_cols = ["close"]
    feature_cols_per_asset = [feature_cols]
    val_env = [make_env_fn(predict_df.tail(episodic_length), asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=123)]
    val_env = DummyVecEnv(val_env)
    norm_env = VecNormalize.load(f"./{mdl_rel_path}/a2c_vec_normalize.pkl", val_env)


    #Load the model
    #model = DQN.load("./working_model_DQN/dqn_multiticker.zip", env=norm_env)
    model = A2C.load(f"./{mdl_rel_path}/best_model.zip", env=norm_env)

    perc_changes_per_episode = [] 
    for ep in range(episodes):
        obs = norm_env.reset()
        done = False
        ep_ret = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = norm_env.step(action)
            if grid_search == False:
                #print(f"info: {infos}")
                pass
            if dones.any():
                break
        if isinstance(infos, list) and 'pv_after' in infos[0]:
            initial_cash = norm_env.venv.envs[0].initial_cash
            print(f"Final PV (episode {ep+1}): {infos[0]['pv_after']:.2f}") if grid_search == False else None
            perc_change = ((infos[0]['pv_after'] - initial_cash) / initial_cash) * 100
            print(f"Percentage change from initial cash: {perc_change:.2f}%") if grid_search == False else None
            perc_changes_per_episode.append(perc_change)
        elif isinstance(infos, dict) and 'pv_after' in infos:
            print(f"DICT Final PV (episode {ep+1}): {infos['pv_after']:.2f}")
    
    #call the plot function of the underlying env
    #don't plot for now
    #if episodes == 1:
    #  env = norm_env.venv.envs[0]  # This is your MultiAssetTradingEnv instance
    #  env.plot(asset_index=0)

    print("Mean eval return:", np.mean(perc_changes_per_episode))
    #return np.mean(perc_changes_per_episode)
    return [model, norm_env]





def ensemble_vote_dqn(models, envs, test_df, episode_length=61):
    """
    Runs an ensemble of 3 DQN models and determines the final action via Majority Voting.
    
    Inputs:
        models (list): List of 3 loaded SB3 models (e.g. [model1, model2, model3])
        envs (list): List of 3 vectorized environments corresponding to the models.
        test_df (pd.DataFrame): The dataframe used for tracking/rendering (context only).
        episode_length (int): How many steps to simulate (default 61).
        
    Outputs:
        all_votes (np.array): Shape (episode_length, 3). Raw votes from each model.
        final_recommendations (np.array): Shape (episode_length,). The consensus action.
    """
    
    # 1. Initialize storage
    # Shape: [TimeSteps, Num_Models]
    history_votes = [] 
    history_consenses = []
    
    # 2. Reset all environments to start fresh
    # We maintain a list of current observations for each model
    obs_list = [env.reset() for env in envs]
    
    #print(f"Starting Ensemble Prediction for {episode_length} steps...")


    #store first pv_before. Which is the same for all envs. pv_before = envs[0].venv.envs[0].pv_before:w
    #start_cash = envs[0].venv.envs[0].pv_before
    
    for step in range(episode_length):
        step_votes = []
        
        # 3. Collect Votes from each Model
        for i, model in enumerate(models):
            # SB3 predict returns (action, state)
            # deterministic=True ensures we use the trained policy, not random noise
            action, _ = model.predict(obs_list[i], deterministic=True)
            
            # Extract scalar value if it's inside an array
            if isinstance(action, np.ndarray):
                action = action.item()
                
            step_votes.append(action)
            
            # Step the environment forward to keep it synced
            # Note: We only need the new observation for the next loop
            obs_list[i], rewards, dones, infos = envs[i].step(np.array([action]))
            
        # 4. Determine Consensus (Majority Vote)
        # mode returns the most common value. 
        # If there is a tie (e.g. 0, 1, 2), it takes the smallest number by default.
        consensus_action = stats.mode(step_votes, keepdims=False)[0]
        
        # Store data
        history_votes.append(step_votes)
        history_consenses.append(consensus_action)
        
    # Convert to numpy arrays for easy analysis
    all_votes = np.array(history_votes)
    final_recommendations = np.array(history_consenses)
    
    return all_votes, final_recommendations



def plot_ensemble_decisions(test_df, final_recommendations, label_name=" ", episode_length=61):
    """
    Plots the Buy/Sell/Hold recommendations on the price chart.
    
    Inputs:
        test_df (pd.DataFrame): DataFrame with a 'Close' column.
        final_recommendations (np.array): Array of 0 (Buy), 1 (Sell), 2 (Hold).
        episode_length (int): Number of steps to plot (e.g., 61).
    """
    # 1. Handle Column Case Sensitivity
    if 'Close' in test_df.columns:
        price_col = 'Close'
    elif 'close' in test_df.columns:
        price_col = 'close'
    else:
        raise ValueError("test_df must have a 'Close' column.")

    # 2. Extract Data
    prices = test_df[price_col].values[:episode_length]
    decisions = final_recommendations[:episode_length]
    steps = np.arange(episode_length)
    
    # 3. Create Plot
    plt.figure(figsize=(12, 6))
    #plt.figure(figsize=(6, 6))
    
    # Plot the main price line
    plt.plot(steps, prices, label='Stock Price', color='black', linewidth=1.5, alpha=0.6)
    
    # 4. Overlay Decision Markers
    buy_indices = np.where(decisions == 0)[0]
    sell_indices = np.where(decisions == 1)[0]
    hold_indices = np.where(decisions == 2)[0]
    
    # Buy: Green Up Triangle
    if len(buy_indices) > 0:
        plt.scatter(buy_indices, prices[buy_indices], 
                    marker='^', color='green', s=100, label='Buy', zorder=5)
        
    # Sell: Red Down Triangle
    if len(sell_indices) > 0:
        plt.scatter(sell_indices, prices[sell_indices], 
                    marker='v', color='red', s=100, label='Sell', zorder=5)
        
    # Hold: Small Gray Dot (to show inaction)
    if len(hold_indices) > 0:
        plt.scatter(hold_indices, prices[hold_indices], 
                    marker='.', color='gray', s=30, label='Hold', alpha=0.5)

    # 5. Formatting
    plt.title(f"Ensemble Strategy: Buy/Sell Decisions {label_name}")
    plt.xlabel("Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()

def final_plot(test_df, dqn_recs, ppo_recs, a2c_recs, episode_length):
    """
    Plots the Buy/Sell/Hold recommendations for DQN, PPO, and A2C on three subplots.
    """
    # 1. Handle Column Case Sensitivity
    if 'Close' in test_df.columns:
        price_col = 'Close'
    elif 'close' in test_df.columns:
        price_col = 'close'
    else:
        raise ValueError("test_df must have a 'Close' column.")

    # 2. Extract Data
    prices = test_df[price_col].values[:episode_length]
    steps = np.arange(episode_length)
    
    recs_list = [('DQN', dqn_recs), ('PPO', ppo_recs), ('A2C', a2c_recs)]

    # 3. Create Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    for ax, (name, recs) in zip(axes, recs_list):
        # Plot the main price line
        ax.plot(steps, prices, label='Stock Price', color='black', linewidth=1.5, alpha=0.6)
        
        # Ensure recs are sliced to episode_length
        current_recs = np.array(recs)[:episode_length]

        # 4. Overlay Decision Markers
        buy_indices = np.where(current_recs == 0)[0]
        sell_indices = np.where(current_recs == 1)[0]
        hold_indices = np.where(current_recs == 2)[0]
        
        # Buy: Green Up Triangle
        if len(buy_indices) > 0:
            ax.scatter(buy_indices, prices[buy_indices], 
                        marker='^', color='green', s=100, label='Buy', zorder=5)
            
        # Sell: Red Down Triangle
        if len(sell_indices) > 0:
            ax.scatter(sell_indices, prices[sell_indices], 
                        marker='v', color='red', s=100, label='Sell', zorder=5)
            
        # Hold: Small Gray Dot (to show inaction)
        if len(hold_indices) > 0:
            ax.scatter(hold_indices, prices[hold_indices], 
                        marker='.', color='gray', s=30, label='Hold', alpha=0.5)

        # 5. Formatting
        ax.set_title(f"{name} Strategy: Buy/Sell Decisions")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    axes[-1].set_xlabel("Steps")
    plt.tight_layout()
    plt.show()



def calc_ensemble_vote_dqn_profit(models, envs, test_df, episode_length=61):
    """
    Runs an ensemble of 3 DQN models, executes Majority Voting, 
    and calculates the cumulative profit percentage.
    
    Inputs:
        models (list): List of 3 loaded SB3 models.
        envs (list): List of 3 vectorized environments.
        test_df (pd.DataFrame): DataFrame containing 'Close' price column.
        episode_length (int): How many steps to simulate.
        
    Outputs:
        all_votes (np.array): Votes from each model.
        final_recommendations (np.array): Consensus decisions.
        total_profit_pct (float): Total profit/loss in percentage.
    """
    
    # 1. Setup Data & Portfolio
    # Ensure we have the prices for calculation
    if 'Close' in test_df.columns:
        prices = test_df['Close'].values
    elif 'close' in test_df.columns:
        prices = test_df['close'].values
    else:
        raise ValueError("test_df must contain a 'Close' column.")
        
    # Portfolio State
    initial_balance = 1000.0  # Virtual starting cash
    balance = initial_balance
    holdings = 0.0
    in_position = False
    
    # Storage
    history_votes = [] 
    history_consenses = []
    
    # 2. Reset Environments
    obs_list = [env.reset() for env in envs]
    
    print(f"Starting Ensemble Prediction for {episode_length} steps...")
    
    for step in range(episode_length):
        current_price = prices[step]
        pv_before = balance + holdings * current_price
        step_votes = []
        
        # --- A. Collect Votes ---
        for i, model in enumerate(models):
            action, _ = model.predict(obs_list[i], deterministic=True)
            if isinstance(action, np.ndarray):
                action = action.item()
            step_votes.append(action)
            # Sync environment
            obs_list[i], _, _, _ = envs[i].step(np.array([action]))
            
        # --- B. Consensus ---
        # 0=Buy, 1=Sell, 2=Hold
        consensus_action = stats.mode(step_votes, keepdims=False)[0]
        
        # --- C. Execute Trade (Backtest Logic) ---
        if consensus_action == 0: # BUY
            if not in_position:
                # Buy as many shares as possible
                holdings = balance / current_price
                balance = 0.0
                in_position = True
                
        elif consensus_action == 1: # SELL
            if in_position:
                # Sell all holdings
                balance = holdings * current_price
                holdings = 0.0
                in_position = False
                
        # If 2 (Hold), do nothing
        
        pv_after = balance + holdings * current_price
        print(f"Step: {step} | PV Before: {pv_before:.2f} | PV After: {pv_after:.2f}")
        
        # Store Data
        history_votes.append(step_votes)
        history_consenses.append(consensus_action)
        
    # 3. Calculate Final Value
    # If still holding at the end, mark to market at the last known price
    final_balance = balance
    if in_position:
        final_balance = holdings * prices[episode_length - 1]
        
    total_profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
    
    # Convert to numpy arrays
    all_votes = np.array(history_votes)
    final_recommendations = np.array(history_consenses)
    
    print(f"Simulation Complete. Total Profit Change: {total_profit_pct:.2f}%")
    
    return all_votes, final_recommendations, total_profit_pct




    print(f"Simulation Complete. Total Profit Change: {total_profit_pct:.2f}%")
    
    return all_votes, final_recommendations, total_profit_pct







def plot_high_lev_ensemble_decisions(test_df, dqn_recommendations, a2c_recommendations, ppo_recommendations, label_name="", episode_length=61, disable_dqn=False, disable_ppo=False, disable_a2c=False,date="",plot_name=""):
    """
    Plots the Buy/Sell/Hold recommendations based on a majority vote (consensus) 
    of the provided models, with options to disable specific models.
    
    Returns:
        final_recommendations (np.array): The consensus actions.
    """
    # 1. Handle Column Case Sensitivity
    if 'Close' in test_df.columns:
        price_col = 'Close'
    elif 'close' in test_df.columns:
        price_col = 'close'
    else:
        raise ValueError("test_df must have a 'Close' column.")

    # 2. Extract Data
    prices = test_df[price_col].values[:episode_length]
    steps = np.arange(len(prices))
    if 'date' in test_df.columns:
        dates = test_df['date'].values[:episode_length]
    else:
        dates = np.arange(episode_length)
    
    # 3. Collect Valid Recommendations
    valid_recs = []
    model_labels = []
    
    if not disable_dqn:
        valid_recs.append(np.array(dqn_recommendations)[:episode_length])
        model_labels.append("DQN")
    
    if not disable_a2c:
        valid_recs.append(np.array(a2c_recommendations)[:episode_length])
        model_labels.append("A2C")
        
    if not disable_ppo:
        valid_recs.append(np.array(ppo_recommendations)[:episode_length])
        model_labels.append("PPO")
        
    if not valid_recs:
        print("No models selected for ensemble.")
        return np.array([])

    # Stack recommendations: Shape (n_models, episode_length)
    stacked_recs = np.vstack(valid_recs)
    
    # 4. Compute Consensus (Mode)
    # stats.mode along axis 0 (across models)
    mode_res = stats.mode(stacked_recs, axis=0, keepdims=False)
    final_recommendations = mode_res[0]
    counts = mode_res[1]
    
    # Ensure 1D array
    if final_recommendations.ndim > 1:
        final_recommendations = final_recommendations.flatten()
    if counts.ndim > 1:
        counts = counts.flatten()

    # Tie-breaker Logic
    # 1. If DQN is disabled (e.g. only A2C & PPO voting) and there is a tie, use DQN recommendation
    if disable_dqn and len(valid_recs) >= 2:
        dqn_arr = np.array(dqn_recommendations)[:episode_length].flatten()
        final_recommendations[counts <= 1] = dqn_arr[counts <= 1]
    
    # 2. If all three models are enabled and recommendations are totally different (1-1-1 split), assign HOLD (2)
    elif len(valid_recs) == 3:
        final_recommendations[counts <= 1] = 2

    # Calculate Profit
    initial_balance = 10000.0
    balance = initial_balance
    holdings = 0.0
    in_position = False
    
    for i, action in enumerate(final_recommendations):
        current_price = prices[i]
        
        if action == 0: # Buy
             if not in_position:
                 holdings = balance / current_price
                 balance = 0.0
                 in_position = True
        elif action == 1: # Sell
             if in_position:
                 balance = holdings * current_price
                 holdings = 0.0
                 in_position = False
                 
    # Final Valuation
    final_balance = balance
    if in_position:
        final_balance = holdings * prices[-1]
        
    profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
    print(f"Ensemble Profit Percentage: {profit_pct:.2f}%")

    # 5. Create Plot
    plt.figure(figsize=(12, 6))
    
    # Plot the main price line
    plt.plot(steps, prices, label='Stock Price', color='black', linewidth=1.5, alpha=0.6)
    
    # 6. Overlay Decision Markers
    decisions = final_recommendations
    
    buy_indices = np.where(decisions == 0)[0]
    sell_indices = np.where(decisions == 1)[0]
    hold_indices = np.where(decisions == 2)[0]
    
    # Buy: Green Up Triangle
    if len(buy_indices) > 0:
        plt.scatter(steps[buy_indices], prices[buy_indices], 
                    marker='^', color='green', s=100, label='Buy', zorder=5)
        for idx in buy_indices:
            plt.annotate(f"{prices[idx]:.2f}", (steps[idx], prices[idx]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        
    # Sell: Red Down Triangle
    if len(sell_indices) > 0:
        plt.scatter(steps[sell_indices], prices[sell_indices], 
                    marker='v', color='red', s=100, label='Sell', zorder=5)
        for idx in sell_indices:
            plt.annotate(f"{prices[idx]:.2f}", (steps[idx], prices[idx]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        
    # Hold: Small Gray Dot (to show inaction)
    if len(hold_indices) > 0:
        plt.scatter(steps[hold_indices], prices[hold_indices], 
                    marker='.', color='gray', s=30, label='Hold', alpha=0.5)

    # 7. Formatting
    title_str = "Ensemble (" + "+".join(model_labels) + "): Buy/Sell Decisions for " + label_name
    plt.title(title_str)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Format X-axis: Remove year, reduce clutter
    n_ticks = 12
    tick_indices = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
    if 'date' in test_df.columns:
        tick_labels = [str(dates[i])[5:] for i in tick_indices] # Assumes YYYY-MM-DD
    else:
        tick_labels = [str(dates[i]) for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=45)
    
    plt.text(0.95, 0.05, f"Profit: {profit_pct:.2f}%", transform=plt.gca().transAxes, fontsize=12, fontweight='bold', horizontalalignment='right', verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    #today = dt.date.today()
    plt.savefig(f"{plot_name}_{label_name}_for_{date}.png")
    plt.show()

    return final_recommendations