

import numpy as np
import itertools
import pandas as pd
import torch
import talib as ta
import importlib
from itertools import product



from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = -1
from dataclasses import dataclass, field

#helpful imports for bulding features using tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

from stable_baselines3.common.utils import set_random_seed

from stable_baselines3 import PPO, SAC, A2C
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

from MultiAssetTradingEnv import *



def build_targets(df, tickerconfig, use_absolute=True):
   #TO-DO: Fill out this function 
    out = []
    for tkr, sub in df.groupby("tic"):
        s = sub.copy()
        for i,h in enumerate(tickerconfig.horizons):
            if use_absolute: 
              s[f"Y_h{h}"] = s["close"].shift(-h) / s["close"]
            else:
              s[f"Y_h{h}"]  = np.log(s["close"].shift(-h) / s["close"])
        out.append(s)
    ydf = pd.concat(out).sort_index()
    return ydf

def split_and_shuffle(df, episode_length=45, ticker_col='tic'):
    """
    Splits the dataframe into chronological blocks of 'episode_length' for each ticker,
    then shuffles the blocks across all tickers. Chronological order is maintained within each block.

    Args:
        df (pd.DataFrame): Input dataframe containing multiple tickers.
        episode_length (int): Number of rows per episode.
        ticker_col (str): Column name for ticker symbol.

    Returns:
        List[pd.DataFrame]: List of shuffled blocks, each of length 'episode_length'.
    """
    blocks = []
    tickers = df[ticker_col].unique()
    for tic in tickers:
        sub_df = df[df[ticker_col] == tic].sort_values('date')
        # Split into blocks of episode_length
        for i in range(0, len(sub_df), episode_length):
            block = sub_df.iloc[i:i+episode_length]
            if len(block) == episode_length:
                blocks.append(block)
    np.random.shuffle(blocks)
    shuffled_df = pd.concat(blocks, ignore_index=True)
    return shuffled_df

#s_df = split_and_shuffle(processed_full, episode_length=30)
#print(s_df.sample())

def split_train_test_by_blocks(shuffled_df, test_ratio=0.2, episode_length=45):
    """
    Splits the shuffled DataFrame (output of split_and_shuffle) into train and test sets by episode blocks.

    Args:
        shuffled_df (pd.DataFrame): Concatenated DataFrame of shuffled blocks.
        test_ratio (float): Fraction of blocks to use for testing.
        episode_length (int): Number of rows per episode.

    Returns:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Testing data.
    """
    n_blocks = len(shuffled_df) // episode_length
    n_test = int(n_blocks * test_ratio)
    n_train = n_blocks - n_test

    # Split by blocks
    blocks = [shuffled_df.iloc[i*episode_length:(i+1)*episode_length] for i in range(n_blocks)]

    # Calculate correlation for each block to stratify the split
    high_corr_blocks = []
    low_corr_blocks = []

    for block in blocks:
        # Use 'correlation' column if available, otherwise calculate trend correlation of 'close'
        if 'correlation' in block.columns:
            corr = block['correlation'].mean()
        elif 'close' in block.columns:
            y = block['close'].values
            x = np.arange(len(y))
            # Avoid division by zero if price is constant
            corr = np.corrcoef(x, y)[0, 1] if np.std(y) > 1e-9 else 0.0
        else:
            corr = 0.0

        if corr >= 0.5:
            high_corr_blocks.append(block)
        else:
            low_corr_blocks.append(block)

    # Ensure 50% of train data has correlation >= 0.5 and 50% <= 0.5
    n_train_high = n_train // 2
    n_train_low = n_train - n_train_high

    train_blocks = high_corr_blocks[:n_train_high] + low_corr_blocks[:n_train_low]
    np.random.shuffle(train_blocks)

    # Remaining blocks go to test
    test_blocks = high_corr_blocks[n_train_high:] + low_corr_blocks[n_train_low:]
    np.random.shuffle(test_blocks)

    train_df = pd.concat(train_blocks, ignore_index=True)
    test_df = pd.concat(test_blocks, ignore_index=True)

    #reset the indexes
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df




def make_env_fn(df, asset_price_cols, feature_cols_per_asset, action_space_type, seed=0):
    def _init():
        #set_random_seed(run_seed)
        env = MultiAssetTradingEnv(
            seed = seed,
            df=df,
            asset_price_cols=asset_price_cols,
            feature_cols_per_asset=feature_cols_per_asset,
            action_space_type=action_space_type ,
            initial_cash=100_000.0,
            reserve_default=0.01,
            reserve_min=0.01,
            confidence_threshold=0.6,
            commission_rate=0.0005,
            slippage_rate=0.0005,
            exposure_change_penalty=0.5,
            max_short_exposure=1.0,
            normalize_obs=False, #TO-DO: vecnormalize will take care of this?
            verbose=False,
            episode_length=60,  # Set a fixed episode length
            include_transaction_and_slippage_cost=False
        )
        #env.seed(seed)
        return env
    return _init



def model_params(model):
    #create empty dict
    params = {}

    
    # 1. Basic Hyperparameters
    #print(f"Learning Rate:      {model.learning_rate}")
    params["learning_rate"] = model.learning_rate
    #print(f"Batch Size:         {model.batch_size}")
    params["batch_size"] = model.batch_size
    #print(f"Gamma (Discount):   {model.gamma}")
    params["gamma"] = model.gamma
    #print(f"Buffer Size:        {model.buffer_size}")
    params["buffer_size"] = model.buffer_size
    
    # 2. Exploration
    #print(f"Exploration Frac:   {model.exploration_fraction}")
    params["exploration_fraction"] = model.exploration_fraction
    #print(f"Exploration Final:  {model.exploration_final_eps}")
    #print(f"Exploration Init:   {model.exploration_initial_eps}")
    
    # 3. Training Loop Details
    #print(f"Train Freq:         {model.train_freq}")
    params["train_freq"] = model.train_freq
    #print(f"Gradient Steps:     {model.gradient_steps}")
    params["gradient_steps"] = model.gradient_steps
    #print(f"Target Update Interval:  {model.target_update_interval}")
    params["target_update_interval"] = model.target_update_interval
    #print(f"Learning Starts:    {model.learning_starts}")
    params["learning_starts"] = model.learning_starts
    
    # 4. Architecture (The Brain)
    # The architecture is nested inside 'policy_kwargs'
    if model.policy_kwargs and 'net_arch' in model.policy_kwargs:
        #print(f"Network Arch:       {model.policy_kwargs['net_arch']}")
        params["net_arch"] = model.policy_kwargs['net_arch']
    else:
        #print("Network Arch:       Default [64, 64]")
        params["net_arch"] = [64, 64]
        
    #print("=" * 30)

    return params

# Usage
# print_model_params(model)






#==========================#########################################

def train_dqn(
                train_df, 
                asset_price_cols, 
                feature_cols_per_asset, 
                total_timesteps=200_000,
                n_envs=1,
                eval_verbose=0,
                seed = 123,
                model_save_path="./dqn_model",
                **dqn_kwargs
             ):
    
    #model_params = {}
    """
    Trains a single PPO agent sequentially on all tickers in train_data_dict.
    The agent is updated with each ticker's data in sequence.

    Args:
        train_data_dict (dict): Dict of {ticker: df} for training.
        asset_price_cols (list): Asset price columns per asset.
        feature_cols_per_asset (list): Feature columns per asset.
        total_timesteps (int): Timesteps per ticker.
        n_envs (int): Number of parallel envs (usually 1 for sequential).

    Returns:
        model (PPO): Trained PPO model (after all tickers).
        vec_env (VecNormalize): VecNormalize wrapper for last ticker.
    """

    #policy_kwargs = dict(
    #    net_arch=dict(pi=[64,64], vf=[64,64])
    #)

    kwargs = dict(
                policy="MlpPolicy",
                learning_rate=1e-4,
                #n_steps=1,
                buffer_size=500_000,
                learning_starts=1000,
                #seed = 123,
                #tau=1.0,
                gamma=0.97,    
                train_freq=4,
                gradient_steps=1,
                #gradient_steps=1,
                exploration_fraction=0.2,
                exploration_final_eps=0.1,
                batch_size = 256,
                exploration_initial_eps=1.0,
                target_update_interval=60,
                device='cpu',
                verbose=0,
                #ent_coef='auto',
                #policy_kwargs = dict(pi=[64,64], qf=[64.64])
                policy_kwargs = dict(net_arch=[64,64]),
                tensorboard_log="./dqn_tensorboard/"
                )

    kwargs.update(dqn_kwargs)

    #print(", ".join(f"{k}: {v}" for k, v in kwargs.items()))

    feature_cols_per_asset = [col for col in feature_cols_per_asset if col != 'close']


    # set random seed and pass on to the DQN A
    set_random_seed(seed)

    env_fns = [make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    #checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/", name_prefix=f"dqn_model")
    stop_train_callback = StopTrainingOnNoModelImprovement(
                            max_no_improvement_evals=5,
                            #min_evals=400,
                            verbose = eval_verbose)

    eval_env = DummyVecEnv([make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=model_save_path,
                                 eval_freq=100_000,
                                 callback_after_eval=stop_train_callback,
                                 verbose=eval_verbose)

    model = DQN(
        env=vec_env,
        seed = seed,
        **kwargs,
        #policy_kwargs=policy_kwargs,
        #ent_coef='auto',
        #clip_range=0.2,

    )

    model_params_dict = model_params(model)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    vec_env.save(f"{model_save_path}/dqn_vec_normalize.pkl")
    model.save(f"{model_save_path}/dqn_multiticker")


    return model, vec_env, model_params_dict



# ==========================#########################################
### PPO Model

def train_ppo(
                train_df, 
                asset_price_cols, 
                feature_cols_per_asset, 
                total_timesteps=200_000,
                n_envs=1,
                eval_verbose=0,
                seed = 123,
                model_save_path="./ppo_model",
                **ppo_kwargs
             ):
    
    #model_params = {}
    """
    Trains a single PPO agent sequentially on all tickers in train_data_dict.
    The agent is updated with each ticker's data in sequence.

    Args:
        train_data_dict (dict): Dict of {ticker: df} for training.
        asset_price_cols (list): Asset price columns per asset.
        feature_cols_per_asset (list): Feature columns per asset.
        total_timesteps (int): Timesteps per ticker.
        n_envs (int): Number of parallel envs (usually 1 for sequential).

    Returns:
        model (PPO): Trained PPO model (after all tickers).
        vec_env (VecNormalize): VecNormalize wrapper for last ticker.
    """

    #policy_kwargs = dict(
    #    net_arch=dict(pi=[64,64], vf=[64,64])
    #)

    kwargs = dict(
                policy="MlpPolicy",
                learning_rate=1e-4,
                n_steps=1,
                gamma=0.97,    
                batch_size = 256,
                device='cpu',
                verbose=0,
                gae_lambda=0.95,
                ent_coef='0.01',
                clip_range=0.2,
                #policy_kwargs = dict(pi=[64,64], qf=[64.64])
                policy_kwargs = dict(net_arch=[64,64]),
                tensorboard_log="./ppo_dqn_tensorboard/"
                )

    kwargs.update(ppo_kwargs)

    #print(", ".join(f"{k}: {v}" for k, v in kwargs.items()))

    feature_cols_per_asset = [col for col in feature_cols_per_asset if col != 'close']


    # set random seed and pass on to the DQN A
    set_random_seed(seed)

    env_fns = [make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    #checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/", name_prefix=f"dqn_model")
    stop_train_callback = StopTrainingOnNoModelImprovement(
                            max_no_improvement_evals=5,
                            #min_evals=400,
                            verbose = eval_verbose)

    eval_env = DummyVecEnv([make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_save_path}",
                                 eval_freq=100_000,
                                 callback_after_eval=stop_train_callback,
                                 verbose=eval_verbose)

    model = PPO(
        env=vec_env,
        seed = seed,
        **kwargs,
        #policy_kwargs=policy_kwargs,
        #ent_coef='auto',
        #clip_range=0.2,

    )

    model_params_dict = ppo_model_params(model)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    vec_env.save(f"{model_save_path}/ppo_vec_normalize.pkl")
    model.save(f"{model_save_path}/ppo_multiticker")


    return model, vec_env, model_params_dict

def ppo_model_params(model):
    #create empty dict
    params = {}
 
    # 1. Basic Hyperparameters
    #print(f"Learning Rate:      {model.learning_rate}")
    params["learning_rate"] = model.learning_rate
    #print(f"Batch Size:         {model.batch_size}")
    params["batch_size"] = model.batch_size
    #print(f"Gamma (Discount):   {model.gamma}")
    params["gamma"] = model.gamma

    params["n_steps"] = model.n_steps
    params["ent_coef"] = model.ent_coef
    params["clip_range"] = model.clip_range
    
 
    # 4. Architecture (The Brain)
    # The architecture is nested inside 'policy_kwargs'
    if model.policy_kwargs and 'net_arch' in model.policy_kwargs:
        #print(f"Network Arch:       {model.policy_kwargs['net_arch']}")
        params["net_arch"] = model.policy_kwargs['net_arch']
    else:
        #print("Network Arch:       Default [64, 64]")
        params["net_arch"] = [64, 64]
        
    #print("=" * 30)

    return params



# ==========================#########################################
### PPO Model

def train_a2c(
                train_df, 
                asset_price_cols, 
                feature_cols_per_asset, 
                total_timesteps=200_000,
                n_envs=1,
                eval_verbose=0,
                seed = 123,
                model_save_path="./a2c_model",
                **ppo_kwargs
             ):
    
    #model_params = {}
    """
    Trains a single PPO agent sequentially on all tickers in train_data_dict.
    The agent is updated with each ticker's data in sequence.

    Args:
        train_data_dict (dict): Dict of {ticker: df} for training.
        asset_price_cols (list): Asset price columns per asset.
        feature_cols_per_asset (list): Feature columns per asset.
        total_timesteps (int): Timesteps per ticker.
        n_envs (int): Number of parallel envs (usually 1 for sequential).

    Returns:
        model (PPO): Trained PPO model (after all tickers).
        vec_env (VecNormalize): VecNormalize wrapper for last ticker.
    """

    #policy_kwargs = dict(
    #    net_arch=dict(pi=[64,64], vf=[64,64])
    #)

    kwargs = dict(
                policy="MlpPolicy",
                learning_rate=1e-4,
                n_steps=1,
                gamma=0.97,    
                device='cpu',
                verbose=0,
                gae_lambda=0.95,
                ent_coef='0.01',
                vf_coef =0.1,
                rms_prop_eps=1e-5,
                normalize_advantage=True,
                max_grad_norm =0.3,
                policy_kwargs = dict(net_arch=[64,64]),
                tensorboard_log="./a2c_tensorboard/"
                )

    kwargs.update(ppo_kwargs)

    #print(", ".join(f"{k}: {v}" for k, v in kwargs.items()))

    feature_cols_per_asset = [col for col in feature_cols_per_asset if col != 'close']


    # set random seed and pass on to the DQN A
    set_random_seed(seed)

    env_fns = [make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    #checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=f"./checkpoints/", name_prefix=f"dqn_model")
    stop_train_callback = StopTrainingOnNoModelImprovement(
                            max_no_improvement_evals=5,
                            #min_evals=400,
                            verbose = eval_verbose)

    eval_env = DummyVecEnv([make_env_fn(train_df, asset_price_cols, feature_cols_per_asset, action_space_type="Discrete", seed=seed)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_save_path}",
                                 eval_freq=100_000,
                                 callback_after_eval=stop_train_callback,
                                 verbose=eval_verbose)

    model = A2C(
        env=vec_env,
        seed = seed,
        **kwargs,
        #policy_kwargs=policy_kwargs,
        #ent_coef='auto',
        #clip_range=0.2,

    )

    #ppo params work for a2c as well.
    model_params_dict = ppo_model_params(model)

    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)
    vec_env.save(f"{model_save_path}/a2c_vec_normalize.pkl")
    model.save(f"{model_save_path}/a2c_multiticker")


    return model, vec_env, model_params_dict



def ppo_model_params(model):
    #create empty dict
    params = {}
 
    # 1. Basic Hyperparameters
    params["learning_rate"] = model.learning_rate
    params["gamma"] = model.gamma

    params["n_steps"] = model.n_steps
    params["ent_coef"] = model.ent_coef
    params["clip_range"] = model.clip_range
    params["batch_size"] = model.batch_size
    params["gae_lambda"] = model.gae_lambda
    
 
    # 4. Architecture (The Brain)
    # The architecture is nested inside 'policy_kwargs'
    if model.policy_kwargs and 'net_arch' in model.policy_kwargs:
        #print(f"Network Arch:       {model.policy_kwargs['net_arch']}")
        params["net_arch"] = model.policy_kwargs['net_arch']
    else:
        #print("Network Arch:       Default [64, 64]")
        params["net_arch"] = [64, 64]
        
    #print("=" * 30)

    return params
