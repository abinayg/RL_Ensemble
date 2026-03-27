"""
Vectorized multi-asset Gym trading environment.

Key behavior:
- Observations: concatenated technical indicators for all assets (shape: (n_assets * n_features,))
- Actions: continuous per-asset exposure fraction in [-1, 1], shape (n_assets,)
    * Each action element = desired exposure fraction of TOTAL portfolio value for that asset.
    * Example: action[i] = 0.2 => want 20% of portfolio value exposed to asset i (long)
- Cash reserve rule:
    * By default at least `reserve_default` fraction of portfolio value must remain as cash (unused).
      Default reserve_default = 0.10 (10%).
    * If the agent is confident, reserve may be lowered to `reserve_min` (default 0.05).
    * "Confidence" is determined using a configurable function. Default: confidence if
      mean absolute action magnitude across assets >= confidence_threshold (configurable).
      You can supply a custom confidence function via env parameter `confidence_fn`.

TO-DO: Don't use the desired exposures. Only two action outputs per each asset.
action array:
index0: Buy confidence level - > 0 to 1 
index1: Hold confidence level -> 0 to 1
index2: Sell confidence level -> 0 to 1

- When desired exposures would breach the allowed investable portion (1 - reserve),
  exposures are scaled down proportionally so the total *long exposure* fits the available capital.
  (Short exposures are allowed subject to `max_short_exposure` but do not consume cash in this simplified margin-free model.)
- Reward:
    * reward = (portfolio_value_after - portfolio_value_before) - transaction_costs - slippage_costs - exposure_change_penalty
    * transaction_costs = commission_rate * trade_dollar_volume (per asset)
    * slippage_costs = slippage_rate * abs(trade_dollar_volume) (per asset)
    * exposure_change_penalty = lambda_freq * sum(abs(exposure_change))  # encourages fewer/ smaller changes
- portfolio_value() returns current cash + sum(shares * current_prices)

Notes / Simplifications:
- This env does NOT model margin requirements for short positions — shorting is supported by allowing negative exposures,
  but in practice you'd want to model margin/collateral if you use large short exposures.
- The rule for scaling exposures only considers *positive long exposures* consuming cash (so total long exposure <= investable_fraction).
  Negative exposures (shorts) are allowed up to `max_short_exposure`.
- Observations are normalized on env init (mean/std) by default. You may want to compute normalization on a train split instead.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from typing import List, Callable, Dict, Any, Optional
import matplotlib.pyplot as plt


class MultiAssetTradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        seed: int,    #seed is a required argument
        df: pd.DataFrame,                             #dataFrame containing the features of the assets
        asset_price_cols: List[str],                  #'Close' price for the assets
        feature_cols_per_asset: List[List[str]],      
        episode_length: int = 15,
        initial_cash: float = 1_000_000.0,
        reserve_default: float = 0.10,
        reserve_min: float = 0.05,
        confidence_threshold: float = 0.6,
        confidence_fn: Optional[Callable[[np.ndarray], bool]] = None,
        commission_rate: float = 0.0005,
        slippage_rate: float = 0.0005,
        exposure_change_penalty: float = 0.2,
        max_short_exposure: float = 1.0,
        normalize_obs: bool = True,
        verbose: bool = False,

        #hold options reward. TO-DO: change the variable names
        trade_on_sideways_market: bool = False,
        sideways_trade_threshold_perc: int = 1,
        #penalty_for_sideways_market: int = 50,
        reward_for_hold_in_sideways_market: float = 6,

        buy_freq_penalty : float = 1,
        penalize_freq_buy_action = True,
        num_days_for_buy_penalty = 2,
        action_space_type = "Box",
        include_transaction_and_slippage_cost = False






    ):
        """
        Parameters:
        - df: pd.DataFrame containing chronological rows for all assets. Each asset's price & indicators must be present.
              Example columns: ["A_close", "A_sma", "B_close", "B_sma", ...] — depends on user.
        - asset_price_cols: list of column names for the asset prices in the same order used for action indexing.
        - feature_cols_per_asset: list (length n_assets) where each element is a list of column names (features) for that asset.
                                   Observations will be concatenated in same order as assets/features.
        - reserve_default: normal cash reserve fraction (e.g., 0.10 => 10% cash).
        - reserve_min: minimum allowable reserve when confident (e.g., 0.05 => 5% cash).
        - confidence_threshold: default threshold for the builtin confidence_fn (mean(abs(actions)) >= threshold).
        - confidence_fn: optional custom function fn(actions: np.ndarray)->bool to decide if agent is confident.
        - commission_rate / slippage_rate: applied proportionally to trade dollar volume per asset.
        - exposure_change_penalty: multiplier for L1 penalty on exposure change (sum over assets).
        - max_short_exposure: limit for absolute short exposure fraction (positive number, e.g., 1.0 allows -1.0 exposures).
        """
        assert len(asset_price_cols) == len(feature_cols_per_asset), "price cols and feature lists must match length"
        self.df = df.reset_index(drop=True).copy()
        self.asset_price_cols = asset_price_cols
        self.feature_cols_per_asset = feature_cols_per_asset
        self.n_assets = len(asset_price_cols)
        self.episode_length = episode_length

        self.action_space_type = action_space_type
        self.seed = seed


        # Build concatenated feature columns for obs
        self.obs_feature_cols = []
        for cols in feature_cols_per_asset:
            self.obs_feature_cols.extend(cols)
        self.n_features_per_asset = [len(cols) for cols in feature_cols_per_asset]
        self.obs_dim = sum(self.n_features_per_asset)

        # Params
        self.initial_cash = float(initial_cash)
        self.reserve_default = float(reserve_default)
        self.reserve_min = float(reserve_min)
        self.commission_rate = float(commission_rate)
        self.slippage_rate = float(slippage_rate)
        self.exposure_change_penalty = float(exposure_change_penalty)
        self.max_short_exposure = float(max_short_exposure)
        self.normalize_obs = normalize_obs
        self.verbose = verbose
        
        self.include_transaction_and_slippage_cost =include_transaction_and_slippage_cost 

        self.trade_on_sideways_market = trade_on_sideways_market
        self.sideways_trade_threshold_perc = sideways_trade_threshold_perc
        #self.penalty_for_sideways_market = penalty_for_sideways_market

        self.reward_for_hold_in_sideways_market = reward_for_hold_in_sideways_market

        self.trade_value = 0.0
        self.max_pv = None


        self.buy_freq_penalty = buy_freq_penalty
        self.penalize_freq_buy_action  = penalize_freq_buy_action
        self.num_days_for_buy_penalty = num_days_for_buy_penalty

        self.step_infos = [] # to store info dicts for each step for plotting

        self.ret_hist = []  # to store portfolio log-return history for volatility calc

        # Confidence function
        if confidence_fn is not None:
            self.confidence_fn = confidence_fn
        else:
            # default: confident if mean absolute exposure >= confidence_threshold
            thresh = float(confidence_threshold)
            self.confidence_fn = lambda actions: np.mean(np.abs(actions)) >= thresh

        # Spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        # For each asset, three action values: [bias_weight of the asset, buy_confidence, hold_confidence, sell_confidence]
        # bias_weight is used to determine the proportion of total cash available(after reservation) to allocate to each asset
        # when taking action, use the np.argmax to determine which action to take for each asset
        if self.action_space_type == "Box":
            self.action_space = spaces.Box(low=0.0001, high=1.0, shape=(self.n_assets, 4), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(3)


        #for DQN, spaces.box doesn't work
        #self.action_space =     spaces.Discrete(4) 
        


        # Bookkeeping
        self.current_start_index = 0
        self.local_step = 0
        self.n_steps = len(self.df)
        self.cash = float(self.initial_cash)
        # positions stored as number of shares per asset (float)
        self.position_shares = np.zeros(self.n_assets, dtype=np.float64)
        # exposures last step (fraction of portfolio value), initialize 0
        self.last_exposures = np.zeros(self.n_assets, dtype=np.float64)
        self.info: Dict[str, Any] = {}

        # Precompute obs normalization
        if self.normalize_obs:
            # compute mean/std across entire df (or you can compute on train split externally)
            obs_df = self.df[self.obs_feature_cols]
            self.obs_mean = obs_df.mean().values.astype(np.float32)
            self.obs_std = obs_df.std().replace(0, 1).values.astype(np.float32)
        else:
            self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32) #remove the "Close" price
            self.obs_std = np.ones(self.obs_dim, dtype=np.float32) #remove the "Close" price

    #def seed(self, seed=None):
    #    self.np_random, seed = gym.utils.seeding.np_random(seed)
    #    np.random.seed(seed)
    #    return [seed] 
        
    def reset(self, start_index: int = 0, seed: Optional[int] = None, return_info: bool = True):
        #if seed is not None:
        #    #np.random.seed(seed)
        #    self.seed(seed)
        #self.current_step = int(start_index)
        self.cash = float(self.initial_cash)
        self.position_shares = np.zeros(self.n_assets, dtype=np.float64)
        self.last_exposures = np.zeros(self.n_assets, dtype=np.float64)
        self.info = {}
        self.max_pv = self.portfolio_value()
        start = random.randrange(0, len(self.df) - self.episode_length)

        # Debug info
        #if self.step_infos:
        #    #print the last trainining episode info
        #    print(f"pv_before: {self.step_infos[-1]["pv_before"]} pv_after: {self.step_infos[-1]["pv_after"]} ")

        #print(f"Starting new episode at index {start}, local_step: {self.local_step}")
        (self.current_start_index, self.local_step) = (start, 0)
        obs = self._get_obs(self.current_start_index)
        self.ret_hist = [ ] # reset return history
        if return_info:
            return obs, self.info
        #print(f"observation space: {self.observation_space} action_space: {self.action_space}")
        return obs

    def _get_prices(self, index) -> np.ndarray:
        """Return current price vector for all assets (length n_assets)."""
        return self.df.loc[index, self.asset_price_cols].values.astype(np.float64)
    
    #def __get_obs_for_reset(self, start_index) -> np.ndarray:
    #    """Return normalized concatenated observation vector for all assets."""
    #    raw = self.df.loc[start_index, self.obs_feature_cols].values.astype(np.float32)
    #    return (raw - self.obs_mean) / self.obs_std

    def _get_obs(self, index) -> np.ndarray:
        """Return normalized concatenated observation vector for all assets."""
        raw = self.df.loc[index, self.obs_feature_cols].values.astype(np.float32)
        return (raw - self.obs_mean) / self.obs_std

    def portfolio_value(self) -> float:
        """Return current portfolio value = cash + sum(shares * current_prices)."""
        prices = self._get_prices(self.current_start_index + self.local_step)
        return float(self.cash + np.dot(self.position_shares, prices))

    # new step function without considering exposure
    def step(self, action: np.ndarray):
        if self.local_step == 0:
            pv_before = self.initial_cash
        else:
            #pv_before = self.portfolio_value()
            pv_before = self.step_infos[-1]['portfolio_value'] #take the previous step portfolio value

        prices = self._get_prices(self.current_start_index + self.local_step)  # price vector at current step

        #action = np.array(action).reshape(self.n_assets, 4) #This needs to be re-shaped for the recurrentPPO usage.
        #print(action.dtype, action.shape)
        if self.action_space_type == "Box": 
          assert self.action_space.contains(action), f"Action {action} outside space"

        # Interpret actions: for each asset, take argmax of [buy, hold, sell]
        #print(f"action shape: {action.shape}")
        if self.action_space_type == "Box":
            actions = np.asarray(action, dtype=np.float32)

        if self.action_space_type == "Box":
            assert actions.shape == (self.n_assets, 4), "Action shape must be (n_assets, 4)"

        #store bias_weights seperately
        #bias_weights = actions[:, 0]
        #TO-DO: For now set all bias weights to 1.0
        bias_weights = np.ones(self.n_assets, dtype=np.float64)

        #argmax returns the index of the maximum value along the specified axis
        # elements in the decisions array are in {0,1,2}
        if self.action_space_type == "Box":
            decisions = np.argmax(actions[:, 1:], axis=1)
        else:   
            decisions = action  #directly assign the value

        confidence_level = np.sum(decisions == 1) / self.n_assets # proportion of buy actions (index1 for buy actio)

        action_info = {
            "bias_weights": bias_weights.copy(),
            "decisions": decisions.copy(),  # 0=buy, 1=hold, 2=sell
        }

        trade_values = np.zeros(self.n_assets, dtype=np.float64)
        commission = np.zeros(self.n_assets, dtype=np.float64)
        slippage_cost = np.zeros(self.n_assets, dtype=np.float64)
        executed_shares = np.zeros(self.n_assets, dtype=np.float64)

        for i in range(self.n_assets):
            (tiny_extra_reward, is_prev_action_buy, penalize_buy_action, penalize_sell_action) = (False, False, False, False)
            current_cash_available = self.cash
            reserve  = self.reserve_min if confidence_level<=0.6 else self.reserve_default

            # if the stock is going sideways (t, and t-1), then there's nothing much to trade. 
            #if not self.trade_on_sideways_market and self.current_step > 1:
            #    if abs(prices[i] - self.df.loc[self.current_step-1, self.asset_price_cols[i]]) / self.df.loc[self.current_step-1, self.asset_price_cols[i]] < (self.sideways_trade_threshold_perc) / 100:
                    #penalize the model for trading in a sideways market. 
                    # appreciate the model for not doing sideways trading

            if self.action_space_type == "Box":
                decisions = decisions[i]

            #if decisions[i] == 0:  # Buy
            if decisions == 0:
                investable_cash = current_cash_available - pv_before * reserve
                #print(f"pv_before {i}: {pv_before} current_cash_available: {current_cash_available} investable_cash: {investable_cash} reserve: {reserve}")
                if investable_cash > 0:   # FOR BUY ONLY

                    #split the cash equally among all the assets
                    #max_affordable_dollars = investable_cash * (  (bias_weights[i] / np.sum(bias_weights)) / self.n_assets )
                    max_affordable_dollars = investable_cash  // self.n_assets 
                    trade_value = max_affordable_dollars  # Buy as much as possible within reserve
                    self.trade_value = trade_value
                    #turnover_dollars = trade_value
                    exec_price = prices[i] * (1.0 + self.slippage_rate)  # Buy price with slippage
                    executed_shares = int(trade_value // exec_price) #make sure this is a integer value
                    self.position_shares[i] += executed_shares
                    self.cash -= trade_value  # Reduce cash by trade value
                    commission = abs(trade_values) * self.commission_rate # abs val is used. Whether buy or sell, commission is always positive
                    slippage_cost = abs(trade_values) * self.slippage_rate
                    #total_commission is calculated at the end. Don't deduct here
                    if self.include_transaction_and_slippage_cost:
                      self.cash -= np.sum(commission)  # Deduct commission from cash



            #elif decisions[i] == 1:  # Sell
            elif decisions == 1:  # Sell
                if self.position_shares[i] == 0:
                    penalize_sell_action = True
                current_position_shares = self.position_shares[i]
                current_position_dollars = current_position_shares * prices[i]
                exec_price = prices[i] * (1.0 - self.slippage_rate)
                # Only sell up to the shares you own
                executed_shares = min(int(current_position_dollars // exec_price), int(current_position_shares))
                self.position_shares[i] -= executed_shares
                trade_value = executed_shares * exec_price
                self.trade_value = trade_value
                #turnover_dollars = trade_value
                self.cash += trade_value
                commission = abs(trade_value) * self.commission_rate
                slippage_cost = abs(trade_value) * self.slippage_rate
                # total_commission is calculated at the end. Don't deduct here
                #self.cash -= np.sum(commission)
                if self.include_transaction_and_slippage_cost:
                  self.cash -= np.sum(commission)  # Deduct commission from cash

            else:
                # Hold action, no trade. 
                trade_value = 0.0
                self.trade_value = trade_value
                #if self.current_step >=2:
                  #if abs(prices[i] - self.df.loc[self.current_step-1, self.asset_price_cols[i]]) / self.df.loc[self.current_step-1, self.asset_price_cols[i]] < (self.sideways_trade_threshold_perc / 100):
                  #   tiny_extra_reward = True 
                  #if self.position_shares[i] > 0:
                  #    tiny_extra_reward = True

            if self.local_step >=2:
              #calculate the benchmark return
              prev_price = self.df.loc[self.local_step-1, self.asset_price_cols[i]]
              curr_price = self.df.loc[self.local_step, self.asset_price_cols[i]]
              benchmark_return = (curr_price - prev_price) / prev_price                    #just by holding on to the asset.
            else:
                benchmark_return = 0.0
            #benchmark_return = 0.0   #for now, set the benchmark return to zero


        # Compute new portfolio value after price movement
        #new_prices = self._get_prices()
        #pv_after = float(self.cash + np.dot(self.position_shares, new_prices)) #if there are any positions left
        pv_after = self.portfolio_value() #current portfolio value after the trades

        self.max_pv = max(self.max_pv, pv_after)
        drawdown = (self.max_pv - pv_after) / self.max_pv 

        stratergy_return = (pv_after - pv_before) / pv_before

        # raw pnl includes mark-to-market and already includes cash changes from execution and commissions
        #raw_pnl = pv_after - pv_before

        # Stable, scale-free log reward

        S = 100.0
        step_ret = np.log(pv_after / pv_before) if pv_before > 0 and pv_after > 0 else -1.00
        self.ret_hist.append(step_ret)      #hold the portfolio log-return history to adjust the rewards
        #return_reward = S * step_ret

        #original reward
        #return_reward = S * step_ret

        #return_reward = S * ((pv_after - pv_before)/pv_before)
        return_reward = S * (stratergy_return - benchmark_return)
        
        # rolling vol (keep a deque of last 20 returns)
        #Compute a short rolling volatility of portfolio log-returns
        # start using the first 5 returns once we have them
        win = 10
        vol = np.std(self.ret_hist[-win:]) if len(self.ret_hist) >= win else 0.005
        #vol_scaled = S * vol

        #volatile penalty        
        lambda_vol = 1.0
        vol_penalty = lambda_vol * vol

        #Drawdown penality
        #lambda_dd = 4.0 
        lambda_dd = 0.2                 #for now, comment out the drawdown penalty
        dd_penalty = lambda_dd * drawdown

        #turnover penalty
        lambda_turnover = 3.0
        turnover_penalty = lambda_turnover * ( (self.trade_value)/ max(pv_before, 1e-6) )

        #reward = return_reward - vol_reward - dd_reward
        ## reward = return_reward - turnover_penalty - dd_penalty - vol_penalty
        reward = return_reward  - dd_penalty

        if tiny_extra_reward:
            reward += 0.001 #give some small reward for not trading in sideways market

        #with reward = return_reward - dd_penalty, dqn is working fine. 

        curr_portfolio_value = self.portfolio_value()

        if self.local_step == self.episode_length - 1 :
            done = True
        elif pv_after < 0.2 * self.initial_cash:  #if the current PV is less than 50% of the initial cash, then end the episode. Helps agent to learn the survival skill
            done = True
        else:
            done = False

        # After trades, advance time to next row (we apply market movement after execution)
        self.local_step += 1
        #done = False


        #Episode end bonus (sharpe like ratio)
        if done:
            rets = np.array(self.ret_hist)
            ep_ret = np.sum(rets)
            ep_vol = np.std(rets) + 1e-8
            sharpe_like = ep_ret/ep_vol

            k= 0.01
            reward+=k*sharpe_like

        #turnover_fraction = self.trade_value / max(pv_before, 1e-9)
        #lambda_turn = 0.02
        #reward-=(lambda_turn * turnover_fraction)
        
        shaping = 0.0
        ####if tiny_extra_reward:
        ####    #shaping += self.reward_for_hold_in_sideways_market * vol_scaled        # ~small nudge
        ####    shaping += self.reward_for_hold_in_sideways_market 
        ####if penalize_buy_action:
        ####    shaping -= self.buy_freq_penalty * vol_scaled        # ~medium
        ####if penalize_sell_action:
        ####    shaping -= self.buy_freq_penalty * vol_scaled        # ~large
        
        # optional turnover penalty
        #shaping -= 0.2 * vol_scaled * turnover_fraction  # turnover in [0,1]
        
        # clamp shaping
        #shaping = float(np.clip(shaping, -0.60, 0.60))  
        #shaping = float(np.clip(shaping, -100.0, 100.0))  
        reward += shaping
        
        #extra reward for not trading in sideways market
        #reward += self.reward_for_hold_in_sideways_market if tiny_extra_reward else 0

        #reward -= self.buy_freq_penalty if penalize_buy_action else 0

        #reward -= self.buy_freq_penalty if penalize_sell_action else 0 #Use the same penalty for the sell action too.



        # Info dict for diagnostics
        info = {
            "step": self.local_step,
            "return reward": return_reward,
            "turnover_penalty": turnover_penalty,
            "dd_penalty": dd_penalty,
            "vol_penalty": vol_penalty,
            "reward": reward,
            #"vol_reward": vol_reward,
            #"dd_reward": dd_reward,
            #"reward": reward,
            #"shaping": shaping,
            "pv_before": float(pv_before),
            "pv_after": float(pv_after),
            #"volatility scaled": float(vol_scaled),
            #"raw_pnl": float(raw_pnl),
            #"commission": commission,
            #"slippage": slippage_cost,
            #"exposure_penalty": exposure_pen,
            #"applied_reserve": reserve,
            #"requested_exposures": desired_exposures.astype(float),
            #"adjusted_exposures": adjusted_exposures.astype(float),
            "position_shares": self.position_shares.copy(),
            "cash": float(self.cash),
            "action_info": action_info,
            "portfolio_value": curr_portfolio_value,
            #"prices": new_prices.copy(),
        }
        #print(f" Step: {self.local_step} | PV before: {pv_before:.2f} | PV after: {pv_after:.2f} | Reward: {reward:.2f} | Cash: {self.cash:.2f} | Positions: {self.position_shares} \n ")
        self.info = info
        #append info for the plot function
        self.step_infos.append(self.info)
        truncated = False
        obs = self._get_obs(self.current_start_index + self.local_step)
        return obs, float(reward), bool(done), truncated,  info



    def render(self, mode="human"):
        pv = self.portfolio_value()
        print(f"Step {self.local_step} | PV {pv} | Cash {self.cash} | Positions: {self.position_shares}\n\n")

    def close(self):
        pass

    def plot(self, asset_index=0):
        """
        Plots the price of the selected asset over time.
        Marks buy/sell actions with scatter points.
        Shows portfolio value and profit/loss percentage on the left.
        
        Args:
            asset_index (int): Index of the asset to plot (default: 0)
        """
        #price_col = self.asset_price_cols[asset_index]
        #prices = self.asset_price_cols['close'].values
        prices = self.df['close'].values
        steps = np.arange(len(prices))

        # Collect buy/sell actions
        buy_steps = []
        sell_steps = []

        # If you have stored actions/decisions for each step, use them.
        # Otherwise, use info dict if available.
        # We'll assume you run the environment and store info at each step in a list: self.step_infos

        # If not present, you can add this in your step function:
        # self.step_infos = [] at reset, and self.step_infos.append(info) at each step.

        if not hasattr(self, "step_infos"):
            print("No step_infos found. Please store info dicts at each step in self.step_infos.")
            return

        for i, info in enumerate(self.step_infos):
            if "action_info" in info:
                decisions = info["action_info"]["decisions"]
                #if decisions[asset_index] == 0:  # Buy
                if decisions == 0:  # Buy
                    buy_steps.append(i)
                #elif decisions[asset_index] == 1:  # Sell
                elif decisions == 1:  # Sell
                    sell_steps.append(i)

        # Portfolio value at last step
        pv = self.portfolio_value()
        pv_initial = self.initial_cash
        profit_pct = ((pv - pv_initial) / pv_initial) * 100

        #extract the portfolio value as a list for plotting. 
        portfolio_val_list = [ item["portfolio_value"] for item in self.step_infos]

        #fig, ax1 = plt.subplots(figsize=(12,6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=False)
        #plt.figure(figsize=(12, 6))
        ax1.plot(steps, prices, label=f"Close Price", color="blue")
        ax1.scatter(buy_steps, prices[buy_steps], marker="^", color="green", label="Buy", s=100)
        ax1.scatter(sell_steps, prices[sell_steps], marker="v", color="red", label="Sell", s=100)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Price")
        ax1.set_title(f"close Price & Trades\nPV: {pv:.2f} | P/L: {profit_pct:.2f}%")
        ax1.legend(loc="upper right")

        #ax2 = ax1.twinx()
        ax2.plot(portfolio_val_list, label=f"Porfolio Value", color="green")
        ax2.set_title(f"Porfolio value and steps")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("INR")
        # Annotate PV and profit/loss on left
        #ax1.gca().text(0.01, 0.95, f"Portfolio Value: {pv:.2f}\nProfit/Loss: {profit_pct:.2f}%", 
        #               transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.show()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Build dummy multi-asset DataFrame for 3 assets with two feature columns each
    T = 200
    dates = pd.date_range("2020-01-01", periods=T)
    # create 3 assets price series
    rng = np.random.default_rng(42)
    close_A = np.cumsum(rng.normal(scale=1.0, size=T)) + 100.0
    close_B = np.cumsum(rng.normal(scale=1.5, size=T)) + 50.0
    close_C = np.cumsum(rng.normal(scale=0.8, size=T)) + 200.0
    df = pd.DataFrame({"date": dates, "A_close": close_A, "B_close": close_B, "C_close": close_C})

    # create 2 simple indicators per asset
    df["A_sma5"] = pd.Series(close_A).rolling(5, min_periods=1).mean()
    df["A_mom3"] = pd.Series(close_A).diff(3).fillna(0.0)
    df["B_sma5"] = pd.Series(close_B).rolling(5, min_periods=1).mean()
    df["B_mom3"] = pd.Series(close_B).diff(3).fillna(0.0)
    df["C_sma5"] = pd.Series(close_C).rolling(5, min_periods=1).mean()
    df["C_mom3"] = pd.Series(close_C).diff(3).fillna(0.0)

    asset_price_cols = ["A_close", "B_close", "C_close"]
    feature_cols_per_asset = [["A_sma5", "A_mom3"], ["B_sma5", "B_mom3"], ["C_sma5", "C_mom3"]]

    env = MultiAssetTradingEnv(
        df=df,
        asset_price_cols=asset_price_cols,
        feature_cols_per_asset=feature_cols_per_asset,
        initial_cash=1_000_000.0,
        reserve_default=0.10,
        reserve_min=0.05,
        confidence_threshold=0.6,
        commission_rate=0.0005,
        slippage_rate=0.0005,
        exposure_change_penalty=0.2,
        max_short_exposure=1.0,
        normalize_obs=True,
        verbose=True,
    )

    obs = env.reset()
    for t in range(5):
        # Example policy: random small exposures
        raw_action = np.random.uniform(0, 1.0, size=(env.n_assets, 4))
        obs, reward, done, truncated, info = env.step(raw_action)
        print(f"Info: {info}\n\n")
        #env.render()
        #print("reward:", reward)
        #print(f"reward: {reward}\n\n")
        #add line break
        #print(f"---------------------------")
        if done:
            break

    print("Final PV:", env.portfolio_value())