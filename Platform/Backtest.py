# Import Packages
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Callable
import datetime as dt
import calendar
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo
pyo.init_notebook_mode(connected = True)
pio.renderers.default = "plotly_mimetype+jupyterlab"
import math
import os


# Main Backtest Class

class Backtest: 
    def __init__(self, data: pd.DataFrame, name:str ):
        # Name and Additional Info Attributes
        self.name = name
        self.data = data
        self.description: str = None
        
        # Rebal Attributes
        self.rebal_obj: Backtest_rebal_obj  = Backtest_rebal_obj(name= name)
        self.rebal_attr: Dict[str, Union[int,str]] = {"fixed_ticks": None, "interval": "month", "day_of_rebal":1} # Default is first of every month

        # Cost Attributes
        self.costs: Dict[str, float] = {"slippage_cost": 0, "execution_cost": 0, "liquidity_cost": 0, "financing_cost": 0} # Default is no costs

        # Calculation Attributes
        self.inputs: List = []
        self.securities: List = []
        self.signal_func: Callable = None
        self.signal_func_params: dict = {}
        self.lookback_window: int = None
        self.investment: int = 100

        # Results Attributes
        self.output: pd.DataFrame = None
        self.stats: pd.DataFrame = None

    def assign_rebal_attr(self, fixed_ticks= None, interval= None, day_of_rebal= None):             
        # Ensure some rebal attribute is assigned
        if fixed_ticks == None:
            if interval == None or day_of_rebal == None:
                raise Exception("You need an Interval and a day_of_rebal")   

        for attr in self.rebal_attr:
            self.rebal_attr[attr] = eval(attr)
            
    def assign_strategy_costs(self, cost_dict: Dict[str,float] = {}):
        for cost in cost_dict:
            self.costs[cost] = cost_dict[cost]

    def assign_signals(self, trading_signal: Callable = None, rebal_signal: Callable = None):
        self.signal_func = trading_signal
        self.rebal_obj.signals = rebal_signal
        
    def plot(self):
        fig = px.line(self.output, x = self.output.index, y = "PNL")
        fig.update_layout(autosize = True)
        fig.show()

    def calculate(self)-> pd.DataFrame:
        # Ensure signal function is assigned
        if self.signal_func == None: 
            raise Exception("Please assign signal function")

        investment: int = self.investment
        test_data: pd.DataFrame = self.data[self.lookback_window:]
        
        holding_col_names: List = [sec + " holdings" for sec in self.securities]
        weights_col_names: List = [sec + " weights" for sec in self.securities]
        
        test_data = test_data.reindex(columns = list(test_data.columns) + ["PNL"]+holding_col_names+weights_col_names+ [ "Costs", "Add_data"])
        test_data["Add_data"] = test_data["Add_data"].astype(object)
        
        for tick in test_data.index: 
            
            # First iteration of test 
            if tick == test_data.index[0]: 

                sig_out = self.signal_func(tick = tick, data = self.data,lookback = self.lookback_window, securities = self.securities, **self.signal_func_params)
                weights = sig_out[0]
                add_data = sig_out[1]
                
                test_data.loc[tick, "PNL"] = investment # Assign initial investment
                test_data.loc[tick, weights_col_names] = weights # Assign initial weights
                
                
                holdings_test = (investment * test_data.loc[tick, weights_col_names].values)/(test_data.loc[tick,self.securities].values) # Assign initial holdings

                test_data.loc[tick, holding_col_names] = holdings_test
                test_data.loc[tick, "Add_data"] = [add_data]
                continue
             
            rebal_obj_data = self.rebal_obj.rebal_check2(data = test_data, current_date = tick, fixed_ticks = self.rebal_attr["fixed_ticks"], interval = self.rebal_attr["interval"], day_of_rebal = self.rebal_attr["day_of_rebal"])
            
            if rebal_obj_data[0]: # If rebal condition is true
                
                sig_out = self.signal_func(tick = tick, data = self.data,lookback = self.lookback_window, securities = self.securities, **self.signal_func_params)
                weights = sig_out[0]
                
                add_data: Dict = sig_out[1]
                add_data["rebal_data"] = rebal_obj_data[1:]

                test_data.loc[tick, weights_col_names] = weights
                test_data.loc[tick, "Add_data"] = [add_data]
                test_data.loc[tick, holding_col_names] = ((test_data.shift(1).loc[tick,["PNL"]].values * test_data.loc[tick,weights_col_names]).values)/(test_data.shift(1).loc[tick, self.securities].values)
            else:

                test_data.loc[tick, weights_col_names] = test_data.shift(1).loc[tick,weights_col_names]
                test_data.loc[tick, holding_col_names] = test_data.shift(1).loc[tick,holding_col_names]
                
            pnl_val = test_data.shift(1).loc[tick, ["PNL"]].values + (test_data.shift(1).loc[tick, holding_col_names].values * (test_data.loc[tick,self.securities] - test_data.shift(1).loc[tick, self.securities]).values).sum() # Pnl pre costs

            # Financing costs in every period
            fin_cost = (test_data.loc[tick,holding_col_names].values * test_data.loc[tick,self.securities].values).sum() *  self.costs["financing_cost"]
            costs_list: List = [abs(fin_cost)]

            if rebal_obj_data[0]: # Calculate rebalancing costs when rebalance occurs (Slippage, Execution, Liquidity)
                turnover = (((test_data.loc[tick,holding_col_names] - test_data.shift(1).loc[tick,holding_col_names]).values) * test_data.loc[tick,self.securities].values).sum()
                rebal_costs = turnover * (self.costs["slippage_cost"] + self.costs["execution_cost"] + self.costs["liquidity_cost"])
                costs_list.append(rebal_costs)

            cost_val = sum(costs_list)
            test_data.loc[tick,["Costs"]] = cost_val
            test_data.loc[tick,"PNL"] = pnl_val - cost_val
            
        self.output = test_data
        self.calc_stats()
        return test_data
            
        
    def calc_stats(self,risk_free_rate: float = 0, mean_method = False):
        pnl = self.output.PNL
        returns: pd.Series = ((pnl.div(pnl.shift(1))-1).dropna())
        excess_returns: pd.Series = returns - risk_free_rate

        if mean_method: # Calculate Sharpe using mean returns
            mean_excess_return: float = float(excess_returns.mean())
            vol: float = float(excess_returns.std())
            sharpe: float = mean_excess_return/vol 
            out = {"Returns": mean_excess_return, "Volatility": vol, "Sharpe": sharpe}
            
        else: # Calculate Sharpe using annualised returns 
            annualised_return: float = (pnl.iloc[-1] / pnl.iloc[0])**(365/len(pnl)) -1
            excess_annualised_return = annualised_return - risk_free_rate
            annualised_vol: float = excess_returns.std()*math.sqrt(252)
            sharpe: float = excess_annualised_return / annualised_vol
            out = {"Returns": excess_annualised_return, "Volatility": annualised_vol, "Sharpe": sharpe}

        stats_df = pd.DataFrame.from_dict(out, orient = "index", columns = ["Results"]).T
        self.stats = stats_df
        
# Helper functions 
def add_data_unpacker(add_data: pd.Series) -> pd.DataFrame: 
    unpack1 = add_data.apply(lambda x: x[0] if isinstance(x,list) else{})
    out = unpack1.apply(pd.Series)
    return out




# Rebalancing Class

class Backtest_rebal_obj: 

    def __init__(self,name, signals = None):

        self.days_to_rebal: Union[int, None]  = None
        self.rebal_duration: Union[int,None] = None
        self.name: str = name
        self.signals: Callable = signals
        self.index: pd.Index = None
        self.last_rebal_date = None
        
    # Rebal Method - 1 [Too restrictive]
    def set_index(self, df: pd.DataFrame): 
        self.index = df.index.unique()

    def assign_rebal_time(self, ticks: int):
        self.rebal_duration = ticks
        self.days_to_rebal = ticks
        
    def decrement_rebal_count(self)-> bool: 
        self.days_to_rebal += -1

    def rebal_check(self, data: pd.DataFrame): 
        if self.last_rebal_date == None:
            self.last_rebal_date: pd.datetime = data.index.sort_values()[0]
        
        # Check signals 
        sig_rebal = False
        time_rebal = False
        
        if self.signals == None: 
            sig_rebal = True
        else: 
            sig_rebal = self.signals(data)
            
        if type(self.rebal_duration) == int: # Check rebal duration != None
            if self.days_to_rebal == 0: 
                self.days_to_rebal = self.rebal_duration
                time_rebal = True 
            else:
                self.decrement_rebal_count() 
        else:
            time_rebal = True

        #print(self.days_to_rebal)
        return (sig_rebal, time_rebal)
        

    # Rebal - Method 2 [currently used in Backtest object]
    def time_rebal_func_low_freq(self, current_date: dt.datetime, fixed_ticks: Union[None,int] = None, interval: str = "month", day_of_rebal: int = 1)->bool: 

        last = self.last_rebal_date

        #Ensure some rebalance method is assigned
        if fixed_ticks == None and day_of_rebal == None: 
            raise Exception("You need fixed_ticks or a day_of_rebal")

        # Calculate minimum rebal date
        if fixed_ticks != None: 
            trading_days = round(fixed_ticks*7/5)
            rebal_min_date = last + dt.timedelta(days = trading_days)
        
        elif interval == "month":
            plus_1_month = last + pd.DateOffset(months = 1)
            day = self.rebal_day(day_of_rebal, plus_1_month.month, plus_1_month.year)
            rebal_min_date = dt.date(plus_1_month.year, plus_1_month.month, day)
    
        elif interval == "quarter":
            current_quarter = (last.month-1)//3 + 1
            next_quarter_year = (last + pd.DateOffset(months = 3)).year
            next_quarter_month = 1 + 3*(current_quarter)%12
            day = self.rebal_day(day_of_rebal, next_quarter_month, next_quarter_year)
            rebal_min_date = dt.date(next_quarter_year, next_quarter_month, day)
            
        elif interval == "year":
            next_year = last.year + 1
            day = self.rebal_day(day_of_rebal, 1, next_year)
            rebal_min_date = dt.date(next_year, 1, day)
            
        else:
            print("error")

        if current_date >= rebal_min_date:
            return True
        else:
            return False

    @staticmethod
    def rebal_day(day: int = None, month: int = None, year: int = None)-> int:
        if day >0: 
            return day
        else:
            no_days_in_month: int = calendar.monthrange(year,month)[1]
            return no_days_in_month + day
        
    
    
    
    def rebal_check2(self, data: pd.DataFrame, current_date, fixed_ticks = None, interval: str = "month", day_of_rebal: int = 1) -> Tuple: # Is there a neater way?

        # Ensure last_rebal_date is initialised 
        if self.last_rebal_date == None or self.last_rebal_date > current_date:
            self.last_rebal_date: pd.datetime = current_date 

                
        # Initialise signals (Time and event-based rebal signals)
        sig_rebal = False
        time_rebal = False
        sig_rebal_data = None

        if self.signals == None: # If there is no event-based rebal function assign flag as True
            sig_rebal = True
        else: # Run fuctional rebal and assign flag
            sig_rebal_data = self.signals(current_date, self.last_rebal_date, data)
            sig_rebal: bool = sig_rebal_data[0] 
        
        time_rebal = self.time_rebal_func_low_freq(current_date= current_date, fixed_ticks= fixed_ticks, interval= interval, day_of_rebal= day_of_rebal)

        rebal = sig_rebal and time_rebal
        
        if rebal:
            self.last_rebal_date = current_date
        
        return (rebal,sig_rebal, time_rebal, sig_rebal_data)


    

