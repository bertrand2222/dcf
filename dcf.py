import os
import sys
from string import ascii_uppercase
import json
from datetime import date
import subprocess
import numpy as np
from tabulate import tabulate
from scipy.optimize import minimize_scalar
import pandas as pd
from colorama import Fore
from dateutil.relativedelta import relativedelta
import requests
from lxml import html
import urllib3
import yahooquery as yq
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
urllib3.disable_warnings()

# import pandas as pd
SCOPES =  ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

PKL_PATH = "df_save.pkl"
IS = 0.25
NB_YEAR_DCF = 10
HISTORY_TIME = 5 
YEAR_G = 0
INCOME_INFOS = [ 'FreeCashFlow','TotalRevenue', "NetIncome",  ]
BALANCE_INFOS = ['currencyCode', 'TotalDebt', 'CashAndCashEquivalents',
                 'CommonStockEquity', "InvestedCapital", "ShareIssued", "MarketCap"]
TYPES = INCOME_INFOS + BALANCE_INFOS
MARKET_CURRENCY = "USD"

BROWSER_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0'}

### get fed fund rate (debt cost)
URL_FED_FUND_RATE = "https://ycharts.com/indicators/effective_federal_funds_rate"
XPATH_FED_FUND_RATE = "/html/body/main/div/div[4]/div/div/div/div/div[2]/div[1]/div[3]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]"


SHARE_PROFILE_FILE = 'share_profile.json'


class MarketInfos():
    """
    Global market informations
    """
    def __init__(self) -> None:

        r = requests.get(URL_FED_FUND_RATE, verify= False, headers= BROWSER_HEADERS, timeout= 20)
        text_fed_fund_rate = html.fromstring(r.content).xpath(XPATH_FED_FUND_RATE)[0].text

        ### get debt cost
        self.debt_cost = float(text_fed_fund_rate.strip("%")) / 100
        print(f"debt cost = {self.debt_cost*100}%")

        ### get free risk rate
        #us treasury ten years yield
        self.free_risk_rate = yq.Ticker("^TNX").history(period = '1y', ).loc["^TNX"]["close"].iloc[-1]/100
        print(f"free risk rate = {self.free_risk_rate*100:.2f}%")

        ### eval market rate
        sptr_6y = yq.Ticker("^SP500TR").history(period = '6y', interval= "1mo").loc["^SP500TR"]
        sptr_6y_rm = sptr_6y.rolling(2).mean()
        # sptr = yq.Ticker("^SP500TR").history(period = '5y', interval= "1mo").loc["^SP500TR"]

        last_date = sptr_6y.index[-2]
        time_delta = HISTORY_TIME
        past_date = last_date - relativedelta(years = time_delta)
        sptr = sptr_6y.loc[past_date:]
        sptr_rm = sptr_6y_rm.loc[past_date:]

        # delta = relativedelta(last_date, past_date)
        # timeDelta = delta.years + delta.months / 12 + delta.days / 365
        self.market_rate =  (sptr_rm.loc[last_date]['close'] / sptr_rm.loc[past_date]['close'])**(1/time_delta) - 1 # S&P 500 mean year rate over 5 years
        print(f"market rate = {self.market_rate*100:.2f}%")

        self.month_change_rate = sptr["adjclose"][:-1].pct_change(periods = 1).rename('rm')
        self.var_rm = self.month_change_rate.var()

        # rate history
        self.rate_history_dic = {}
        self.rate_current_dic = {}

        # share profile
    def update_rate_dic(self, currency_1, currency_2):
        
        if currency_1 == currency_2 : 
            return
        change_rate = currency_1 + currency_2 + "=X"
        if change_rate not in self.rate_history_dic :
            currency_history = yq.Ticker(change_rate).history(period= '5y',
                                                                interval= "1mo").loc[change_rate]
            self.rate_history_dic[change_rate] = currency_history["close"].iloc[:-1]
            self.rate_current_dic[change_rate] = currency_history["close"].iloc[-1]

            

class Share():
    """
    Object containing a share and its financial informations
    """
    def __init__(self, isin : str, source : str = "yahoo",
                 market_infos : MarketInfos = None, shares_profile : dict = None):
        self.tk = None
        self.isin = isin
        self.history = None
        self.symbol = None
        self.beta = None
        self.short_name = None
        if source == "yahoo" :
            self.symbol = isin
        self.source = source
        self.fcf : float = None
        self.cmpc = None
        self.mean_g_fcf = None
        self.net_debt = None
        self.y_financial_data = None
        self.short_name = None
        self.currentprice = None
        self.financial_currency = None
        self.price_to_fcf = None
        self.per = np.nan
        self.debt_to_equity = np.nan
        self.price_to_book = np.nan
        self.mean_g_tr = None
        self.mean_g_netinc = None
        self.capital_cost = None
        self.roic = np.nan
        self.market_infos = market_infos
        self.shares_profile = shares_profile
        self.unknown_last_fcf = None
        self.price_currency = None
        self.q_financial_data = None
        self.nb_shares = None
        self.market_cap = None
        self.net_market_cap = None
        self.g = None
        self.financial_currency_history = None

    def eval_beta(self) :
        """
        Compute the share beta value regarding the evolution of share price with reference market
        """
        regular_history = self.history['adjclose'].iloc[:-1].copy()

        if self.price_currency != MARKET_CURRENCY :
            self.market_infos.update_rate_dic(self.price_currency, MARKET_CURRENCY)
            change_rate = self.price_currency + MARKET_CURRENCY + "=X"
            regular_history = regular_history * self.market_infos.rate_history_dic[change_rate]

        month_change_share = regular_history.pct_change().rename("share")

        cov_df = pd.concat([self.market_infos.month_change_rate,
                            month_change_share], axis = 1, join= 'inner',).dropna(how = 'any')
        cov = cov_df.cov()['rm'].loc['share']
        beta = cov/ self.market_infos.var_rm
        self.beta = beta

        return beta
    
    def querry_price_from_web(self):
        """
        retrieve price info from web interface if api can not
        """
        if not self.symbol in self.shares_profile :

            url_yahoo = "https://finance.yahoo.com/quote/{0}?p={0}".format(self.symbol)

            short_name_xpath = '//*[@id="quote-header-info"]/div[2]/div[1]/div[1]/h1'
            currency_xpath = '//*[@id="quote-header-info"]/div[2]/div[1]/div[2]/span'
            r = requests.get(url_yahoo, verify= False, headers= BROWSER_HEADERS, )
            self.short_name = html.fromstring(r.content).xpath(short_name_xpath)[0].text
            self.price_currency = html.fromstring(r.content).xpath(currency_xpath)[0].text.split()[-1].strip(')')
            self.shares_profile[self.symbol] = { "short_name" : self.short_name , "price_currency" : self.price_currency }
        else :
            self.short_name = self.shares_profile[self.symbol]["short_name"]
            self.price_currency = self.shares_profile[self.symbol]["price_currency"]

    def querry_financial_info(self, pr = False, ):
        """
        Get the share associated financial infos from yahoo finance api 
        """
        tk = yq.Ticker(self.symbol)
        self.tk = tk
        print(f'\rquerry {self.symbol}    ', flush=True, end="")

        try :
            self.history = self.tk.history(period = '5y', interval= "1mo").loc[self.symbol][['close', 'adjclose']]
        except KeyError:
            print(f"unknown share symbol {self.symbol}")
            return(1)
        self.currentprice = self.history['adjclose'].iloc[-1]

        price = tk.price[self.symbol]
        try :
            self.price_currency = price['currency']
            self.short_name = price["shortName"]
            self.market_cap = price['marketCap']

        except TypeError :
            print(f"price data not availlable from api for {self.symbol}")
            self.querry_price_from_web()


        # print(self.tk.get_financial_data(TYPES,))
        self.y_financial_data : pd.DataFrame = tk.get_financial_data(TYPES,)
        # print(self.y_financial_data)
        # print(tk.valuation_measures)
        # stop
        self.unknown_last_fcf = np.isnan(self.y_financial_data["FreeCashFlow"].iloc[-1])
        self.y_financial_data = self.y_financial_data.ffill(axis = 0
                                                            ).drop_duplicates(
                                                                subset = 'asOfDate',
                                                                keep= 'last').set_index('asOfDate')
        self.q_financial_data = tk.get_financial_data(TYPES , frequency= "q",
                                                      trailing= False).set_index('asOfDate')

        if isinstance(self.q_financial_data, pd.DataFrame) :
            self.q_financial_data.ffill(axis = 0, inplace = True)
            last_date_y = self.y_financial_data.index[-1]
            last_date_q = self.q_financial_data.index[-1]
            for d in BALANCE_INFOS:
                if d in self.q_financial_data.columns :
                    self.y_financial_data.loc[last_date_y , d] = \
                        self.q_financial_data.loc[last_date_q , d]

        self.nb_shares = self.y_financial_data["ShareIssued"].iloc[-1]
        if self.market_cap is None :
            self.market_cap = self.nb_shares * self.currentprice

        self.financial_currency = self.y_financial_data['currencyCode'].iloc[-1]

        self.eval_beta()

        self.financial_currency_history = self.history[['close']].iloc[:-1].copy()
        if self.price_currency != self.financial_currency:
            self.market_infos.update_rate_dic(self.price_currency, self.financial_currency)
            rate_symb = self.price_currency + self.financial_currency + "=X"
   
            rate = self.market_infos.rate_current_dic[rate_symb]
            self.currentprice *= rate
            self.market_cap *= rate
            self.financial_currency_history = self.financial_currency_history.mul(self.market_infos.rate_history_dic[rate_symb],axis = 0)
            

        if self.compute_financial_info(pr = pr) :
            return 1
        return 0

    def compute_financial_info(self, pr = True) :
        """
        Compute intermedate financial infos used as input for dcf calculation
        """
        free_risk_rate = self.market_infos.free_risk_rate
        market_rate = self.market_infos.market_rate
        y_financial_data = self.y_financial_data
        q_financial_data =  self.q_financial_data

        last_financial_info =  y_financial_data.iloc[-1]

        # self.nb_shares = int(self.market_cap / self.currentprice)
        stock_equity = last_financial_info['CommonStockEquity']
        market_cap = self.market_cap
        self.capital_cost = free_risk_rate + self.beta * (market_rate - free_risk_rate)
        try :
            total_debt = last_financial_info['TotalDebt']
        except KeyError:
            print(f"no total debt available for {self.short_name}")
            return 1
        self.cmpc = self.capital_cost * stock_equity/(total_debt + stock_equity) + \
                    self.market_infos.debt_cost * (1-IS) * total_debt/(total_debt + stock_equity)
        self.net_debt = total_debt - last_financial_info['CashAndCashEquivalents']


        self.net_market_cap = market_cap + self.net_debt
        if stock_equity >= 0 :
            self.debt_to_equity = total_debt / stock_equity
            self.price_to_book = market_cap / stock_equity

        y_financial_data['date'] = y_financial_data.index
        y_financial_data['year'] = y_financial_data['date'].apply(lambda  x : x.year )
        y_financial_data.drop_duplicates(subset= "year", keep= "last", inplace= True)

        # net_income = y_financial_data['NetIncome'][-3:].mean()
        net_income = y_financial_data['NetIncome'][-1]
        last_delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[-2],)

        self.per = market_cap / net_income

        invested_capital = last_financial_info['InvestedCapital']
        if invested_capital >= 0 :
            self.roic = net_income / invested_capital

        if last_delta_t.years < 1 and \
            isinstance(q_financial_data, pd.DataFrame) and \
            ('FreeCashFlow' in q_financial_data.columns) and \
            (self.q_financial_data.index[-1] > y_financial_data.index[-2]) :

            complement_q_financial_infos = self.q_financial_data[self.q_financial_data.index
                                                                 > y_financial_data.index[-2]]
            complement_time = relativedelta(complement_q_financial_infos.index[-1],
                                            y_financial_data.index[-2]).months / 12
            self.fcf = (y_financial_data['FreeCashFlow'][-4:-1].sum() \
                        + complement_q_financial_infos['FreeCashFlow'].sum()
                        ) / (3 + complement_time)

        # elif y_financial_data['FreeCashFlow'].iloc[-1] == y_financial_data['FreeCashFlow'].iloc[-2] :
        elif self.unknown_last_fcf :
            print(f"unknown last free cash flow  for {self.short_name}")
            self.fcf = y_financial_data['FreeCashFlow'][-4:-1].mean()

        else :
            self.fcf = y_financial_data['FreeCashFlow'][-3:].mean()

        # if self.fcf < 0 :
        #     self.fcf = y_financial_infos['FreeCashFlow'][-1]

        if self.fcf < 0 :
            self.fcf = y_financial_data['FreeCashFlow'].mean()

        ### price to fcf multiple
        df_multiple = y_financial_data[['ShareIssued', 'FreeCashFlow' ]].copy()
        df_multiple.index = df_multiple.index.date
        df_multiple = pd.concat([self.financial_currency_history, df_multiple], axis = 1).sort_index().ffill().dropna()
        df_multiple['price_to_fcf'] = df_multiple['ShareIssued'] * df_multiple['close'] / df_multiple['FreeCashFlow'] 
        self.price_to_fcf = df_multiple.loc[df_multiple['FreeCashFlow'] > 0 , 'price_to_fcf'].mean()
        # if self.price_to_fcf < 0 :
        #     self.price_to_fcf = df_multiple['price_to_fcf'].max()
        
        if self.price_to_fcf < 0 :
            print(f"negative price_to_fcf mean for {self.short_name} can not compute DCF")
            return(1)

        ### calculation of fcf growth
        delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[YEAR_G],)
        nb_years_fcf = delta_t.years + delta_t.months / 12

        fcf_se_ratio = y_financial_data['FreeCashFlow'].iloc[-1]\
            /y_financial_data['FreeCashFlow'].iloc[YEAR_G]
        
        if fcf_se_ratio < 0:
            self.mean_g_fcf = np.nan
        else :
            self.mean_g_fcf = (fcf_se_ratio)**(1/nb_years_fcf) - 1


        nb_year_inc = delta_t.years + delta_t.months / 12
        self.mean_g_tr = (y_financial_data['TotalRevenue'].iloc[-1]\
                          /y_financial_data['TotalRevenue'].iloc[YEAR_G])**(1/nb_year_inc) - 1
        # inc_se_ratio = y_financial_data['NetIncome'].iloc[-1]\
        #                 /y_financial_data['NetIncome'].iloc[YEAR_G]
        # if inc_se_ratio < 0 :
        #     self.mean_g_netinc = np.nan
        # else :
        #     self.mean_g_netinc = inc_se_ratio**(1/nb_year_inc) - 1

        if pr :
            print("\r")
            print(y_financial_data)
            print(f"Prix courant: {self.currentprice:.2f} {self.financial_currency:s}" )
            print(f"Cout moyen pondere du capital: {self.cmpc*100:.2f}%")
            print(f"Croissance moyenne du chiffre d'affaire sur {nb_year_inc:f} \
                  ans: {self.mean_g_tr*100:.2f}%")
            print(f"Croissance moyenne du free cash flow sur {nb_year_inc:f} \
                  ans: {self.mean_g_fcf*100:.2f}%")
        return(0)

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        """
        Get price corresponding to given growth rate with discouted
        cash flow methods
        """
        if self.cmpc is None :
            self.querry_financial_info(pr = pr)
        if g is None :
            g = self.mean_g_tr
        if start_fcf is None :
            start_fcf = self.fcf

        eval_dcf_(g, self, start_fcf, pr)

    def eval_g(self, start_fcf : float = None, pr=False, use_multiple = True):

        """
        Evaluate company assumed growth rate from fundamental financial data
        """

        if self.cmpc is None :
            if self.querry_financial_info(pr = pr) :
                return np.nan
        if self.cmpc < 0:
            print(f"negative cmpc for {self.short_name} can not compute DCF")
            return np.nan
        if not start_fcf is None :
            self.fcf = start_fcf
        if self.fcf < 0 :
            print(f"negative free cash flow mean for {self.short_name} can not compute DCF")
            return np.nan
        if use_multiple:
            up_bound = 2
        else : 
            up_bound = self.cmpc
        res = minimize_scalar(eval_dcf_, args=(self, False),
                              method= 'bounded', bounds = (-1, up_bound))
        g = res.x

        self.g = g
        if pr:
            print(f"Croissance correspondant au prix courrant: {g*100:.2f}%")
            self.get_dcf(g, start_fcf= start_fcf, pr = pr)

        return g

    def eval_dcf(self, g, pr = False, use_multiple = True):
        """
        compute company value regarding its actuated free cash flows and compare it 
        to the market value of the company
        return : 0 when the growth rate g correspond to the one assumed by the market price.
        """
        cmpc = self.cmpc
        net_debt = self.net_debt
        if isinstance(g, (list, np.ndarray)):
            g = g[0]

        if use_multiple :
            vt = self.fcf * (1+g)**(NB_YEAR_DCF ) * self.price_to_fcf
        else :
            vt = self.fcf * (1+g)**(NB_YEAR_DCF ) / (cmpc - g)
        vt_act = vt / (1+cmpc)**(NB_YEAR_DCF)

        a = (1+g)/(1+cmpc)
        # fcf * sum of a**k for k from 1 to NB_YEAR_DCF 
        fcf_act_sum = self.fcf * ((a**NB_YEAR_DCF - 1)/(a-1) - 1 + a**(NB_YEAR_DCF))
        enterprise_value = fcf_act_sum + vt_act
     

        if pr :
            fcf = np.array([self.fcf * (1+g)**(k) for k in range(1,1 + NB_YEAR_DCF)])
            act_vec = np.array([1/((1+cmpc)**k) for k in range(1,1 + NB_YEAR_DCF)])
            fcf_act = fcf * act_vec
            print("\r")
            val_share = (enterprise_value - net_debt)/ self.nb_shares
            nyear_disp = min(10,NB_YEAR_DCF)
            annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
            table = np.array([ np.concatenate((fcf[:nyear_disp] ,[vt])),
                              np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
            print(f"Prévision pour une croissance de {g*100:.2f}% :")
            print(tabulate(table, floatfmt= ".4e",
                           showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"],
                           headers= annees))

            print(f"Valeur DCF de l'action: {val_share:.2f} {self.price_currency:s}")

        # return ((enterpriseValue - netDebt)/ share.marketCap - 1)**2
        return (enterprise_value / self.net_market_cap - 1)**2

def eval_dcf_(g, *data):
    """
    reformated Share.eval_dcf() function for compatibility with minimize_scalar
    """
    share : Share = data[0]
    pr = data[1]
    # share, pr = data
    return share.eval_dcf(g, pr)


class DCFAnal():
    """
    object containing a reverse dcf analysis and all its context
    """
    def __init__(self, symbol_list : list[str] = None) -> None:

        self.df = None
        self.market_infos = MarketInfos()
        self.symbol_list = symbol_list

        if os.path.isfile(SHARE_PROFILE_FILE) :
            with open(SHARE_PROFILE_FILE, "r", encoding="utf-8") as file :
                self.shares_profile = json.load(file)
                file.close()
        else :
            self.shares_profile = {}

        self.share_list = [Share(sym, market_infos= self.market_infos,
                                 shares_profile= self.shares_profile) for sym in symbol_list]

    def resume_list(self, use_multiple = True) :

        """
        Generate an analysis summary dataframe
        """

        share_list = self.share_list
        g_l = [s.eval_g(use_multiple= use_multiple) for s in share_list]

        with open(SHARE_PROFILE_FILE, "w", encoding="utf-8") as outfile :
            json.dump(self.shares_profile, outfile, indent = 4)
            outfile.close()
        df = pd.DataFrame(index= self.symbol_list,
                          data= {'short_name' : [s.short_name for s in share_list] ,
                                    'current_price' : [s.currentprice for s in share_list],
                                    'currency' : [s.financial_currency for s in share_list],
                                    'beta' : [s.beta for s in share_list],
                                    'price_to_fcf' : [s.price_to_fcf for s in share_list],
                                    'capital_cost' :[s.capital_cost for s in share_list],
                                    'cmpc' :[s.cmpc for s in share_list],
                                    'assumed_g' : g_l ,  
                                    'per' :  [s.per for s in share_list ],
                                    'roic' : [s.roic for s in share_list], 
                                    'debt_to_equity' : [s.debt_to_equity for s in share_list],
                                    'price_to_book' : [s.price_to_book for s in share_list] ,
                                    # 'mean_g_fcf': [s.mean_g_fcf for s in share_list] ,
                                    # 'mean_g_tr' : [s.mean_g_tr for s in share_list],
                                    # 'mean_g_inc' : [s.mean_g_netinc for s in share_list]
                                    })


        # df["diff_g"] = df['mean_g_tr'] - df['assumed_g']
        df.sort_values(by = ['assumed_g', 'debt_to_equity']  , inplace= True, ascending= True)

        self.df = df
        df.to_pickle(PKL_PATH)

    def load_df(self):
        """
        Loads previously saved analysis dataframe
        """
        self.df = pd.read_pickle(PKL_PATH)
        self.share_list = self.df.index

    def to_excel(self, xl_outfile : str = None):

        """
        Export in Excel format the analysis dataframe 
        
        """

        letters = list(ascii_uppercase)
        # self.xl_outfile = xl_outfile
        writer = pd.ExcelWriter(xl_outfile,  engine="xlsxwriter")
        df = self.df
        col_letter = {c : letters[i+1] for i, c in enumerate(df.columns)}
        df.to_excel(writer, sheet_name= "rdcf")
        wb = writer.book
        number = wb.add_format({'num_format': '0.00'})
        percent = wb.add_format({'num_format': '0.00%'})
        # Add a format. Light red fill with dark red text.
        # format1 = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
        # Add a format. Green fill with dark green text.
        # format2 = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        # Add a format. Light red fill .
        format3 = wb.add_format({"bg_color": "#F8696B",})

        worksheet = writer.sheets['rdcf']
        worksheet.add_table(0,0,len(df.index),len(df.columns) ,
                            {"columns" : [{'header' : 'symbol'}]
                                + [{'header' : col} for col in df.columns],
                            'style' : 'Table Style Light 8'})
        worksheet.set_column('B:B', 30, )
        worksheet.set_column(
            f"{col_letter['current_price']}:{col_letter['current_price']}", 13, number)
        worksheet.set_column(
            f"{col_letter['beta']}:{col_letter['price_to_fcf']}", 8, number)
        worksheet.set_column(f"{col_letter['capital_cost']}:{col_letter['assumed_g']}", 13, percent)
        worksheet.set_column(f"{col_letter['per']}:{col_letter['price_to_book']}", 13, number)
        worksheet.set_column(f"{col_letter['roic']}:{col_letter['roic']}", 13, percent )
        # worksheet.set_column(f"{col_letter['mean_g_fcf']}:{col_letter['diff_g']}", 13, percent )

        # format assumed g
        worksheet.conditional_format(
            f"{col_letter['assumed_g']}2:{col_letter['assumed_g']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num',
            'max_type': 'max', 'mid_type' : 'percentile',
            'min_value' : -0.2, 'mid_value' : 50,  
            'min_color' : '#63BE7B', "max_color" : '#F8696B', 
            "mid_color" : "#FFFFFF"})
        # format debt ratio

        worksheet.conditional_format(
            f"{col_letter['debt_to_equity']}2:{col_letter['debt_to_equity']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num',
                'max_type': 'num', 'mid_type' : 'num',
                'min_value' : 0, 'mid_value' : 1, "max_value" : 2, 
                'min_color' : '#63BE7B', "max_color" : '#F8696B', 
                "mid_color" : "#FFFFFF"})
        # format PER
        worksheet.conditional_format(
            f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}",
            {"type": "cell", "criteria": "<", "value": 0, "format": format3})
        worksheet.conditional_format(
            f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}",
            {"type": "3_color_scale", 'min_type': 'num','max_type': 'num',
                'mid_type' : 'percentile',
                'min_value' : 3, 'mid_value' : 50, "max_value" : 50,
                'min_color' : '#63BE7B', "max_color" : '#F8696B', 
                "mid_color" : "#FFFFFF"})
        # format ROIC
        worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}",
                                     {"type": "cell", "criteria": "<",
                                      "value": 0, "format": format3})
        worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}",
                                    {"type": "3_color_scale", 'min_type': 'num','max_type': 'num',
                                     'mid_type' : 'percentile',
                                    'min_value' : 0, 'mid_value' : 50, "max_value" : 0.15,
                                    "min_color" : '#F8696B', 'max_color' : '#63BE7B' ,
                                    "mid_color" : "#FFFFFF"})

        # format Price to Book
        worksheet.conditional_format(
            f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}",
                                     {"type": "cell", "criteria": "<", 
                                      "value": 0, "format": format3})
        worksheet.conditional_format(
            f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}",
                                    {"type": "3_color_scale", 'min_type': 'num',
                                     'max_type': 'num', 'mid_type' : 'percentile',
                                    'min_value' : 1, 'mid_value' : 50, "max_value" : 10, 
                                    'min_color' : '#63BE7B', "max_color" : '#F8696B',
                                    "mid_color" : "#FFFFFF"})
        
        #
        # worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format1})
        # worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": ">", "value": 0, "format": format2})
        writer.close()

        if sys.platform == "linux" :
            subprocess.call(["open", xl_outfile])
        else :
            os.startfile(xl_outfile)

        # valss = df.values.tolist()
        # for vals in valss:
        #     vals[1] = f"{vals[1]:.2f}"
        #     for i in range(6,len(vals)):
        #         if isinstance(vals[i], str) :
        #             continue
        #         vals[i] = f"{Fore.GREEN}{vals[i]:.2%}{Fore.RESET}" if vals[i] >=0 else f"{Fore.RED}{vals[i]:.2%}{Fore.RESET}"

        # table = tabulate(valss, headers= df.columns, showindex= list(df.index), stralign= "right") 
        # print("\r")
        # print(table)

def upload_file(outfile):
    """
    Insert new file.
    Returns : Id's of the file uploaded

    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w", encoding= "utf-8") as token:
            token.write(creds.to_json())

    try:
        # create drive api client
        service = build("drive", "v3", credentials=creds)

        name = "rdcf_"+ str(date.today())
        file_metadata = {"name": name}
        media = MediaFileUpload(outfile)
        # pylint: disable=maybe-no-member

        print(f"upload {name}.xlsx to google drive")
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        print(f'File ID: {file.get("id")}')

    except HttpError as error:
        print(f"An error occurred: {error}")
        file = None

    return file.get("id")
