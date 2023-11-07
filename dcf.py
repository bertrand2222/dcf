import yahooquery as yq
import numpy as np
from tabulate import tabulate
from scipy.optimize import minimize_scalar
import pandas as pd
from colorama import Fore
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import os
import requests
from lxml import html
import urllib3
from string import ascii_uppercase
urllib3.disable_warnings()
import json
import subprocess
# import pandas as pd

IS = 0.25
NB_YEAR_DCF = 5
YEAR_G = 0
INCOME_INFOS = [ 'FreeCashFlow','TotalRevenue', "NetIncome",  ]
BALANCE_INFOS = ['currencyCode', 'TotalDebt', 'CashAndCashEquivalents', 'CommonStockEquity', "InvestedCapital", "ShareIssued"]
letters = list(ascii_uppercase)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0'}

### get fed fund rate (debt cost)
url_fed_fund_rate = "https://ycharts.com/indicators/effective_federal_funds_rate"
xpath_fed_fund_rate = "/html/body/main/div/div[4]/div/div/div/div/div[2]/div[1]/div[3]/div[2]/div/div[1]/table/tbody/tr[1]/td[2]"

r = requests.get(url_fed_fund_rate, verify= False, headers= headers)
text_fed_fund_rate = html.fromstring(r.content).xpath(xpath_fed_fund_rate)[0].text
debt_cost = float(text_fed_fund_rate.strip("%")) / 100

### get free risk rate
free_risk_rate = yq.Ticker("^TNX").history(period = '1d', ).loc["^TNX"]["close"].iloc[-1] /100 #us treasury ten years yield

### eval market rate
MARKET_CURRENCY = "USD"
sptr = yq.Ticker("^SP500TR").history(period = '5y', interval= "1mo").loc["^SP500TR"]
last_date = sptr.index[-2]
past_date = sptr.index[0]
delta = relativedelta(last_date, past_date)
timeDelta = delta.years + delta.months / 12 + delta.days / 365
market_rate =  (sptr.loc[last_date]['close'] / sptr.loc[past_date]['close'])**(1/timeDelta) - 1 # S&P 500 mean year rate over 5 years

print("debt cost = {}%".format(debt_cost*100))
print("free risk rate = {0:.2f}%".format(free_risk_rate*100))
print("market rate = {0:.2f}%".format(market_rate*100))

month_change_rm = sptr["adjclose"][:-1].pct_change(periods = 1).rename('rm')
var_rm = month_change_rm.var()

# rate history
rate_history_dic = {}
rate_current_dic = {}

# share profile
SHARE_PROFILE_FILE = 'share_profile.json'
if os.path.isfile(SHARE_PROFILE_FILE) :
    with open(SHARE_PROFILE_FILE, "r") as file :
        share_profile_dic = json.load(file)
        file.close()
else :
    share_profile_dic = {}
class Share():
    def __init__(self, isin : str, source : str = "yahoo" ) -> None:
        self.isin = isin
        self.symbol = None
        if source == "yahoo" :
            self.symbol = isin
        self.source = source
        self.fcf : float = None 
        self.cmpc = None 
        self.mean_g_fcf = None
        self.netDebt = None
        self.y_financial_data = None
        self.short_name = None
        self.currentprice = None
        self.financial_currency = None
        self.per = np.nan
        self.debt_ratio = np.nan
        self.priceToBook = np.nan
        self.mean_g_tr = None
        self.mean_g_netinc = None
        self.capital_cost = None
        self.roic = np.nan

    def eval_beta(self) :

        regular_history = self.history[:-1]

     
        if self.price_currency != MARKET_CURRENCY :
            change_rate = self.price_currency + MARKET_CURRENCY + "=X"
            if change_rate not in rate_history_dic :
                currency_history = yq.Ticker(change_rate).history(period= '5y', interval= "1mo").loc[change_rate]
                rate_history_dic[change_rate] = currency_history["close"][:-1]
                rate_current_dic[change_rate] = currency_history["close"].iloc[-1]
        
            regular_history = regular_history * rate_history_dic[change_rate]
     
        month_change_share = regular_history.pct_change().rename("share")
    
        cov_df = pd.concat([month_change_rm, month_change_share], axis = 1, join= 'inner',)
        cov = cov_df.cov()['rm'].loc['share']
        beta = cov/ var_rm
        self.beta = beta

        return beta

    def querry_financial_info(self, pr = False, ):
        
        tk = yq.Ticker(self.symbol)
        self.tk = tk
        print('\rquerry {}    '.format(self.symbol), flush=True, end="")

        self.history = self.tk.history(period = '5y', interval= "1mo").loc[self.symbol]['adjclose']
        self.currentprice = self.history.iloc[-1]

        if not self.symbol in share_profile_dic :

            url_yahoo = "https://finance.yahoo.com/quote/{0}?p={0}".format(self.symbol)

            short_name_xpath = '//*[@id="quote-header-info"]/div[2]/div[1]/div[1]/h1'
            currency_xpath = '//*[@id="quote-header-info"]/div[2]/div[1]/div[2]/span'
            
            # r = requests.get(url_yahoo, verify= False, headers= headers, cookies = cookies)
            r = requests.get(url_yahoo, verify= False, headers= headers, )
            self.short_name = html.fromstring(r.content).xpath(short_name_xpath)[0].text
            self.price_currency = html.fromstring(r.content).xpath(currency_xpath)[0].text.split()[-1].strip(')')
            share_profile_dic[self.symbol] = { "short_name" : self.short_name , "price_currency" : self.price_currency }
        
        else :
            self.short_name = share_profile_dic[self.symbol]["short_name"]
            self.price_currency = share_profile_dic[self.symbol]["price_currency"]
        # price = tk.price[self.symbol]
        # if('Quote not found' in price) :
        #     print(price)
        #     return(1)
        
        # self.price_currency = price['currency'] 
        # self.short_name = price["shortName"]
        # self.marketCap = price['marketCap']
        
    
        types = INCOME_INFOS + BALANCE_INFOS
        self.y_financial_data : pd.DataFrame = tk.get_financial_data(types,).ffill(axis = 0).drop_duplicates(subset = 'asOfDate', keep= 'last').set_index('asOfDate')
        self.q_financial_data = tk.get_financial_data(types , frequency= "q", trailing= False).set_index('asOfDate')
 
        if isinstance(self.q_financial_data, pd.DataFrame) :
            self.q_financial_data.ffill(axis = 0, inplace = True)
            last_date_y = self.y_financial_data.index[-1]
            last_date_q = self.q_financial_data.index[-1]
            for d in BALANCE_INFOS:
                if d in self.q_financial_data.columns :
                    self.y_financial_data.loc[last_date_y , d] =  self.q_financial_data.loc[last_date_q , d]

        self.nb_shares = self.y_financial_data["ShareIssued"].iloc[-1]
        self.marketCap = self.nb_shares * self.currentprice
   
        self.eval_beta()
        financial_currency = self.y_financial_data['currencyCode'].iloc[-1]
        
        if self.price_currency != financial_currency:
            rate_symb = self.price_currency + financial_currency + "=X"
            if rate_symb not in rate_current_dic:
                rate_tk = yq.Ticker(rate_symb)
                rate_current_dic[rate_symb] = rate_tk.history(period = '1d', ).loc[rate_symb]["close"].iloc[-1]
                # rate_current_dic[rate_symb] = rate_tk.price[rate_symb]['regularMarketPrice']
            rate = rate_current_dic[rate_symb]
            self.currentprice *= rate
            self.marketCap *= rate
        
        self.compute_financial_info(pr = pr)
        return(0)

    def compute_financial_info(self, pr = True) :

        y_financial_data = self.y_financial_data
        q_financial_data =  self.q_financial_data
        
        last_financial_info =  y_financial_data.iloc[-1]
     
        self.nb_shares = int(self.marketCap / self.currentprice)
        stockEquity = last_financial_info['CommonStockEquity']
        marketCap = self.marketCap
        self.capital_cost = free_risk_rate + self.beta * (market_rate - free_risk_rate)
        totalDebt = last_financial_info['TotalDebt']
        self.cmpc = self.capital_cost * stockEquity/(totalDebt + stockEquity) + debt_cost * (1-IS) * totalDebt/(totalDebt + stockEquity)
        self.netDebt = totalDebt - last_financial_info['CashAndCashEquivalents']


        self.netMarketCap = marketCap + self.netDebt
        if stockEquity >= 0 : 
            self.debt_ratio = self.netDebt / stockEquity
            self.priceToBook = marketCap / stockEquity
        

        y_financial_data['date'] = y_financial_data.index
        y_financial_data['year'] = y_financial_data['date'].apply(lambda  x : x.year )
        y_financial_data.drop_duplicates(subset= "year", keep= "last", inplace= True)
        
        netIncome = y_financial_data['NetIncome'][-3:].mean()
        last_delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[-2],)

        if netIncome >= 0 : 
            self.per = marketCap / netIncome
        investedCapital = last_financial_info['InvestedCapital']
        if investedCapital >= 0 : 
            self.roic = netIncome / investedCapital
        

        if last_delta_t.years < 1 and isinstance(q_financial_data, pd.DataFrame) and ('FreeCashFlow' in q_financial_data.columns) and  (self.q_financial_data.index[-1] > y_financial_data.index[-2]) :
           
            complement_q_financial_infos = self.q_financial_data[self.q_financial_data.index > y_financial_data.index[-2]]
            complement_time = relativedelta(complement_q_financial_infos.index[-1], y_financial_data.index[-2]).months / 12
            self.fcf = (y_financial_data['FreeCashFlow'][-4:-1].sum() + complement_q_financial_infos['FreeCashFlow'].sum()) / (3 + complement_time)
            
        else : 
            self.fcf = y_financial_data['FreeCashFlow'][-3:].mean()
        
        # if self.fcf < 0 :
        #     self.fcf = y_financial_infos['FreeCashFlow'][-1]

        if self.fcf < 0 :
            self.fcf = y_financial_data['FreeCashFlow'].mean()

     
        delta_t = relativedelta(y_financial_data.index[-1], y_financial_data.index[YEAR_G],)
        nb_years_fcf = delta_t.years + delta_t.months / 12

        fcf_se_ratio = y_financial_data['FreeCashFlow'].iloc[-1]/y_financial_data['FreeCashFlow'].iloc[YEAR_G]
        if fcf_se_ratio < 0:
            self.mean_g_fcf = np.nan
        else :
            self.mean_g_fcf = (fcf_se_ratio)**(1/nb_years_fcf) - 1

        self.financial_currency = last_financial_info['currencyCode']

        nb_year_inc = delta_t.years + delta_t.months / 12
   
        self.mean_g_tr = (y_financial_data['TotalRevenue'].iloc[-1]/y_financial_data['TotalRevenue'].iloc[YEAR_G])**(1/nb_year_inc) - 1
        inc_se_ratio = y_financial_data['NetIncome'].iloc[-1]/y_financial_data['NetIncome'].iloc[YEAR_G]
        if inc_se_ratio < 0 :
            self.mean_g_netinc = np.nan
        else : 
            self.mean_g_netinc = inc_se_ratio**(1/nb_year_inc) - 1
      
        if pr :

            print("\r")
            print(y_financial_data)
            print("Prix courant: {0:.2f} {1:s}".format(self.currentprice, self.financial_currency) )
            print("Cout moyen pondere du capital: {0:.2f}%".format(self.cmpc*100) )
            print("Croissance moyenne du chiffre d'affaire sur {0:f} ans: {1:.2f}%".format(nb_year_inc,self.mean_g_tr*100))
            print("Croissance moyenne du free cash flow sur {0:f} ans: {1:.2f}%".format(nb_year_inc,self.mean_g_fcf*100))

        pass

    def get_dcf(self, g : float = None, start_fcf : float = None, pr = False) -> float :
        
        if self.cmpc is None :
            self.querry_financial_info(pr = pr)
        
        if g is None :
            g = self.mean_g_tr

        if start_fcf is None :
            start_fcf = self.fcf

        get_dcf_(g, self, start_fcf, pr)
    
    def eval_g(self, start_fcf : float = None, pr=False):

        if self.cmpc is None :
            if self.querry_financial_info(pr = pr) : return(np.nan)

        if self.cmpc < 0:
            print("negative cmpc for {} can not compute DCF".format(self.short_name))
            return np.nan
        if start_fcf is None :
            start_fcf = self.fcf
        if start_fcf < 0 :
            print("negative free cash flow mean for {} can not compute DCF".format(self.short_name))
            return np.nan
        res = minimize_scalar(get_dcf_, args=(self, start_fcf, False), method= 'bounded', bounds = (-1, self.cmpc) )
        g = res.x

        self.g = g
        if pr:
            print("Croissance correspondant au prix courrant: {:.2f}%".format(g*100))
            self.get_dcf(g, start_fcf= start_fcf, pr = pr)
            
        return g

def get_dcf_(g, *data):
    share, start_fcf, pr = data       
    nb_year = NB_YEAR_DCF
    cmpc = share.cmpc
    netDebt = share.netDebt
    if isinstance(g, (list, np.ndarray)):
        g = g[0]

    vt = start_fcf * (1+g)**(NB_YEAR_DCF + 1) / (cmpc - g)
    vt_act = vt / (1+cmpc)**(nb_year + 1)

    fcf = np.array([start_fcf * (1+g)**(k) for k in range(1,1 + nb_year)])
    act_vec = np.array([1/((1+cmpc)**k) for k in range(1,1 + nb_year)])
    fcf_act = fcf * act_vec

    enterpriseValue = fcf_act.sum() + vt_act
    
    if pr :
        print("\r")
        val_share = (enterpriseValue - netDebt)/ share.nb_shares
        nyear_disp = min(10,nb_year)
        annees = list(2023 + np.arange(0, nyear_disp)) +  ["Terminal"]
      
        table = np.array([ np.concatenate((fcf[:nyear_disp] ,[vt])), np.concatenate((fcf_act[:nyear_disp], [vt_act]))])
        print("Prévision pour une croissance de {:.2f}% :".format(g*100))
        print(tabulate(table, floatfmt= ".4e", showindex= [ "Free Cash Flow", "Free Cash Flow actualisé"], headers= annees))

        print("Valeur DCF de l'action: {0:.2f} {1:s}".format(val_share, share.price_currency))
    
    # return ((enterpriseValue - netDebt)/ share.marketCap - 1)**2
    return (enterpriseValue / share.netMarketCap - 1)**2

def resume_list(symbol_list : list[str | tuple], xl_outfile : str = None):
    share_list = [Share(sym) for sym in symbol_list]
    g_l = [s.eval_g() for s in share_list]
    
    with open(SHARE_PROFILE_FILE, "w") as outfile :
        json.dump(share_profile_dic, outfile, indent = 4)
        outfile.close() 
    df = pd.DataFrame(index= symbol_list, data= {'short_name' : [s.short_name for s in share_list] ,
                                                'current_price' : [s.currentprice for s in share_list],
                                                'currency' : [s.financial_currency for s in share_list],
                                                'capital_cost' :[s.capital_cost for s in share_list],
                                                'cmpc' :[s.cmpc for s in share_list],
                                                'assumed_g' : g_l ,  
                                                'per' :  [s.per for s in share_list ],
                                                'roic' : [s.roic for s in share_list], 
                                                'debt_ratio' : [s.debt_ratio for s in share_list],
                                                'price_to_book' : [s.priceToBook for s in share_list] ,
                                                'mean_g_fcf': [s.mean_g_fcf for s in share_list] , 
                                                'mean_g_tr' : [s.mean_g_tr for s in share_list], 
                                                'mean_g_inc' : [s.mean_g_netinc for s in share_list] 
                                                })


    df["diff_g"] = df['mean_g_tr'] - df['assumed_g']
    col_letter = {c : letters[i+1] for i, c in enumerate(df.columns)}
    df.sort_values(by = ['assumed_g', 'debt_ratio']  , inplace= True, ascending= True)


   
    writer = pd.ExcelWriter(xl_outfile,  engine="xlsxwriter")
    df.to_excel(writer, sheet_name= "dcf")
    wb = writer.book
    number = wb.add_format({'num_format': '0.00'})
    percent = wb.add_format({'num_format': '0.00%'})
    # Add a format. Light red fill with dark red text.
    format1 = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
    # Add a format. Green fill with dark green text.
    format2 = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    # Add a format. Light red fill .
    format3 = wb.add_format({"bg_color": "#F8696B",})

    worksheet = writer.sheets['dcf']
    worksheet.add_table(0,0,len(df.index),len(df.columns)  , {"columns" : [{'header' : 'symbol'}] + [{'header' : col} for col in df.columns], 'style' : 'Table Style Light 8'})
    worksheet.set_column('B:B', 20, )
    worksheet.set_column(f"{col_letter['current_price']}:{col_letter['current_price']}", 13, number)
    worksheet.set_column(f"{col_letter['capital_cost']}:{col_letter['assumed_g']}", 13, percent)
    worksheet.set_column(f"{col_letter['per']}:{col_letter['price_to_book']}", 13, number)
    worksheet.set_column(f"{col_letter['roic']}:{col_letter['roic']}", 13, percent )
    worksheet.set_column(f"{col_letter['mean_g_fcf']}:{col_letter['diff_g']}", 13, percent )

    # format assumed g
    worksheet.conditional_format(f"{col_letter['assumed_g']}2:{col_letter['assumed_g']}{len(df.index)+1}", 
                                 {"type": "3_color_scale", 'min_type': 'num','max_type': 'max', 'mid_type' : 'percentile',
                            'min_value' : -0.2, 'mid_value' : 50,  'min_color' : '#63BE7B', "max_color" : '#F8696B', "mid_color" : "#FFFFFF"})
    
    # format debt ratio
    worksheet.conditional_format(f"{col_letter['debt_ratio']}2:{col_letter['debt_ratio']}{len(df.index)+1}", 
                                 {"type": "3_color_scale", 'min_type': 'num','max_type': 'num', 'mid_type' : 'percentile',
                            'min_value' : 0, 'mid_value' : 50, "max_value" : 2, 'min_color' : '#63BE7B', "max_color" : '#F8696B', "mid_color" : "#FFFFFF"})
    # format PER
    worksheet.conditional_format(f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format3})
    worksheet.conditional_format(f"{col_letter['per']}2:{col_letter['per']}{len(df.index)+1}", 
                                 {"type": "3_color_scale", 'min_type': 'num','max_type': 'num', 'mid_type' : 'percentile',
                            'min_value' : 3, 'mid_value' : 50, "max_value" : 50, 'min_color' : '#63BE7B', "max_color" : '#F8696B', "mid_color" : "#FFFFFF"})
    # format ROIC
    worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format3})
    worksheet.conditional_format(f"{col_letter['roic']}2:{col_letter['roic']}{len(df.index)+1}", 
                                 {"type": "3_color_scale", 'min_type': 'num','max_type': 'num', 'mid_type' : 'percentile',
                            'min_value' : 0, 'mid_value' : 50, "max_value" : 0.15, "min_color" : '#F8696B', 'max_color' : '#63BE7B' , "mid_color" : "#FFFFFF"})

    # format Price to Book
    worksheet.conditional_format(f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format3})
    worksheet.conditional_format(f"{col_letter['price_to_book']}2:{col_letter['price_to_book']}{len(df.index)+1}", 
                                 {"type": "3_color_scale", 'min_type': 'num','max_type': 'num', 'mid_type' : 'percentile',
                            'min_value' : 1, 'mid_value' : 50, "max_value" : 10, 'min_color' : '#63BE7B', "max_color" : '#F8696B', "mid_color" : "#FFFFFF"})
    
    #
    worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": "<", "value": 0, "format": format1})
    worksheet.conditional_format(f"{col_letter['mean_g_fcf']}2:{col_letter['diff_g']}{len(df.index)+1}", {"type": "cell", "criteria": ">", "value": 0, "format": format2})
    writer.close()
    subprocess.call(["open", xl_outfile])
    
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




