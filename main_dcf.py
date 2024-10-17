import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dcf import *

from datetime import date

outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

share_list = [
    "TLW.L", # Tullow Oil
    "TSM",
    "PBR",
    "TTE.PA",
    "MC.PA", #LVMH
    "6501.T",  # Hitachi 
    "RMS.PA", #hermes
    'GTT.PA',
    "ACA.PA", #credit agricole
    "OR.PA", #l'Oreal  
    "TE.PA", #technip energie
    "RUI.PA",
    "BNP.PA",
    "GLE.PA",
    "ANSS", # ansys
    "ASML.AS",
    "DELL",
    "TSLA", 
    "ERA.PA", #eramet
    "AMZN", #AMZN  
    "SPM.MI",
    "SAF.PA", #Safran
    "AKE.PA",
    "NVDA",
    "SAN.PA", # sanofi
    "DG.PA", #Vinci
    "RS", # reliance steel
    "APAM.AS",
    "AI.PA", #air liquide
    "DSY.PA", #dassault system
    "MSFT", #microsoft
    "SGO.PA", #saint gobain
    "CS.PA", #Axa 
    "FGR.PA", #eiffage
    "GOOG",
    # "CA.PA", #carrefour
    "META",
    "AIR.PA", #Airbus
    "CAP.PA", #capgemi
    "HO.PA", #Tales
    "COFA.PA", #coface
    "STLAP.PA", # stelantis
    "EPD", # enterprise product partner
    "DBG.PA",
    "SIE.DE", # siemens
    "BN.PA",
    "RR.L", # rolls royce
    "INTC",
    "SCI", 
    'STMPA.PA', #stmicroelec
    "SW.PA",
    "SUBCY",
    'PAH3.DE',
    'CRH',
    "ENGI.PA",
    "VK.PA",
    "RELIANCE.NS",
    "ENI.MI", # ENI
    "RI.PA",
    "CVX",
    "XOM",
    "SU.PA",
    "8GC.F",
    "AMUN.PA",
    "8058.T",
    "8002.T",
    "BAC",
    "USB",
    "BMW.DE",
    "VOW.DE",
    "EC.PA", # total gabon
    "VRLA.PA", # veralia
    "RXL.PA",
    "MAU.PA",
    "EQNR",   # equinor  
    "SCR.PA",
    "CLIQ.DE",
    "TM", # Toyota
    "SMSN.IL", #samsung electronic
    "1211.HK", # BYD co electric car
    "AAPL"
      ]

share_list_test = [
    "TLW.L", # Tullow Oil
    "TSM",
    "PBR",
    "TTE.PA",
]
# share = Share("TSM")
# share.querry_financial_info()
# print(share.y_financial_data)
# print(share.price_currency)
# print(share.df_multiple)

dcf_anal = DCFAnal(share_list, 
                   capital_cost_equal_market = True)
dcf_anal.process_list()
# dcf_anal.load_df()

dcf_anal.to_excel(outfile)
# upload_file(outfile)




# share = Share("ACA.PA")
# share.eval_g(2.6e9, True)
# share = Share("TTE.PA")
# share.eval_g( pr= True, )
# share.get_dcf( pr= True)
