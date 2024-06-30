import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from dcf import *

from datetime import date

outfile = os.path.join(os.environ["USERPROFILE"], r"Documents\rdcf.xlsx")

share_list = [
    "TTE.PA",
    "RMS.PA", #hermes
    'GTT.PA',
    "ACA.PA", #credit agricole
    "OR.PA", #l'Oreal  
    "TE.PA", #technip energie
    "RUI.PA",
    "BNP.PA",
    "FRVIA.PA", # faurecia
    "GLE.PA",
    "ASML.AS",
    "DELL",
    "TSLA", 
    "MT.AS", # arcelormital
    "ERA.PA", #eramet
    "MC.PA", #LVMH
    "CARL-B.CO",
    "TEP.PA", #teleperformance
    "AMZN", #AMZN  
    "SPM.MI",
    "SAF.PA", #Safran
    "PBR",
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
    "BN.PA",
    "RR.L", # rolls royce
    "TSM",
    "TLW.L", # Tullow Oil
    "INTC",
    "SCI", 
    'STMPA.PA', #stmicroelec
    "RWE.DE" ,
    "SW.PA",
    "SUBCY",
    'PAH3.DE',
    'CRH',
    "ENGI.PA",
    "VK.PA",
    "RELIANCE.NS",
    "ENI.MI", # ENI
    "RI.PA",
    "FNAC.PA",
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
    "DIS",
    "SCR.PA",
    "CLIQ.F",
    "TM", # Toyota
    "SMSN.IL", #samsung electronic
    "1211.HK", # BYD co electric car
    "6501.T",  # Hitachi 
    "AAPL"
      ]


# share_list_1 = ["MSFT"]
# dcf_anal = DCFAnal(share_list_1)

dcf_anal = DCFAnal(share_list)
dcf_anal.resume_list()
# dcf_anal.load_df()

dcf_anal.to_excel(outfile)
# upload_file(outfile)



# share = Share("PBR")
# share = Share("ACA.PA")
# share.eval_g(2.6e9, True)
# share = Share("TTE.PA")
# share.eval_g( pr= True, )
# share.get_dcf( pr= True)
