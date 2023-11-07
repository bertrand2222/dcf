import sys, os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)
from dcf import *

outfile = "/home/bertrand/Documents/dcf.xlsx"
share_list = [
    "OR.PA", #l'Oreal  
    "AAPL", 
    "GE",
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
    "TTE",
    "SPM.MI",
    "SAF.PA", #Safran
    "PBR",
    "RMS.PA", #hermes
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
    "CA.PA", #carrefour
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
    "AC.PA",
    "TSM",
    "TLW.L", # Tullow Oil
    "INTC",
    "SCI", 
    "ACA.PA",
    'STMPA.PA', #stmicroelec
    "RWE.DE" ,
    "PYPL", #paypal
    "SW.PA",
    "SUBCY",
    'GTT.PA',
    'PAH3.DE',
    'CRH',
    "ENGI.PA",
    "VK.PA",
    "ATE.PA",
    "ALO.PA", # Alstom
    "ENI.MI", # ENI
    "RI.PA",
    "FNAC.PA",
    "SU.PA",
    "8GC.F",
    "AMUN.PA",
    "HEIA.AS",
    "8058.T",
    "8002.T",
    "BAC",
    "USB",
    "BMW.DE",
    "VOW.DE",
    "EC.PA", # total gabon
    "ETL.PA", # eutelsat
    "VRLA.PA", # veralia
    "SII.PA", 
    "RXL.PA",
    "MAU.PA",
    "EQNR",   # equinor  
    "DIS",
      ]

resume_list(share_list, outfile)
# resume_list(["GLEN.L"])

# share = Share("PBR")
# share.eval_g(2.6e9, True)
# share = Share("TTE.PA")
# share.eval_g( pr= True, )
# share.get_dcf( pr= True)