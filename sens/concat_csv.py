import numpy as np
import pandas as pd 
from pathlib import Path

datadir = Path('/Volumes/FF520/pyesa/adata/l96')
vtlist = [24,48,72,96]
nelist = [8,16,24,32,40]

for sample in ['all','far','near']:
    for dtype in ['aspect','slope','rmsd']:
        frames = []
        for vt in vtlist:
            for ne in nelist:
                df = pd.read_csv(datadir/f'{sample}_mul-mul_{dtype}_vt{vt}ne{ne}.csv')
                frames.append(df)
        dfc = pd.concat(frames)
        print(dfc)
        dfc[['FT','member','asa','diag','minnorm','ridge','pcr','pls']].to_csv(datadir/f'{sample}_mul-mul_{dtype}.csv',index=False)

for csvname in ['res_hess','res_hessens','res_hessens_loc3']:
    for dtype in ['estp_mean','estm_mean','calcp_mean','calcm_mean','rmsd_p','rmsd_m']:
        frames = []
        for vt in vtlist:
            for ne in nelist:
                df = pd.read_csv(datadir/f'{csvname}_{dtype}_vt{vt}ne{ne}.csv')
                frames.append(df)
        dfc = pd.concat(frames)
        dfc[['FT','member','asa','minnorm','ridge','pcr','pls']].to_csv(datadir/f'{csvname}_{dtype}.csv',index=False)