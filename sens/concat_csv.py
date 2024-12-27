import numpy as np
import pandas as pd 
from pathlib import Path

datadir = Path('data')
vtlist = [24,48,72,96]
nelist = [8,16,24,32,40]

for dtype in ['aspect','slope','rmsd']:
    frames = []
    for vt in vtlist:
        for ne in nelist:
            df = pd.read_csv(datadir/f'all_mul-mul_{dtype}_vt{vt}ne{ne}.csv')
            frames.append(df)
    dfc = pd.concat(frames)
    print(dfc)
    dfc[['FT','member','asa','diag','minnorm','ridge','pcr','pls']].to_csv(datadir/f'all_mul-mul_{dtype}2.csv',index=False)