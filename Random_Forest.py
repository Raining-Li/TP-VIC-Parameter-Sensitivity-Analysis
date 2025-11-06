import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import os

def datalists(path):
    arr = []
    lists = os.listdir(path)
    for l in lists:
        fni = path + '/' +l
        arr.append(fni)
    return arr

# readdata
vardir=r"E:\SA\kge_spaef\allbasin_allvar"
outdir0=r"E:\SA\kge_spaef\RF-SHAP_result"

txtlist=datalists(vardir)

print(txtlist)

for txt in txtlist:
    df = pd.read_csv(txt,sep='\t')

    varlist=['INFILT','Ds','Ds_MAX','Ws','C','Expt_1','Ksat_1','DEPTH_1','DEPTH_2','DEPTH_3','Dp','Z0_SOIL','Z0_SNOW','RESM1','LAPSE_RATE','VEG_LAI_SNOW_MULTIPLIER','VEG_LAI_WATER_FACTOR','SNOW_LIQUID_WATER_CAPACITY','SNOW_NEW_DENSITY','SNOW_NEW_ALB','SNOW_ALB_ACCUM_A','SNOW_ALB_ACCUM_B','SNOW_ALB_THAW_A','SNOW_ALB_THAW_B','SNOW_MAX_SNOW_TEMP','SNOW_MIN_RAIN_TEMP','Rarc','Rmin','LAI','ALB','ROU','root_depth','root_fraction']

    txtname=txt.split('/')[-1].split('.')[0]

    outdir=os.path.join(outdir0,txtname)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for v in range(len(varlist)):

        target_col = varlist[v]
        feature_cols = ['lon','lat','elevation', 'temp', 'prec', 'wind', 'pres', 'shum', 'lrad', 'srad','vp']

        outpath = os.path.join(outdir, f'{target_col}_shap.txt')

        if not os.path.exists(outpath):

            X = df[feature_cols]
            y = df[target_col]

            # RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)

            # shap.Explainer
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

            shap_array = shap_values.values  # numpy array
            shap_df = pd.DataFrame(shap_array, columns=[f"{col}_shap" for col in X.columns])

            combined_df = pd.concat([X.reset_index(drop=True), shap_df], axis=1)

            # save
            outpath=os.path.join(outdir,f'{target_col}_shap.txt')
            combined_df.to_csv(outpath, sep='\t', index=False, float_format="%.6f")



