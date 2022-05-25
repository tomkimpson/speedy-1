from scipy.stats import wasserstein_distance as wd



#Import everything
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import pandas as pd
import seaborn as sns
import sys
from collections import defaultdict
from tqdm.notebook import tqdm
import matplotlib as mpl





def extract_file_data(df,columns):
    
    surface = df.sel(pressure=925.0)
        
    data = surface.to_dataframe()[columns]
    
    return data


 
def process_nc_file(fname,weights,true_latitude,columns):    
    
        #Get data 
        df = xr.open_dataset(fname)
        
        
        #Reset the latitude
        df = df.assign_coords(latitude=(true_latitude))
        

       
        #Get the data you want     
        grid_data = extract_file_data(df,columns)
        

        
        return grid_data

def process_directory(directory,weights,true_lat,columns):

    
    #Empty arrays to hold data
    dfs = []
    
    nc_files = sorted(glob.glob(directory+'/model_output*.nc')) #Dont process last file which is only super short
                
    for n in tqdm(nc_files): #for every model.nc file

        df_snapshot = process_nc_file(n,weights,true_lat,columns)
        dfs.append(df_snapshot)
        
    df = pd.concat(dfs,ignore_index=False)
    
    return df 


        
    
def get_global_weights():
    
     #Get the latitude weights from a special location
    r1 = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/speedyone/paper/Fig1_10year_Williams/'
    f = r1 + 'speedyoneWILLIAMS_L2_52_RN_10y/model_output00001.nc'
    df = xr.open_dataset(f)
    
    temp_file = df.temperature
    weights = np.cos(np.deg2rad(temp_file.latitude))
         
    weights.name = "weights"
    
    
    return weights, temp_file.latitude   


def compare_ensemble_against_control(control_dirs,competitor_dirs,columns):
    print('here inside func')

    
    
    #Get weights and correct latitudes now
    weights,latitude = get_global_weights()
    
    #Create an array that will hold all the data you are about
    lenT = 10
    out_array = np.zeros((lenT*len(control_dirs)*len(competitor_dirs),4))
    global_counter = 0
    
    for i in control_dirs:
        print ('Loading control data from', i)
        #Get the dataframe
        df_i = process_directory(i,weights,latitude,columns)
        #Get the dimensions: x,y,time
        long_idx = df_i.index.unique(level='longitude') #length 96
        lat_idx = df_i.index.unique(level='latitude') #length 48
        t_idx = df_i.index.unique(level='forecast_period') #length 3651

        
        ensemble_number_control = i.split('_m')[-1][0]

        #Convert to a numpy array
        np_control = np.array(df_i.values.reshape(len(long_idx),len(lat_idx),len(t_idx)))
        for j in competitor_dirs:
            print ('Loading competitor data from', j)
            ensemble_number_competitor=j.split('_m')[-1][0]
                       
            #Get the competitor dataframe 
            df_j = process_directory(j,weights,latitude,columns)
            #...and also make this a numpy array
            np_competitor = np.array(df_j.values.reshape(len(long_idx),len(lat_idx),len(t_idx)))

            
            #Get the WD between i and j, averaged across every grid point, up to a time t
          
            for t in range(1,lenT+1):
                tfactor = int(t * len(t_idx)/lenT)
                print(tfactor,t,len(t_idx),lenT)
                wasserstein_t_weighted = []
                wasserstein_t_unweighted = []
                for k in range(len(long_idx)):
                    for m in range(len(lat_idx)):
                        wasserstein= wd(np_competitor[k,m,0:tfactor], np_control[k,m,0:tfactor])
                        wasserstein_weighted = weights[m].values*wasserstein #Weight the value by the latitude weight


                        wasserstein_t_weighted.extend([wasserstein_weighted])
                        wasserstein_t_unweighted.extend([wasserstein])
                
                #Output to the global np array
                print(ensemble_number_control,ensemble_number_competitor,tfactor, np.mean(wasserstein_t_weighted))
                out_array[global_counter,:] = ensemble_number_control,ensemble_number_competitor,tfactor, np.mean(wasserstein_t_weighted)
                global_counter = global_counter +1
                
                
    return out_array
                
                        

columns = ['temperature']



root = '/network/group/aopp/predict/TIP016_PAXTON_RPSPEEDY/speedyone/paper/10yearWD/'


control_dirs = []
for m in range(0,4+1):
    dir_name = root + f'speedyone10yr_L2_52_RN_m{m}_WD'
    control_dirs.append(dir_name)
 

competitor_dirs = []
for m in range(5,9+1):
    dir_name = root + f'speedyone10yr_L2_52_RN_m{m}_WD'
    competitor_dirs.append(dir_name)
 


output = compare_ensemble_against_control(control_dirs,competitor_dirs,columns)
np.save(root+'throwawayWD.npy',output)


