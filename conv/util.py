import xarray 
import numpy as np
import os
DATADIR = os.environ.get("DATADIR",'../data')

def get_dataset():
    ds =  xarray.load_dataset(os.path.join(DATADIR,'IBTrACS.since1980.v04r00.nc'))
    return ds.assign_coords(storm=ds.sid)

def train_validation_test(ds:xarray.Dataset,seed:float=None,test_fraction:float=0.1,validation_fraction:float=0.2):
    ds = ds.assign_coords(storm=ds.sid)
    rng = np.random.default_rng(seed=seed)
    train_fraction = 1 - test_fraction - validation_fraction
    ## Check for at least one USA wind 
    main_track_shuffle = rng.permutation(ds.sid[(ds.track_type == b'main')&(~np.isnan(ds.usa_wind[:,0]))])
    train_idx = int(len(main_track_shuffle) * train_fraction)
    test_idx = train_idx + int(len(main_track_shuffle) * validation_fraction)

    return main_track_shuffle[:train_idx],main_track_shuffle[train_idx:test_idx],main_track_shuffle[:test_idx]


def get_x(ds:xarray.Dataset,idx:np.ndarray):
    return np.column_stack((
        ds.lat.loc[idx].to_numpy(),
        ds.lon.loc[idx].to_numpy(),
        ds.usa_wind.loc[idx].to_numpy(),
        ds.usa_pres.loc[idx].to_numpy(),
        ds.dist2land.loc[idx].to_numpy()
    ))

def get_sst_ds():
    sst_ds = xarray.open_dataset(os.path.join(DATADIR,'sst.wkmean.1990-present.nc'))
    sst_ds = sst_ds.assign_coords(lon=(((sst_ds.lon + 180) % 360) - 180))
    return sst_ds

def coriolis(lat):
    return np.sin(np.deg2rad(lat))

def make_X_y(ds,sst_ds,selected_storms,timesteps=5):
    Xout = []
    yout =[]
    storms = []
    for storm in selected_storms:
        usa_pres = ds.usa_pres.loc[storm]
        usa_wind = ds.usa_wind.loc[storm]
        ## All enteries have 360 points.
        valid_coords = ~(np.isnan(usa_wind) | np.isnan(usa_pres))
        lat = ds.lat.loc[storm][valid_coords]
        lon = ds.lon.loc[storm][valid_coords]
        storm_speed = ds.storm_speed.loc[storm][valid_coords]
        storm_dir = ds.storm_dir.loc[storm][valid_coords]
        u = storm_speed*np.sin(np.deg2rad(storm_dir))
        v = storm_speed*np.cos(np.deg2rad(storm_dir))
        usa_pres = usa_pres[valid_coords]
        usa_wind = usa_wind[valid_coords]
        time = ds.time.loc[storm][valid_coords]
        cor_param = coriolis(lat)
        try:
            sst = sst_ds.sst.interp(time=time,lat=lat,lon=lon)
        except ValueError:
            continue
        if np.isnan(sst).any():
            continue

        X = np.transpose(np.array([usa_wind,usa_pres,u,v,cor_param,sst,lat,lon]))
        for i in range(0,len(usa_wind)):
            if i+timesteps+1>=len(usa_wind):
                break
            Xout.append(X[i:i+timesteps])
            yout.append(X[i+timesteps+1][:-4])
            storms.append(storm)
    return np.stack(Xout),np.stack(yout),np.array(storms)

def get_storm_seeds(ds,sst_ds,selected_storms,timestep:int=5):
    """
    Similar but generates X, vector of storm seed and y, vector of entire storm prediction track
    """
    Xout,yout,start_times=[],[],[]

    for storm in selected_storms:
        usa_pres = ds.usa_pres.loc[storm]
        usa_wind = ds.usa_wind.loc[storm]
        ## All enteries have 360 points.
        valid_coords = ~(np.isnan(usa_wind) | np.isnan(usa_pres))
        lat = ds.lat.loc[storm][valid_coords]
        lon = ds.lon.loc[storm][valid_coords]
        storm_speed = ds.storm_speed.loc[storm][valid_coords]
        storm_dir = ds.storm_dir.loc[storm][valid_coords]
        u = storm_speed*np.sin(np.deg2rad(storm_dir))
        v = storm_speed*np.cos(np.deg2rad(storm_dir))
        usa_pres = usa_pres[valid_coords]
        usa_wind = usa_wind[valid_coords]
        time = ds.time.loc[storm][valid_coords]
        cor_param = coriolis(lat)
        try:
            sst = sst_ds.sst.interp(time=time,lat=lat,lon=lon)
        except ValueError:
            continue
        if np.isnan(sst).any():
            continue
        if timestep+1>=len(usa_wind):
            continue

        X = np.transpose(np.array([usa_wind,usa_pres,u,v,cor_param,sst,lat,lon]))
    
        Xout.append(X[:timestep])
        yout.append(X)
        start_times.append(time[timestep].values)
    return Xout,yout,np.array(start_times)