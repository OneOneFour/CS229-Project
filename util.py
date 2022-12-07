import xarray 
import numpy as np

def get_dataset():
    ds =  xarray.load_dataset("IBTrACS.since1980.v04r00.nc",)
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

