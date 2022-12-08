import numpy as np
import util 
import argparse
import os 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("timepoints",type=int)
    args = parser.parse_args()

    ibtracs_ds = util.get_dataset()
    sst_ds = util.get_sst_ds()
    train_storms,valid_storms,test_storms = util.train_validation_test(ibtracs_ds,seed=42)

    X_train,y_train,group_train = util.make_X_y(ibtracs_ds,sst_ds,train_storms,timesteps=args.timepoints)
    X_validation,y_validation,group_val = util.make_X_y(ibtracs_ds,sst_ds,valid_storms,timesteps=args.timepoints)

    X_train = np.transpose(X_train,axes=(0,2,1))
    X_validation = np.transpose(X_validation,axes=(0,2,1))

    DATADIR = os.environ.get("DATADIR","../data")

    np.save(os.path.join(DATADIR,f"X_train_{args.timepoints}.npy"),X_train)
    np.save(os.path.join(DATADIR,f"y_train_{args.timepoints}.npy"),y_train)

    np.save(os.path.join(DATADIR,f"X_validation_{args.timepoints}.npy"),X_validation)
    np.save(os.path.join(DATADIR,f"y_validation_{args.timepoints}.npy"),y_validation)