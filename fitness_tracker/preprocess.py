import joblib 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from load_data import PROCESSED,load_row,split_xy

def main():
    train,test=load_row()
    # x_tr,y_tr,s_tr=split_xy(train)
    print(train.head(2))


if __name__ == "__main__":
    main()





    