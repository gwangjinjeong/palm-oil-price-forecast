
import argparse
import os
import subprocess, sys
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd
import numpy as np
from glob import glob
import copy
import joblib

from sklearn.preprocessing import MinMaxScaler

import logging
import logging.handlers
from logging.config import dictConfig

import json
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

import time
from datetime import datetime as dt
import datetime
from dateutil.relativedelta import *


###############################
######### 전역변수 설정 ##########
###############################
KST = dt.today() + relativedelta(hours=9)
KST_aday_before = KST - relativedelta(days=1) 
lst_remove = ['CCMc1', 'CCMc2','CCMc3', 'CJc1', 'CJc2','CJc3',
              'DBLc1','DBLc2', 'DBLc3', 'TTAc1',  'TTAc2', 'TTAc3',
              'YOc1', 'YOc2', 'YOc3']

###############################
######### util 함수 설정 ##########
###############################
def _get_logger():
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        l.addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  
logger = _get_logger()

def scaling_value(df : pd.DataFrame,
                  col_name : str,
                  output_dir : str,
                  ric) -> tuple:
    
    series = df[col_name].values
    scaler = MinMaxScaler()
    series = series.reshape(-1,1)
    scaler.fit(series)
    series = scaler.transform(series)
    joblib.dump(scaler, os.path.join(output_dir, f'{ric}_{col_name}_scaler.pkl'))    
    return series

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_input_dir', type=str, default="/opt/ml/processing/input") 
    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--split_start', type=str, default='2011-01-17')    
    parser.add_argument('--split_end', type=str, default=KST.strftime('%Y-%m-%d'))
    parser.add_argument('--num_fold', type=str, default='5')
    parser.add_argument('--item_id', type=str, default="ZREUTERS") 
    parser.add_argument("--scaler_switch", type = str, default = '1', help = '1이면 Scaling ON, 0이면 Scaling OFF')
    parser.add_argument("--prediction_length", type = str, default = '30')
    return parser.parse_args()

if __name__=='__main__':
    total_start = time.time()

    ################################
    ###### 0. 커맨드 인자 파싱   #######
    ################################
    logger.info("Starting preprocessing.")
    logger.info("### Argument Info ###")
    args = parse_args()
    logger.info(f"args.base_input_dir: {args.base_input_dir}")    
    logger.info(f"args.base_output_dir: {args.base_output_dir}")
    logger.info(f"args.item_id: {args.item_id}")        
    logger.info(f"args.split_start: {args.split_start}")   
    logger.info(f"args.split_end: {args.split_end}")   
    logger.info(f"args.scaler_switch: {args.scaler_switch}")
    logger.info(f"args.num_fold: {args.num_fold}")
    logger.info(f"args.prediction_length: {args.prediction_length}")

    base_output_dir = args.base_output_dir
    base_input_dir = args.base_input_dir
    item_id = args.item_id
    split_start = args.split_start
    split_end = args.split_end
    num_fold = int(args.num_fold)
    scaler_switch = int(args.scaler_switch)
    prediction_length = int(args.prediction_length)
    
    #######################################################
    ###### 0. AWS 리소스 사용을 위한 Boto3 변수 선언  ########
    #######################################################                            
    boto_session = boto3.Session()
    region = boto_session.region_name
    s3_resource = boto_session.resource('s3')
    s3_client = boto_session.client('s3')
                                      
    ################################
    ###### 1. 공통 전처리 단계  ########
    ################################
    prop_start = time.time()

    logger.info(f"#### 1-1) 파일 읽기")
    df_raw = pd.read_csv(f'{base_input_dir}/integrated.csv',
                         parse_dates=['ZDATE'])
    df_proc = df_raw.copy() 
    logger.info(f"The number for data : {df_raw.shape}")

    logger.info(f"#### 1-2) 형 변환")
    df_proc["ZDATE"] = pd.to_datetime(df_proc["ZDATE"])
    df_proc["CLOSE"] = df_proc["CLOSE"].astype(np.float32)
    df_proc["HIGH"] = df_proc["HIGH"].astype(np.float32)
    df_proc["LOW"] = df_proc["LOW"].astype(np.float32)
    df_proc["OPEN"] = df_proc["OPEN"].astype(np.float32)

    logger.info(f"#### 1-3) 제외할 품목 선택")
    df_proc = df_proc[~df_proc[item_id].isin(lst_remove)].reset_index(drop = True)
    logger.info(f"The number for data after RIC Item selection : {df_proc.shape}")
    
    logger.info(f"#### 1-4) 제외할 컬럼 선별")
    df_proc = df_proc[['ZDATE','CLOSE', item_id]]        
    logger.info(f"The number for data after Column selection : {df_proc.shape}")

    logger.info(f"#### 1-5) 제외할 Date 선별")
    df_proc = df_proc[(df_proc['ZDATE'] >= split_start) & (df_proc['ZDATE'] <= split_end)]
    logger.info(f"The number for data after Date Selection : {df_proc.shape}")

    logger.info(f"#### 1-6) 컬럼 이름 변경")
    df_proc.rename(columns = {'ZDATE' : 'timestamp',
                              'CLOSE' : 'target',
                              item_id : 'item_id'},
                   inplace=True)    
    df_proc = df_proc.reset_index(drop = True)
    prop_end = time.time()
    logger.info(f"#### 데이터 프로세싱 Runtime ####")
    logger.info(f"{df_proc.head(2)}")
    logger.info(f"{prop_end - prop_start} sec")
    
    ####################################################
    ####   2. Autogluon timeseries 데이터 셋으로 만들기  ####
    ####################################################
    data_cleansing_start = time.time()
    
    logger.info(f"#### 2-1) Convert Pandas Dataframe into Autogluon TimeseriesDataframe ")
    
    df_ds_autogluon = TimeSeriesDataFrame.from_data_frame(
        df = df_proc,
        id_column = "item_id",
        timestamp_column = "timestamp"
    )
    
    ####################################################
    ####  3. Autogluon timeseries 업샘플링, 결측치 처리  ####
    ####################################################
    logger.info(f"#### 3-1) Autogluon timeseries 업샘플링")
    df_ds_autogluon = df_ds_autogluon.to_regular_index(freq = "D")
    
    logger.info(f"#### 3-2) Autogluon timeseries 결측치 처리")    
    df_ds_autogluon = df_ds_autogluon.fill_missing_values(method = "auto") #ffill > bfill
    data_cleansing_end = time.time()
    
    logger.info(f"#### 데이터 정제 Runtime ####")
    logger.info(f"{data_cleansing_end - data_cleansing_start} sec")
    
    ##########################
    ###### 4. 스케일링  ########
    #########################
    if scaler_switch == 1:
        logger.info(f"#### 4-1) 데이터 스케일링")
        scale_start = time.time()
        filled_series = []
        for item_id, time_series in df_ds_autogluon.groupby(level="item_id", sort=False):
            scaled_series = time_series.droplevel("item_id")
            scaled_series['target'] = scaling_value(df = scaled_series,
                                                    col_name = 'target',
                                                    output_dir = f'{base_output_dir}/scaler',
                                                    ric = item_id)
            filled_series.append(pd.concat({item_id: scaled_series},
                                           names=["item_id"]))
        df_ds = TimeSeriesDataFrame(pd.concat(filled_series))

        scale_end = time.time()
        logger.info(f"#### 데이터 스케일링 Runtime ####")
        logger.info(f"{scale_end - scale_start} sec")
    
    else:
        logger.info(f"\n ### No Scaling")
        df_ds = df_ds_autogluon.copy()

    #################################################
    #####   5. 훈련, 테스트 데이터 세트로 분리 및 저장  ######
    #################################################
    logger.info(f"#### 5-1) Autogluon timeseries 업샘플링")
    df_train = df_ds.copy()
    
    # train 데이터 나누기
    for cnt in range(1, num_fold + 1): #위에 것이 fold 1을 담당함
        df_test = df_train
        df_train = df_test.slice_by_timestep(None, -prediction_length)
        df_train.to_csv(f'{base_output_dir}/train/train_fold{cnt}.csv')
        df_test.to_csv(f'{base_output_dir}/test/test_fold{cnt}.csv')
        # logger.info(f"df_test_fold{cnt+1} = df_train_fold{cnt}[df_train_fold{cnt}['ds'] >= {split_end}]")
    
    logger.info(f"\n### End All of data preprocessing")
    
    total_end = time.time()
    print(f"전체 Run time 시간 : {total_end - total_start:.1f} sec({((total_end - total_start)/60):.1f} min)")
