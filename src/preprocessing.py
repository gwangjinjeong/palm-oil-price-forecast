
import argparse
import os
import requests
import tempfile
import subprocess, sys

import pandas as pd
import numpy as np
from glob import glob
import copy
from collections import OrderedDict
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

import logging
import logging.handlers

import json
import base64
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

import time
from datetime import datetime as dt
import datetime
from pytz import timezone
from dateutil.relativedelta import *

###############################
######### 전역변수 설정 ##########
###############################
KST = dt.today() + relativedelta(hours=9)
ric_list = ['BOc1', 'BOc2', 'BOc3','BOPLKL','BRRTSc1', 'BRRTSc2', 'BRRTSc3', 'CAD=', 'EUR=', 'JPY=', 'KRW=', 'MYR=', 'GBP=', 'INR=','Cc1', 'Cc2', 'Cc3','CCMc1', 'CCMc2', 'CCMc3',
            'CLc1', 'CLc2', 'CLc3','CNY=','COMc1', 'COMc2','COMc3','CTc1', 'CTc2', 'CTc3', 'DJCI', 'DJCIBR', 'DJCICL', 'DJCICN', 'DJCIEN', 'DJCIGR', 'DJCIIA', 'DJCING', 
            'DJCISO', 'DJCIWH', 'DJT','FCHI','FCPOc1', 'FCPOc2', 'FCPOc3','FGVHKL',
            'FKLIc1', 'FKLIc2', 'FKLIc3','FTSE','GCc1', 'GCc2', 'GCc3','GDAXI','GENMKL','HSI','IOIBKL','IXIC','JNIc1','JNIc2','JNIc3','KCc1', 'KCc2', 'KCc3','KLKKKL','KLSE','KQ11', 'KS11',
            'KWc1', 'KWc2', 'KWc3','LCOc1', 'LCOc2', 'LCOc3','LWBc1', 'LWBc2', 'LWBc3','MCCc1', 'MCCc2', 'MCCc3','MXSCKL','Oc1', 'Oc2', 'Oc3','PEPTKL','RRc1', 'RRc2', 'RRc3','RSc1', 'RSc2', 'RSc3',
            'Sc1', 'Sc2', 'Sc3','SIMEKL','SOPSKL','SSEC', 'THPBKL', 'Wc1', 'Wc2', 'Wc3'
           ]

col_names_asis = ['ds','high','low','open','ric']
col_names_tobe = ['ds','high','low','open','y']

###############################
######### util 함수 설정 ##########
###############################
def _get_logger():
    '''
    로깅을 위해 파이썬 로거를 사용
    # https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
    '''
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  
logger = _get_logger()

def download_object(file_name):
    try:
        s3_client = boto3.client("s3")
        download_path = Path('test') / file_name.replace('/','_')
        s3_client.download_file(
            BUCKET_NAME_USECASE,
            file_name,
            str(download_path)
        )
        return "Success"
    except Exception as e:
        return e

def download_parallel_multiprocessing(path_list):
    with ProcessPoolExecutor() as executor:
        future_to_key = {executor.submit(download_object, key): key for key in path_list}
        for future in futures.as_completed(future_to_key):
            key = future_to_key[future]
            exception = future.exception()
            if not exception:
                yield key, future.result()
            else:
                yield key, exception
                                
def get_list_in_s3(key_id : str,
                   secret_key_id : str,
                   bucket_name : str,
                   s3_path : str) -> list:
    
    s3 = boto3.client('s3',
                      aws_access_key_id = ACCESS_KEY_ID,
                      aws_secret_access_key = ACCESS_SECRET_KEY,
                      region_name = 'ap-northeast-2')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket = bucket_name,
                               Prefix = s3_path)  # 원하는 bucket 과 하위경로에 있는 object list # dict type
    contents_list = [] # object list의 Contents를 가져옴
    for page in pages:
        for obj in page['Contents']:
            contents_list.append(obj)
    return contents_list

def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**default_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders


def download_files(s3_client, bucket_name, local_path, file_names, folders):

    local_path = Path(local_path)

    for folder in folders:
        folder_path = Path.joinpath(local_path, folder)
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        file_path = Path.joinpath(local_path, file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )
        
def get_dataframe(base_preproc_input_dir, file_name_prefix ):    
    '''
    파일 이름이 들어가 있는 csv 파일을 모두 저장하여 데이터 프레임을 리턴
    '''
    
    input_files = glob('{}/{}*.csv'.format(base_preproc_input_dir, file_name_prefix))
    #claim_input_files = glob('{}/dataset*.csv'.format(base_preproc_input_dir))    
    logger.info(f"input_files: \n {input_files}")    
    
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(base_preproc_input_dir, "train"))
        
    raw_data = [ pd.read_csv(file, index_col=0) for file in input_files ]
    df = pd.concat(raw_data)
   
    logger.info(f"dataframe shape \n {df.shape}")    
    logger.info(f"dataset sample \n {df.head(2)}")        
    #logger.info(f"df columns \n {df.columns}")    
    
    return df

def get_secret():
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    secret_name = "prod/sagemaker"
    region_name = "ap-northeast-2"
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    try:
        get_secret_value_response = client.get_secret_value(
            SecretId='prod/sagemaker',
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException': # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException': # An error occurred on the server side.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException': # You provided an invalid value for a parameter.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException': # You provided a parameter value that is not valid for the current state of the resource.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException': # We can't find the resource that you asked for.
            raise e
    else:
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return secret
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return decoded_binary_secret
        
def fill_missing_dates(df_in : pd.DataFrame,
                       freq : str
                      ) -> pd.DataFrame : 
    df = df_in.copy()
    if df["ds"].dtype == np.int64:
            df.loc[:, "ds"] = df.loc[:, "ds"].astype(str)
    df.loc[:, "ds"] = pd.to_datetime(df.loc[:, "ds"])
    r = pd.date_range(start = df["ds"].min(),
                      end = df["ds"].max(),
                      freq = freq)
    df = df.set_index("ds").reindex(r).rename_axis("ds").reset_index()
    return df

def fill_missing_price_value(df: pd.DataFrame, col: str, limit_linear : int = 20 ) -> pd.DataFrame :
    initial_is_na = sum(df[col].isnull())
    series = df.loc[:, col].astype(float)
    series = series.interpolate(method="linear", limit=limit_linear, limit_direction="both")
    series = [0 if v < 0 else v for v in series]
    df[col] = series
    return df

def scaling_value(df : pd.DataFrame,
                  col_name : str,
                  ric,
                  s3_resource,
                  BUCKET_NAME_USECASE,
                  S3_PATH_GOLDEN) -> tuple:

    series = df[col_name].values
    scaler = MinMaxScaler()
    series = series.reshape(-1,1)
    scaler.fit(series)
    series = scaler.transform(series)
    with tempfile.TemporaryFile() as fp:
        joblib.dump(scaler, fp)
        fp.seek(0)
        s3_resource.put_object(Body = fp.read(),
                               Bucket = BUCKET_NAME_USECASE,
                               Key = f"{S3_PATH_GOLDEN}/{KST.strftime('%Y/%m/%d')}/scaler-files/{ric}_{col_name}_scaler.pkl")
    return series

def convert_type(raw, cols, type_target):
    '''
    해당 데이터 타입으로 변경
    '''
    df = raw.copy()
    
    for col in cols:
        df[col] = df[col].astype(type_target)
    
    return df

if __name__=='__main__':
    ################################
    ###### 커맨드 인자 파싱   ##########
    ################################
    split_date_default = dt.today() + relativedelta(hours = 9) - relativedelta(months=1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_output_dir', type=str, default="/opt/ml/processing/output")
    parser.add_argument('--base_preproc_input_dir', type=str, default="/opt/ml/processing/input")   
    parser.add_argument('--split_date', type=str, default=split_date_default.strftime('%Y-%m-%d'))       
    parser.add_argument('--label_column', type=str, default="ric") 
    parser.add_argument("--scaler_switch", type = str, default = 1, help = '1이면 Scaling ON, 0이면 Scaling OFF')
        
    # parse arguments
    args = parser.parse_args()     

    logger.info("######### Argument Info ####################################")
    logger.info(f"args.base_output_dir: {args.base_output_dir}")
    logger.info(f"args.base_preproc_input_dir: {args.base_preproc_input_dir}")    
    logger.info(f"args.label_column: {args.label_column}")        
    logger.info(f"args.split_date: {args.split_date}")   
    logger.info(f"args.scaler_switch: {args.scaler_switch}")   
    
    base_output_dir = args.base_output_dir
    base_preproc_input_dir = args.base_preproc_input_dir
    label_column = args.label_column
    split_date = args.split_date    
    scaler_switch = int(args.scaler_switch)
    ############################################
    ###### Secret Manager에서 키값 가져오기  #######
    ########################################### 
    logger.info(f"\n### Loading the key value using Secret Manager")

    keychain = json.loads(get_secret())
    ACCESS_KEY_ID = keychain['ACCESS_KEY_ID_ent']
    ACCESS_SECRET_KEY = keychain['ACCESS_SECRET_KEY_ent']

    BUCKET_NAME_USECASE = keychain['BUCKET_NAME_USECASE_ent']
    S3_PATH_STAGE = keychain['S3_PATH_STAGE']
    S3_PATH_GOLDEN = keychain['S3_PATH_GOLDEN']
    S3_PATH_TRAIN = keychain['S3_PATH_TRAIN']
    S3_PATH_log = keychain['S3_PATH_LOG']

    boto_session = boto3.Session(ACCESS_KEY_ID, ACCESS_SECRET_KEY)
    region = boto_session.region_name
    s3_resource = boto_session.resource('s3')
    s3_client = boto_session.client('s3')
    ############################################
    ###### 1. 데이터 Integration  #######
    ########################################### 
    total_start = time.time()
    start = time.time()
    stage_dir = f'{base_output_dir}/stage/stage.csv"'
    logger.info(f"\n### Data Integration")
    path_list = []
    df_sum = pd.DataFrame()

    for (path, dir, files) in os.walk(base_preproc_input_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                path_list.append("%s/%s" % (path, filename))
                
    logger.info(f"The number for data : {len(path_list)}")
    for file in path_list:
        df_tmp= pd.read_csv(file, encoding='utf-8') 
        df_sum = pd.concat([df_sum, df_tmp])
    df_sum = df_sum.sort_values(by='Date').reset_index(drop=True)
    df_sum.to_csv(f"{base_output_dir}/stage/stage.csv", index = False)
    end = time.time()
    
    logger.info(f"Data Integration is done")
    logger.info(f"Runtime : {end - start:.1f} sec({((end - start)/60):.1f} min)")
    logger.info(f"The number for data : {len(path_list)}")
    logger.info(f"Integrated data sample: head(2) \n {df_sum.head(2)}")
    logger.info(f"Integrated data sample: tail(2) \n {df_sum.tail(2)}")
    
    #################################
    ####   2. 첫번쨰 전처리 단계     ####
    ####   품목선별, 열 삭제, 형변환  ####
    ################################    
    start = time.time()
    logger.info(f"\n ### RIC Item selection")    
    df_sum = df_sum[df_sum['RIC'].isin(ric_list)].reset_index()
    logger.info(f"The number for data after RIC Item selection : {df_sum.shape}")

    logger.info(f"\n ### Column selection")    
    df_sum = df_sum[['Date','HIGH', 'LOW', 'OPEN', 'CLOSE','RIC']]
    logger.info(f"The number for data after Column selection : {df_sum.shape}")
    logger.info(f"\n ### type conversion")    
    df_sum.loc[:, "Date"] = pd.to_datetime(df_sum.loc[:, "Date"])
    df_sum.loc[:, "HIGH"] = df_sum.loc[:, "HIGH"].astype(np.float32)
    df_sum.loc[:, "LOW"] = df_sum.loc[:, "LOW"].astype(np.float32)
    df_sum.loc[:, "OPEN"] = df_sum.loc[:, "OPEN"].astype(np.float32)
    df_sum.loc[:, "CLOSE"] = df_sum.loc[:, "CLOSE"].astype(np.float32)
    ####################################################
    ####   3. Autogluon timeseries 데이터 셋으로 만들기  ####
    ####################################################
    logger.info(f"\n ### Autogluon timeseriesdataframe Conversion")        
    df_list = OrderedDict()
    for name in ric_list:
        df_tmp = df_sum[df_sum['RIC'] == name]
        df_tmp = df_tmp.drop('RIC', axis=1)
        df_list[name] = df_tmp[df_tmp['Date'] >= '2014-07-02'].reset_index(drop = True)
    ####################################################
    ############   4. 열 이름 변경, 결측치 처리  ############
    ###################################################
    logger.info(f"\n ### Rename columns")        
    col_names = ['ds','high','low','open','y']
    for name, value in df_list.items():
        df_list[name].columns = col_names

    logger.info(f"\n ### Fill missing value (Date)")        
    for name, value in df_list.items():
        df_list[name]  = fill_missing_dates(value, 'B')
        num_added = len(df_list[name]) - len(value)
        is_na = sum(df_list[name]['y'].isnull())
    
    logger.info(f"\n ### Fill missing value (Price)")        
    for name, value in df_list.items():
        df_proc1 = fill_missing_price_value(value, 'y')
        df_proc1 = fill_missing_price_value(value, 'high')
        df_proc1 = fill_missing_price_value(value, 'low')
        df_proc1 = fill_missing_price_value(value, 'open')
        df_list[name] = df_proc1
        
    ####################################################
    #################   5. Scaling  ###################
    ###################################################
    if int(scaler_switch) == 1:
        logger.info(f"\n ### Scaling")            
        scale_dir = f"{base_output_dir}/scaler-files"
        os.makedirs(scale_dir, exist_ok=True)
        for name, value in df_list.items():
            for col in ['y','high','open','low']:
                value.loc[:, col] = scaling_value(value, col, name, s3_client, BUCKET_NAME_USECASE, S3_PATH_GOLDEN)
            df_list[name] = value
    else:
        logger.info(f"\n ### No Scaling")
    end = time.time()
    logger.info(f"\n### All Date Transform is done")
    print(f"All Date Transform Run time : {end - start:.1f} sec({((end - start)/60):.1f} min)")

    #################################################
    #####   6. 훈련, 테스트 데이터 세트로 분리 및 저장  ######
    #################################################
    logger.info(f"\n ### Split train, test dataset")            
    df_golden = pd.DataFrame()
    for name, value in df_list.items():
        value = value.assign(ric = name)
        df_golden = pd.concat([df_golden, value])
    df_golden = df_golden.reset_index(drop = True)
    
    # train 데이터 나누기
    df_train = df_golden[df_golden['ds'] < split_date]
    df_train.to_csv(f"{base_output_dir}/train/train.csv", index = False)
    
    df_test = df_golden[df_golden['ds'] >= split_date]
    df_test.to_csv(f"{base_output_dir}/test/test.csv", index = False)
    
    logger.info(f"\n ### Final result for train dataset ")
    logger.info(f"\n ####preprocessed train shape \n {df_train.shape}")        
    logger.info(f"preprocessed train sample: head(2) \n {df_train.head(2)}")
    logger.info(f"preprocessed train sample: tail(2) \n {df_train.tail(2)}")
    
    logger.info(f"\n ####preprocessed test shape \n {df_test.shape}")            
    logger.info(f"preprocessed test sample: head(2) \n {df_test.head(2)}")
    logger.info(f"preprocessed test sample: tail(2) \n {df_test.tail(2)}")

    logger.info(f"\n### End All of data preprocessing")
    total_end = time.time()
    print(f"Run time 시간 : {total_end - total_start:.1f} sec({((total_end - total_start)/60):.1f} min)\n")
    
