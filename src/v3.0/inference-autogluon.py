
import argparse
import os
import requests
import tempfile
import subprocess, sys
import json

import glob
import pandas as pd
import joblib
import pickle
import tarfile
from io import StringIO, BytesIO

import logging
import logging.handlers

import time
from dateutil.relativedelta import *
from datetime import datetime as dt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

import boto3

KST = dt.today() + relativedelta(hours=9)

ACCESS_KEY_ID = 'AKIA4FBGLVPYQAWRMRZ5'
ACCESS_SECRET_KEY = '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'
boto3_session = boto3.Session(aws_access_key_id = ACCESS_KEY_ID,
                              aws_secret_access_key = ACCESS_SECRET_KEY,
                              region_name = 'ap-northeast-2')

s3_client = boto3_session.client('s3')
sm_client = boto3_session.client('sagemaker')
qs_client = boto3_session.client('quicksight')    
sts_client = boto3_session.client("sts")
user_account_id = sts_client.get_caller_identity()["Account"]

def _get_logger():
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  
logger = _get_logger()

def get_bucket_key_from_uri(uri):
    uri_aws_path = uri.split('//')[1]
    uri_bucket = uri_aws_path.rsplit('/')[0]
    uri_file_path = '/'.join(uri_aws_path.rsplit('/')[1:])
    return uri_bucket, uri_file_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type = str, default = "/opt/ml/processing/output", help='예측 결과값이 저장되는 곳, Inference 결과가 저장된다.')
    parser.add_argument('--ml_algorithm_name', type=str, default = 'Autogluon')
    parser.add_argument('--item', type = str, default = 'PO-MYRBD')
    parser.add_argument('--model_package_group_name', type=str, default = 'PO-MYRBD')  
    parser.add_argument('--qs_data_name', type=str, default = 'forecast result')    
    return parser.parse_args()
        
if __name__=='__main__':
    logger.info("############## Argument Info ###########################")
    logger.info("### start training code")    
    logger.info("### Argument Info ###")
    args = parse_args()
    logger.info(f"args.item: {args.item}")
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.ml_algorithm_name: {args.ml_algorithm_name}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
    logger.info(f"args.qs_data_name: {args.qs_data_name}")

    output_dir = args.output_dir
    ml_algorithm_name = args.ml_algorithm_name
    model_package_group_name = args.model_package_group_name
    qs_data_name = args.qs_data_name
    item = args.item
    item = item.upper()
    
    #### 폴더가 존재하지 않아 생기는 문제 해소를 위해 생성 ####
    model_dir = f'{output_dir}/model'
    os.makedirs(model_dir, exist_ok=True)
    result_dir = f'{output_dir}/result'
    os.makedirs(result_dir, exist_ok=True)
    manifest_base_path = f'{output_dir}/manifest'
    os.makedirs(manifest_base_path, exist_ok=True)
    
    ##########################################################
    ###### 적합한 모델의 URI 찾고, 탑 성능 모델 이름 가져오기 ##########
    #########################################################
    logger.info("\n######### Finding suitable model uri ####################################")
    model_registry_list = sm_client.list_model_packages(ModelPackageGroupName = model_package_group_name)['ModelPackageSummaryList']
    for model in model_registry_list:
        if (model['ModelPackageGroupName'] == model_package_group_name and
            model['ModelApprovalStatus'] == 'Approved' and
            model['ModelPackageDescription'] == ml_algorithm_name):
            mr_arn = model['ModelPackageArn']
            break
            
    model_spec = sm_client.describe_model_package(ModelPackageName=mr_arn)
    test_data_dir = model_spec['CustomerMetadataProperties']['test_data']
    
    s3_model_uri = model_spec['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    champion_model = model_spec['CustomerMetadataProperties']['champion_model']

    logger.info(f"Found suitable model uri: {s3_model_uri}")
    logger.info(f"And top model name: {champion_model}")
    
    logger.info("\n#########Download suitable model file  ####################################")
    model_bucket, model_key = get_bucket_key_from_uri(s3_model_uri)  
    logger.info(f"\nmodel_bucket: {model_bucket}, model_key: {model_key}  ####################################")
    model_obj = s3_client.get_object(Bucket = model_bucket, Key = model_key)
    
    ##########################################################
    ###### 모델 압축 풀고 TimeseriesDataFrame으로 변환 ##########
    #########################################################
    
    logger.info("\n######### Model zip file extraction ####################################")
    with tarfile.open(fileobj=model_obj['Body'], mode='r|gz') as file:
        file.extractall(output_dir)
    logger.info(f"list in {model_dir}: {os.listdir(model_dir)}")
    
    logger.info("\n######### Convert df_test dataframe into TimeSeriesDataFrame  ###########")        
    sum_df = pd.read_csv(os.path.join(test_data_dir, 'test_fold1.csv'))
    sum_df.loc[:, "timestamp"] = pd.to_datetime(sum_df.loc[:, "timestamp"])
    
    tdf_train = TimeSeriesDataFrame.from_data_frame(
        sum_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    logger.info(f"sum_df sample: head(2) \n {sum_df.head(2)}")
    logger.info(f"sum_df sample: tail(2) \n {sum_df.tail(2)}")

    ################################
    ###### Prediction 시작 ##########
    ###############################
    logger.info("\n######### Start prediction  ###########")        
    loaded_trainer = pickle.load(open(f"{model_dir}/models/trainer.pkl", 'rb'))
    logger.info(f"loaded_trainer: {loaded_trainer}")
    prediction_ag_model = loaded_trainer.predict(data = tdf_train,
                                         model = champion_model)
    logger.info(f"pred result sample: head(2) \n {prediction_ag_model.head(2)}")
    if item[:3].upper() == 'PO-':
        polist = ['PO-MYRBD-M1','PO-MYRBD-M2','PO-MYRBD-M3','PO-MYRBD-Q1','PO-MYRBD-Q2','PO-MYRBD-Q3']
        df_po = pd.DataFrame()
        for po_myrbd in polist:
            pred_result = prediction_ag_model.loc[po_myrbd]
            pred_result = pred_result.iloc[-30:, :].reset_index()
            pred_result = pred_result.assign(item = po_myrbd)
            df_po = pd.concat([df_po, pred_result])
        pred_result = df_po
    else:
        pred_result = prediction_ag_model.loc[item]
        pred_result = pred_result_all.iloc[-30:,:].reset_index()
        pred_result = pred_result.assign(item = po_myrbd)
        
    champion_model = champion_model.replace('/',' trial')
    logger.info(f"pred result sample: head(2) \n {pred_result.head(2)}")
    logger.info(f"pred result sample: tail(2) \n {pred_result.tail(2)}")
    pred_result.to_csv(os.path.join(result_dir,
                                    f'prediction_result.csv'),
                       index=False)   
