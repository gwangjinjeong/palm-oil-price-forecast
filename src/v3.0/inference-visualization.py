
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
import calendar
from datetime import datetime as dt

import boto3

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


###############################
######### util 함수 설정 ##########
###############################
def _get_logger():
    loglevel = logging.DEBUG
    l = logging.getLogger(__name__)
    if not l.hasHandlers():
        l.setLevel(loglevel)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))        
        l.handler_set = True
    return l  
logger = _get_logger()
def register_manifest(source_path,
                      target_path,
                      BUCKET_NAME_USECASE):
    template_json = {"fileLocations": [{"URIPrefixes": []}],
                     "globalUploadSettings": {
                         "format": "CSV",
                         "delimiter": ","
                     }}
    paginator = s3_client.get_paginator('list_objects_v2')
    response_iterator = paginator.paginate(Bucket = BUCKET_NAME_USECASE,
                                           Prefix = source_path.split(BUCKET_NAME_USECASE+'/')[1]
                                          )
    for page in response_iterator:
        logger.info(f"\n#### page {page}")
        for content in page['Contents']:
            template_json['fileLocations'][0]['URIPrefixes'].append(f's3://{BUCKET_NAME_USECASE}/'+content['Key'])
    with open(f'./manifest_testing.manifest', 'w') as f:
        json.dump(template_json, f, indent=2)

    res = s3_client.upload_file('./manifest_testing.manifest',
                                BUCKET_NAME_USECASE,
                                f"{target_path.split(BUCKET_NAME_USECASE+'/')[1]}/visual_validation.manifest")
    return f"{target_path.split(BUCKET_NAME_USECASE+'/')[1]}/visual_validation.manifest"
    

def refresh_of_spice_datasets(user_account_id,
                              qs_data_name,
                              manifest_file_path,
                              BUCKET_NAME_USECASE):
    
    ds_list = qs_client.list_data_sources(AwsAccountId=user_account_id)
    datasource_ids = [summary["DataSourceId"] for summary in ds_list["DataSources"] if qs_data_name in summary["Name"]]    
    for datasource_id in datasource_ids:
        response = qs_client.update_data_source(
            AwsAccountId=user_account_id,
            DataSourceId=datasource_id,
            Name=qs_data_name,
            DataSourceParameters={
                'S3Parameters': {
                    'ManifestFileLocation': {
                        'Bucket': BUCKET_NAME_USECASE,
                        'Key':  manifest_file_path
                    },
                },
            })
        logger.info(f"datasource_id:{datasource_id} 의 manifest를 업데이트: {response}")
    
    res = qs_client.list_data_sets(AwsAccountId = user_account_id)
    datasets_ids = [summary["DataSetId"] for summary in res["DataSetSummaries"] if qs_data_name in summary["Name"]]
    ingestion_ids = []

    for dataset_id in datasets_ids:
        try:
            ingestion_id = str(calendar.timegm(time.gmtime()))
            qs_client.create_ingestion(DataSetId = dataset_id,
                                       IngestionId = ingestion_id,
                                       AwsAccountId = user_account_id)
            ingestion_ids.append(ingestion_id)
        except Exception as e:
            logger.info(e)
            pass
    for ingestion_id, dataset_id in zip(ingestion_ids, datasets_ids):
        while True:
            response = qs_client.describe_ingestion(DataSetId = dataset_id,
                                                    IngestionId = ingestion_id,
                                                    AwsAccountId = user_account_id)
            if response['Ingestion']['IngestionStatus'] in ('INITIALIZED', 'QUEUED', 'RUNNING'):
                time.sleep(5)     #change sleep time according to your dataset size
            elif response['Ingestion']['IngestionStatus'] == 'COMPLETED':
                print("refresh completed. RowsIngested {0}, RowsDropped {1}, IngestionTimeInSeconds {2}, IngestionSizeInBytes {3}".format(
                    response['Ingestion']['RowInfo']['RowsIngested'],
                    response['Ingestion']['RowInfo']['RowsDropped'],
                    response['Ingestion']['IngestionTimeInSeconds'],
                    response['Ingestion']['IngestionSizeInBytes']))
                break
            else:
                logger.info("refresh failed for {0}! - status {1}".format(dataset_id,
                                                                          response['Ingestion']['IngestionStatus']))
                break
    return response
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, help='prediction_data')
    parser.add_argument("--qs_data_name", type=str, default='forecast_result')
    parser.add_argument('--model_package_group_name', type=str, default = BUCKET_NAME_USECASE)  
    return parser.parse_args()
        
if __name__=='__main__':
    ############################################
    ###### Secret Manager에서 키값 가져오기  #######
    ########################################### 
    logger.info(f"\n### Loading Key value from Secret Manager")
    BUCKET_NAME_USECASE = 'po-myrbd-m1'

    
    ######################################
    ## 커맨드 인자, Hyperparameters 처리 ##
    ######################################  
    logger.info("######### Argument Info ####################################")
    logger.info("### start training code")    
    logger.info("### Argument Info ###")
    args = parse_args()             
    logger.info(f"args.source_path: {args.source_path}")
    logger.info(f"args.qs_data_name: {args.qs_data_name}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
 
    source_path = args.source_path
    qs_data_name = args.qs_data_name    
    model_package_group_name = args.model_package_group_name
    
    target_path = source_path.rsplit('/', 1)[0]+'/manifest'
    logger.info(f"\n#### target_path : {target_path}")

    logger.info(f"\n#### register_manifest")
    manifest_file_path = register_manifest(source_path, 
                                           target_path,
                                           BUCKET_NAME_USECASE)
    logger.info(f'### manifest_file_path : {manifest_file_path}')
    logger.info(f"\n#### refresh_of_spice_datasets")
    res = refresh_of_spice_datasets(user_account_id,
                                    qs_data_name,
                                    manifest_file_path,
                                    BUCKET_NAME_USECASE)
    logger.info(f'### refresh_of_spice_datasets : {res}')
