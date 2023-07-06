
import glob
import os
import pandas as pd

from collections import defaultdict
import numpy as np
from collections import Counter

import time
from datetime import datetime as dt
from dateutil.relativedelta import *
import argparse
import json
import boto3
from io import StringIO, BytesIO
import joblib
import sys
import subprocess
import logging
import logging.handlers
import calendar
import tarfile

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

def check_performance_threshold(iput_df : pd.DataFrame,
                                identifier: str,
                                threshold : float = -100):
    tmp = {}
    satisfied_df = iput_df[iput_df['score_val'] > threshold]
    if len(satisfied_df) > 0:
        tmp['identifier'] = identifier
        tmp['model'] = list(satisfied_df['model'])
        tmp['performance'] = list(satisfied_df['score_val'])
    return tmp

def get_model_performance_report(data):
    result = defaultdict(list)
    models_ext = [row["model"] for row in data if row]
    models = [item for sublist in models_ext for item in sublist]
    performance_ext = [row["performance"] for row in data if row]
    performance = [item for sublist in performance_ext for item in sublist]
    
    count_models = Counter(models)
    
    for keys, values in zip(models, performance):
        result[keys].append(values)

    for key, values in result.items():
        result[key] = []
        result[key].append(count_models[key])
        result[key].append(sum(values) / len(values))
        result[key].append(np.std(values))
    
    # 정렬 1순위 : 비즈니스담당자의 Metric에 선정된 Count 높은 순, 2순위: 표준편차가 작은 순(그래서 -처리해줌)
    result = sorted(result.items(), key=lambda k_v: (k_v[1][0], -k_v[1][2]), reverse=True) 
    return result

def register_model_in_aws_registry(model_zip_path: str,
                                   model_package_group_name: str,
                                   model_description: str,
                                   model_tags: dict,############################# parameter 異붽��좉쾬: golden train path�� test path 
                                   model_status: str,
                                   sm_client) -> str:
    create_model_package_input_dict = {
        "ModelPackageGroupName": model_package_group_name,
        "ModelPackageDescription": model_description, # ex AutoGluon - WeightedEnsemble
        "CustomerMetadataProperties": model_tags,
        "ModelApprovalStatus": model_status,
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": '763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/autogluon-inference:0.4-cpu-py38',
                    "ModelDataUrl": model_zip_path
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        }
    }
    create_model_package_response = sm_client.create_model_package(**create_model_package_input_dict)
    model_package_arn = create_model_package_response["ModelPackageArn"]
    return model_package_arn



def register_manifest(source_path,
                      target_path,
                      s3_client,
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
                              BUCKET_NAME_USECASE,
                              qs_client):
    
    ds_list = qs_client.list_data_sources(AwsAccountId='835451595761')
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
        logger.info(f"datasource_id:{datasource_id} manifest : {response}")
    
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
    parser.add_argument('--algorithm_name', type=str, default = 'Autogluon')
    parser.add_argument('--leaderboard_path', type=str, default="/opt/ml/processing/input/leaderboard")   
    parser.add_argument('--model_base_path', type=str)
    parser.add_argument('--manifest_base_path', type=str)
    parser.add_argument('--prediction_base_path', type=str)
    parser.add_argument('--threshold', type=str, default="-100")   
    parser.add_argument('--model_package_group_name', type=str, default = 'po-myrbd-m1')  
    parser.add_argument('--qs_data_name', type=str, default = 'model_result')    

    return parser.parse_args()


if __name__=='__main__':
    ######################################
    ## 커맨드 인자, Hyperparameters 처리 ##
    ######################################
    args = parse_args()
    logger.info("######### Argument Info ####################################")
    logger.info("### start training code")    
    logger.info("### Argument Info ###")
    logger.info(f"args.algorithm_name: {args.algorithm_name}")    
    logger.info(f"args.leaderboard_path: {args.leaderboard_path}")    
    logger.info(f"args.model_base_path: {args.model_base_path}")
    logger.info(f"args.manifest_base_path: {args.manifest_base_path}")
    logger.info(f"args.prediction_base_path: {args.prediction_base_path}")
    logger.info(f"args.threshold: {args.threshold}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
    logger.info(f"args.qs_data_name: {args.qs_data_name}")
  
    algorithm_name = args.algorithm_name
    leaderboard_path = args.leaderboard_path
    model_base_path = args.model_base_path
    manifest_base_path = args.manifest_base_path
    prediction_base_path = args.prediction_base_path
    threshold = float(args.threshold)
    model_package_group_name = args.model_package_group_name
    qs_data_name = args.qs_data_name
                        
    BUCKET_NAME_USECASE = 'po-myrbd-m1'
    lb_list = sorted(os.listdir(leaderboard_path))
    logger.info(f"leaderboard file list in {leaderboard_path}: {lb_list}")
    satisfied_info = []
    train_data_base_path = manifest_base_path
    test_data_base_path = manifest_base_path
    train_replace_dict = {'trained-model' : 'golden-data',
                          'manifest' : 'train'}
    for key in train_replace_dict.keys():
        train_data_base_path = train_data_base_path.replace(key, train_replace_dict[key])
    test_replace_dict = {'trained-model' : 'golden-data',
                          'manifest' : 'test'}
    for key in test_replace_dict.keys():
        test_data_base_path = test_data_base_path.replace(key, test_replace_dict[key])
        
    for idx, f_path in enumerate(lb_list):
        leaderboard = pd.read_csv(f'{leaderboard_path}/{f_path}').sort_values(by = ['score_val', 'score_test'],
                                                                              ascending = False)
        satisfied_info.append(check_performance_threshold(iput_df = leaderboard,
                                                          identifier = f'fold{idx}',
                                                          threshold = threshold))
    model_report = get_model_performance_report(satisfied_info)
    
    if model_report[0][1][0] == len(lb_list): # Fold 내 모든 성능이 비즈니스 담당자가 설정한 값을 만족한다면
        logger.info(f"\n#### Pass the 1st minimum performance valiation")
        manifest_file_path = register_manifest(prediction_base_path, 
                                               manifest_base_path,
                                               s3_client,
                                               BUCKET_NAME_USECASE)
        model_package_arn = register_model_in_aws_registry(model_zip_path = f"{model_base_path}/model.tar.gz",
                                                           model_package_group_name = model_package_group_name,
                                                           model_description = algorithm_name,
                                                           model_tags = {'champion_model' : str(model_report[0][0]),
                                                                         'passed_the_number_of_folds' : str(model_report[0][1][0]),
                                                                         'average_metric' : str(model_report[0][1][1]),
                                                                         'std_metric' : str(model_report[0][1][2]),
                                                                         'train_data' : str(train_data_base_path),
                                                                         'test_data' : str(test_data_base_path),
                                                                        },
                                                           model_status = 'PendingManualApproval',
                                                           sm_client = sm_client)
        logger.info('### Passed ModelPackage Version ARN : {}'.format(model_package_arn))
        res = refresh_of_spice_datasets(user_account_id,
                                        qs_data_name,
                                        manifest_file_path,
                                        BUCKET_NAME_USECASE,
                                        qs_client)
        logger.info('### refresh_of_spice_datasets : {}'.format(res))
    else:
        logger.info(f"\n#### Filtered at 1st valiation")
        model_package_arn = register_model_in_aws_registry(model_zip_path = f"{model_base_path}/model.tar.gz",
                                                           model_package_group_name = model_package_group_name,
                                                           model_description = algorithm_name,
                                                           model_tags = {'champion_model' : str(model_report[0][0]),
                                                                         'passed_the_number_of_folds' : str(model_report[0][1][0]),
                                                                         'average_metric' : str(model_report[0][1][1]),
                                                                         'std_metric' : str(model_report[0][1][2]),
                                                                         'train_data' : str(train_data_base_path),
                                                                         'test_data' : str(test_data_base_path),
                                                                        },
                                                           model_status = 'Rejected',
                                                           sm_client = sm_client)
        logger.info('### Rejected ModelPackage Version ARN : {}'.format(model_package_arn))
