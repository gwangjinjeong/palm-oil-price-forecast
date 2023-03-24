
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
from datetime import datetime as dt

import boto3


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

def get_bucket_key_from_uri(uri):
    uri_aws_path = uri.split('//')[1]
    uri_bucket = uri_aws_path.rsplit('/')[0]
    uri_file_path = '/'.join(uri_aws_path.rsplit('/')[1:])
    return uri_bucket, uri_file_path

def get_secret():
    secret_name = "dev/ForecastPalmOilPrice"
    region_name = "ap-northeast-2"
    
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name,
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
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
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type = str, default = "/opt/ml/processing/output", help='예측 결과값이 저장되는 곳, Inference 결과가 저장된다.')
    parser.add_argument('--ml_algorithm_name', type=str, default = 'Autogluon')  
    parser.add_argument('--model_package_group_name', type=str, default = BUCKET_NAME_USECASE)  
    parser.add_argument('--qs_data_name', type=str, default = 'forecast result')    

    return parser.parse_args()
        
if __name__=='__main__':
    ########################################### 
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'autogluon==0.6.1'])
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    
    ############################################
    ###### Secret Manager에서 키값 가져오기  #######
    ########################################### 
    logger.info(f"\n### Loading Key value from Secret Manager")
    keychain = json.loads(get_secret())
    ACCESS_KEY_ID = keychain['AWS_ACCESS_KEY_ID']
    ACCESS_SECRET_KEY = keychain['AWS_ACCESS_SECRET_KEY']
    BUCKET_NAME_USECASE = keychain['PROJECT_BUCKET_NAME']
    DATALAKE_BUCKET_NAME = keychain['DATALAKE_BUCKET_NAME']
    S3_PATH_REUTER = keychain['S3_PATH_REUTER']
    S3_PATH_WWO = keychain['S3_PATH_WWO']
    S3_PATH_STAGE = keychain['S3_PATH_STAGE']
    S3_PATH_GOLDEN = keychain['S3_PATH_GOLDEN']
    S3_PATH_TRAIN = keychain['S3_PATH_TRAIN']
    S3_PATH_FORECAST = keychain['S3_PATH_PREDICTION']
    
    boto3_session = boto3.Session(aws_access_key_id = ACCESS_KEY_ID,
                                  aws_secret_access_key = ACCESS_SECRET_KEY,
                                  region_name = 'ap-northeast-2')
    
    s3_client = boto3_session.client('s3')
    sm_client = boto3_session.client('sagemaker')
    qs_client = boto3_session.client('quicksight')

    sts_client = boto3_session.client("sts")
    user_account_id = sts_client.get_caller_identity()["Account"]
    ######################################
    ## 커맨드 인자, Hyperparameters 처리 ##
    ######################################  
    logger.info("######### Argument Info ####################################")
    logger.info("### start training code")    
    logger.info("### Argument Info ###")
    args = parse_args()             
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.ml_algorithm_name: {args.ml_algorithm_name}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
    logger.info(f"args.qs_data_name: {args.qs_data_name}")
# prediction_output_path = f"s3://crude-palm-oil-prices-forecast/predicted-data/2023/03/19/1679292475.0/result"

    output_dir = args.output_dir
    ml_algorithm_name = args.ml_algorithm_name
    model_package_group_name = args.model_package_group_name
    qs_data_name = args.qs_data_name

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
    train_data_dir = model_spec['CustomerMetadataProperties']['train_data']
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
    df_train = pd.read_csv(os.path.join(train_data_dir, 'train_fold1.csv'))
    df_test = pd.read_csv(os.path.join(test_data_dir, 'test_fold1.csv'))
    sum_df = pd.concat([df_train, df_test]).reset_index(drop = True)
    sum_df.loc[:, "ds"] = pd.to_datetime(sum_df.loc[:, "ds"])
    
    tdf_train = TimeSeriesDataFrame.from_data_frame(
        sum_df,
        id_column="ric",
        timestamp_column="ds",
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
    logger.info(f"prediction_ag_model sample: head(2) \n {prediction_ag_model.head(2)}")
    prediction_result = prediction_ag_model.loc['FCPOc3']
    logger.info(f"prediction_ag_model sample: head(2) \n {prediction_result.head(2)}")
    prediction_result.to_csv(f'{result_dir}/prediction_result.csv')
