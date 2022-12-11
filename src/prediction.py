
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


def get_bucket_key_from_uri(uri):
    uri_aws_path = uri.split('//')[1]
    uri_bucket = uri_aws_path.rsplit('/')[0]
    uri_file_path = '/'.join(uri_aws_path.rsplit('/')[1:])
    return uri_bucket, uri_file_path

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
        
if __name__=='__main__':
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'autogluon==0.6.0'])
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    ############################################
    ###### Secret Manager에서 키값 가져오기  #######
    ########################################### 
    logger.info(f"\n### Loading Key value from Secret Manager")
    
    keychain = json.loads(get_secret())
    ACCESS_KEY_ID = keychain['ACCESS_KEY_ID_ent']
    ACCESS_SECRET_KEY = keychain['ACCESS_SECRET_KEY_ent']

    BUCKET_NAME_USECASE = keychain['BUCKET_NAME_USECASE_ent']
    S3_PATH_STAGE = keychain['S3_PATH_STAGE']
    S3_PATH_GOLDEN = keychain['S3_PATH_GOLDEN']
    S3_PATH_TRAIN = keychain['S3_PATH_TRAIN']
    S3_PATH_log = keychain['S3_PATH_LOG']
    boto3_session = boto3.Session(ACCESS_KEY_ID, ACCESS_SECRET_KEY)

    region = boto3_session.region_name

    s3_resource = boto3_session.resource('s3')
    s3_client = boto3_session.client('s3')
    sm_client = boto3.client('sagemaker',
                             aws_access_key_id = ACCESS_KEY_ID,
                             aws_secret_access_key = ACCESS_SECRET_KEY,
                             region_name = 'ap-northeast-2')
    
    ################################
    ###### 커맨드 인자 파싱   ##########
    ################################    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_input_dir', type=str, default="/opt/ml/processing/input", help='train,testset 불러오는곳')
    parser.add_argument('--output_dir', type = str, default = "/opt/ml/processing/output", help='예측 결과값이 저장되는 곳, test dataset과 prediction 결과가 merge되서 저장된다.')
    parser.add_argument('--model_package_group_name', type=str, default='palm-oil-price-forecast')   
    args = parser.parse_args()     
    logger.info("\n######### Argument Info ####################################")
    logger.info(f"args.base_input_dir: {args.base_input_dir}")
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")

    base_input_dir = args.base_input_dir
    output_dir = args.output_dir
    model_package_group_name = args.model_package_group_name
    model_dir = '/opt/ml/model'
    
    ##########################################################
    ###### 적합한 모델의 URI 찾고, 탑 성능 모델 이름 가져오기 ##########
    #########################################################
    logger.info("\n######### Finding suitable model uri ####################################")
    logger.info(f"Model Group name: {model_package_group_name}")
    model_registry_list = sm_client.list_model_packages(ModelPackageGroupName = model_package_group_name)['ModelPackageSummaryList']
    for model in model_registry_list:
        if (model['ModelPackageGroupName'] == model_package_group_name and
            model['ModelApprovalStatus'] == 'Approved'):
            mr_arn = model['ModelPackageArn']
            break
    describe_model = sm_client.describe_model_package(ModelPackageName=mr_arn)
    s3_model_uri = describe_model['InferenceSpecification']['Containers'][0]['ModelDataUrl']
    top_model_name = describe_model['ModelPackageDescription'].split(',')[1]

    logger.info(f"Found suitable model uri: {s3_model_uri}")
    logger.info(f"And top model name: {top_model_name}")
    
    logger.info("\n#########Download suitable model file  ####################################")
    model_bucket, model_key = get_bucket_key_from_uri(s3_model_uri)  
    model_obj = s3_client.get_object(Bucket = model_bucket, Key = model_key)
    
    ##########################################################
    ###### 모델 압축 풀고 TimeseriesDataFrame으로 변환 ##########
    #########################################################
    logger.info("\n######### Model zip file extraction ####################################")
    with tarfile.open(fileobj=model_obj['Body'], mode='r|gz') as file:
        file.extractall(model_dir)    
    logger.info(f"list in /opt/ml/model: {os.listdir(model_dir)}")        
    
    logger.info("\n######### Convert df_test dataframe into TimeSeriesDataFrame  ###########")        
    df_train = pd.read_csv(os.path.join(f'{base_input_dir}/train/train.csv'))
    df_train.loc[:, "ds"] = pd.to_datetime(df_train.loc[:, "ds"])
    tdf_train = TimeSeriesDataFrame.from_data_frame(
        df_train,
        id_column="ric",
        timestamp_column="ds",
    )
    df_test = pd.read_csv(f"{base_input_dir}/test/test.csv")
    df_test.loc[:, "ds"] = pd.to_datetime(df_test.loc[:, "ds"])
    tdf_test = TimeSeriesDataFrame.from_data_frame(
        df_test,
        id_column="ric",
        timestamp_column="ds",
    )
    logger.info(f"df_test sample: tail(2) \n {tdf_train.tail(2)}")
    logger.info(f"df_test sample: head(2) \n {tdf_test.head(2)}")
    
    ################################
    ###### Prediction 시작 ##########
    ###############################
    logger.info("\n######### Start prediction  ###########")        
    loaded_trainer = pickle.load(open(f"{model_dir}/models/trainer.pkl", 'rb'))
    logger.info(f"loaded_trainer: {loaded_trainer}")
    prediction_ag_model = loaded_trainer.predict(data = tdf_train,
                                                 model = top_model_name)
    logger.info(f"prediction_ag_model sample: head(2) \n {prediction_ag_model.head(2)}")

    prediction_result = pd.merge(tdf_test.loc['FCPOc3']['y'], prediction_ag_model.loc['FCPOc3'],
                                 left_index = True, right_index = True, how = 'left')
    prediction_result.to_csv(f'{output_dir}/prediction_result.csv')
