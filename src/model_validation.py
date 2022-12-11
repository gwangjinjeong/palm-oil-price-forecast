
import glob
import os
import pandas as pd
import time
from datetime import datetime as dt
import argparse
import json
import boto3
from io import StringIO, BytesIO
import joblib
import sys
import subprocess
import logging
import logging.handlers

import tarfile


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

def get_secret():
    secret_name = "prod/sagemaker"
    region_name = "ap-northeast-2"

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

def convert_series_to_description(leaderboard : pd.Series):
    return ','.join(leaderboard.loc[0,['model','score_test','score_val']].to_string().split())

def get_bucket_key_from_uri(uri):
    uri_aws_path = uri.split('//')[1]
    uri_bucket = uri_aws_path.rsplit('/')[0]
    uri_file_path = '/'.join(uri_aws_path.rsplit('/')[1:])
    return uri_bucket, uri_file_path

if __name__=='__main__':
    ################################
    ###### 커맨드 인자 파싱   ##########
    ################################    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_input_path', type=str, default="/opt/ml/processing/input")   
    parser.add_argument('--s3_model_uri', type=str, default="/opt/ml/processing/model")   
    parser.add_argument('--model_package_group_name', type=str, default='palm-oil-price-forecast')   
    args = parser.parse_args()     

    logger.info("######### Argument Info ####################################")
    logger.info(f"args.base_input_path: {args.base_input_path}")
    logger.info(f"args.s3_model_uri: {args.s3_model_uri}")
    logger.info(f"args.model_package_group_name: {args.model_package_group_name}")
    
    base_input_path = args.base_input_path
    s3_model_uri = args.s3_model_uri
    model_package_group_name = args.model_package_group_name
    
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
    
    ############################################
    ##### Model, Leaderboard 파일 가져오기 #####
    ########################################### 
    logger.info(f"\n### Loading Model, Leaderboard zip files ")
    logger.info(f"\n#### Extract output.tar.gz and Read a Leaderboard ")
    ## 22.11.29 추가: 이전 step인, step_train에서 model.tar.gz의 uri는 가져올 수 있었지만, output.tar.gz는 못가져왔다. 이를 model.tar.gz에서 output.tar.gz으로 바꾸는방식으로 우회하자
    leaderboard_uri = s3_model_uri.replace('model.tar.gz','output.tar.gz')#,f'{base_input_path}/output.tar.gz'
    logger.info(f"\n#### output.tar.gz uri : {leaderboard_uri}")
    output_bucket, output_key = get_bucket_key_from_uri(leaderboard_uri)  
    output_obj = s3_client.get_object(Bucket = output_bucket, Key = output_key)
   
    logger.info("\n######### Model zip file extraction ####################################")
    with tarfile.open(fileobj=output_obj['Body'], mode='r|gz') as file:
        file.extractall(base_input_path)    
    logger.info(f"file list in {base_input_path}: {os.listdir(base_input_path)}")        
    
    # if leaderboard_path.endswith("tar.gz"):
    #     tar = tarfile.open(leaderboard_path, "r:gz")
    #     tar.extractall(base_input_path)
    #     tar.close()
    # elif leaderboard_path.endswith("tar"):
    #     tar = tarfile.open(leaderboard_path, "r:")
    #     tar.extractall(base_input_path)
    #     tar.close()

    leaderboard = pd.read_csv(f'{base_input_path}/leaderboard.csv').sort_values(by = ['score_val', 'score_test'],
                                                                                ascending = False)
    logger.info(f"leaderboard train sample: head(5) \n {leaderboard.head()}")
    logger.info(f"\n#### Set  ")
    model_package_group_name = model_package_group_name
    modelpackage_inference_specification =  {
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": '763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/autogluon-inference:0.4-cpu-py38',
                    "ModelDataUrl": s3_model_uri#'#args.model_path_uri
                }
            ],
            "SupportedContentTypes": [ "text/csv" ],
            "SupportedResponseMIMETypes": [ "text/csv" ],
        }
    }
    if len(leaderboard[leaderboard['score_val'] > -0.13]) > 0:
        logger.info(f"\n#### Pass the first performance filtering")
        
        create_model_package_input_dict = {
            "ModelPackageGroupName" : model_package_group_name,
            "ModelPackageDescription" : convert_series_to_description(leaderboard),
            "ModelApprovalStatus" : "PendingManualApproval"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)
        create_model_package_response = sm_client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]
        logger.info('### Passed ModelPackage Version ARN : {}'.format(model_package_arn))
        
    else:
        logger.info(f"\n#### None of them passed the filtering")
        create_model_package_input_dict = {
            "ModelPackageGroupName" : model_package_group_name,
            "ModelPackageDescription" : convert_series_to_description(leaderboard),
            "ModelApprovalStatus" : "Rejected"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)
        create_model_package_response = sm_client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]
        logger.info('### Rejected ModelPackage Version ARN : {}'.format(model_package_arn))
