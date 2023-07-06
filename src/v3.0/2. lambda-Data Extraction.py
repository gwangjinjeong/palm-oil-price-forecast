import json
import boto3
import pandas as pd
from io import StringIO
import os
import time
from datetime import datetime as dt
import datetime
from dateutil.relativedelta import *

import boto3
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

s3_client_me = boto3.client('s3')
sts_client = boto3.client('sts')
another_account = sts_client.assume_role(
    RoleArn="arn:aws:iam::718639997651:role/send-data-to-ns-di-palm",
    RoleSessionName="cross_account_lambda"
    )

ACCESS_KEY = another_account['Credentials']['AccessKeyId']
SECRET_KEY = another_account['Credentials']['SecretAccessKey']
SESSION_TOKEN = another_account['Credentials']['SessionToken']

# create service client using the assumed role credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    aws_session_token=SESSION_TOKEN,
    )


def read_file(bucket, file_location):
    """
    S3 버킷에서 파일을 읽어 pandas DataFrame으로 변환하는 함수.

    Parameters:
    - bucket: S3 버킷 이름
    - file_location: 읽어올 파일의 위치

    Returns:
    - df: 파일 내용을 담은 pandas DataFrame
    """
    file_obj = s3_client.get_object(Bucket=bucket, Key=file_location)
    file_content = file_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(file_content))

    return df

def combine_multipart_files(s3_client, bucket, prefix):
    """
    S3 버킷에서 멀티파트 CSV 파일을 읽어 결합하는 함수.

    Parameters:
    - bucket: S3 버킷 이름
    - prefix: S3에서 파일을 찾는 데 사용할 경로 접두사

    Returns:
    - combined_df: 결합된 pandas DataFrame
    """
    # Boto3 S3 클라이언트 초기화

    # 파일 위치 목록 생성
    # response = s3_client.list_objects_v2(Bucket = bucket, Prefix=prefix)
    # file_locations = [item['Key'] for item in response['Contents']]
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    file_locations = []
    for page in page_iterator:
        file_locations += [keys['Key'] for keys in page['Contents']]
    # ThreadPoolExecutor를 사용하여 파일을 병렬로 읽습니다.
    with ThreadPoolExecutor(max_workers=10) as executor:
        dataframes = list(executor.map(read_file, [bucket]*len(file_locations), file_locations))
    print(len(dataframes))
    # 읽어온 모든 DataFrame을 결합합니다.
    combined_df = pd.concat(dataframes)
    return combined_df



#destination : staged-data
def lambda_handler(event, context):
    ###################################
    ###### 1. Data Integration  #######
    ################################### 
    total_start = time.time()
    start = time.time()
    s3_bucket = 'palmoil-price-forecast-prd'
    prefix = 'EikonDataAPI/'
    df_integrated = combine_multipart_files(s3_client, s3_bucket, prefix)
    end = time.time()
    print('Data fetching time')
    print(end-start)
    print('length of file')
    print(len(df_integrated))
    
    ###################################
    ###### 2. Save Data  ##############
    ###################################
    csv_buffer = StringIO()
    df_integrated.to_csv(csv_buffer, index=False)
    destination_bucket = 'staged-data'
    KST = dt.today() + relativedelta(hours=9)
    yyyy, mm, dd = str(KST.year), str(KST.month).zfill(2), str(KST.day).zfill(2)
    timestamp = time.mktime(KST.timetuple())
    file_path = f'{yyyy}/{mm}/{dd}/{timestamp}'
    res = s3_client_me.put_object(Bucket = destination_bucket,
                                  Key = f'{file_path}/integrated.csv',
                                  Body = csv_buffer.getvalue())
    print(res)
    print(KST)