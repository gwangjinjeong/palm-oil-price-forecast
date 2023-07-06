import json
import requests
import time
import os
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import date
from dateutil.relativedelta import *
from dateutil.rrule import *
import pandas as pd
import boto3
from io import StringIO


def get_secret():

    secret_name = "dev/ForecastPalmOilPrice"
    region_name = "ap-northeast-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
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


def lambda_handler(event, context):
    keychain = json.loads(get_secret())
    ACCESS_KEY_ID = keychain['AWS_ACCESS_KEY_ID']
    ACCESS_SECRET_KEY = keychain['AWS_ACCESS_SECRET_KEY']
    DATALAKE_BUCKET_NAME = keychain['DATALAKE_BUCKET_NAME']
    BUCKET_NAME_USECASE = keychain['PROJECT_BUCKET_NAME']
    WWO_API_KEY = keychain['WWO_API_KEY']

    folder_name = os.path.join(BUCKET_NAME_USECASE, 'WordWeatherOnlineAPI')

    session = boto3.Session(ACCESS_KEY_ID, ACCESS_SECRET_KEY)
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(DATALAKE_BUCKET_NAME)
    
    weather_key = WWO_API_KEY
    
    try:
        file_names = list(filter(bool, [obj.key.split('/')[1].split('.')[0] for obj in bucket.objects.filter(Prefix=folder_name)])) # ['sarawak','johor','kedah','kelantan','malacca',"negeri sembian",'pahang','penang','perak','perlis','abah','selangor','terengganu']
        for obj in bucket.objects.filter(Prefix=folder_name):
            key = obj.key
            if key.split('/')[2] == '': continue
            body = obj.get()['Body']
            csv_string = body.read().decode('utf-8')
            df_old = pd.read_csv(StringIO(csv_string))
            # 초기값 세팅
            sdate = dt.strptime(df_old.iloc[-1,0], '%Y-%m-%d').date()+relativedelta(days=+1)
            edate = dt.today().date()
            result = {'date' : [],
              'location' : [],
              'latitude' : [],
              'longitude' : [],
              'maxtempC' : [],
              'mintempC' : [],
              'avgtempC' : [],
              'sunHour' : [],
              'precipMM' : [],
              'humidity' : [],
              'summary' : [],
              'iconUrl' : [],
              'sunrise' : [],
              'sunset' : []
             }
            state = key.split('/')[2].split('.')[0]
            csv_buffer = StringIO()
            while(sdate < edate):
                sdate_str = sdate.strftime('%Y-%m-%d')
                edate_str = edate.strftime('%Y-%m-%d')
                base_url = f'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?key={weather_key}&q={state}&format=json&date={sdate_str}&enddate={edate_str}&tp=24&lang=ko&includelocation=yes&extra'
                res = requests.get(base_url)
                data = res.json()
                # 전처리
                location = data['data']['nearest_area'][0]['region'][0]['value']
                latitude = data['data']['nearest_area'][0]['latitude']
                longitude = data['data']['nearest_area'][0]['longitude'] 
                for d in data['data']['weather']:
                    result['date'].append(d['date'])
                    result['location'].append(location)
                    result['latitude'].append(latitude)
                    result['longitude'].append(longitude)
                    result['maxtempC'].append(d['maxtempC'])
                    result['mintempC'].append(d['mintempC'])
                    result['avgtempC'].append(d['avgtempC'])
                    result['sunHour'].append(d['sunHour'])
                    result['precipMM'].append(d['hourly'][0]['precipMM'])
                    result['humidity'].append(d['hourly'][0]['humidity'])
                    result['summary'].append(d['hourly'][0]['lang_ko'][0]['value'])
                    result['iconUrl'].append(d['hourly'][0]['weatherIconUrl'][0]['value'])
                    result['sunrise'].append(d['astronomy'][0]['sunrise'])
                    result['sunset'].append(d['astronomy'][0]['sunset'])
                sdate = sdate+relativedelta(months=+1)
                edate = sdate+relativedelta(day=31)
                print(state,'#START#',sdate,'#END#',edate)
                time.sleep(0.5)
                if edate > dt.today().date():
                    print('edate > dt.today')
                    edate = dt.today().date()
            df_new = pd.DataFrame.from_dict(result,orient='index').T
            df_sum = pd.concat([df_old,df_new],axis=0).reset_index(drop=True)
            dup = df_sum.duplicated(['date'], keep='first')
            print(f'{state} 중복데이터 갯수',dup.sum())
            df_sum.to_csv(csv_buffer,index=False)
            s3_resource.Object(DATALAKE_BUCKET_NAME, folder_name + '/' + state + '.csv').put(Body = csv_buffer.getvalue())
  

    except Exception as e:
        print('Error Message: ',e)
        print("#"*10+"End Of Today's Service"+"#"*10)
        print(f"You will need to start the {state} And")
        print(f"- Start date: {sdate}\n- End date: {edate}")
        print("#"*25)

    return {
    'statusCode': 200,
    'body': json.dumps('Hello from Lambda!')
    }