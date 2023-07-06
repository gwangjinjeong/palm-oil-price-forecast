import json
import boto3
import time
import sys
def lambda_handler(event, context):
    # TODO implement

    client = boto3.client('quicksight')
    response = client.create_ingestion(DataSetId='<dataset-id>',IngestionId='<ingetion-id>',AwsAccountId='<aws-account-id>')
    while True:
        response = client.describe_ingestion(DataSetId='<dataset-id>',IngestionId='<ingetion-id>',AwsAccountId='<aws-account-id>')
            if response['Ingestion']['IngestionStatus'] in ('INITIALIZED', 'QUEUED', 'RUNNING'):
                time.sleep(10) 
            elif response['Ingestion']['IngestionStatus'] == 'COMPLETED':
                print("refresh completed. RowsIngested {0}, RowsDropped {1}, IngestionTimeInSeconds {2}, IngestionSizeInBytes {3}".format(
                    response['Ingestion']['RowInfo']['RowsIngested'],
                    response['Ingestion']['RowInfo']['RowsDropped'],
                    response['Ingestion']['IngestionTimeInSeconds'],
                    response['Ingestion']['IngestionSizeInBytes']))
                break
            else:
                print("refresh failed! - status {0}".format(response['Ingestion']['IngestionStatus']))
                sys.exit(1)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
