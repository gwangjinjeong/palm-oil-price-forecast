
import argparse
import os
import requests
import tempfile
import subprocess, sys
import json

import glob
import pandas as pd
import joblib # from sklearn.externals import joblib
import pickle
import tarfile # model registry에는 uri만 등록된다.
from io import StringIO, BytesIO

import logging
import logging.handlers
from logging.config import dictConfig

from dateutil.relativedelta import *
from datetime import datetime as dt
import time

import boto3

KST = dt.today() + relativedelta(hours=9)

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default='/opt/ml/processing/input/train')
    parser.add_argument("--test_dir", type=str, default='/opt/ml/processing/input/test')
    parser.add_argument('--output_dir', type = str, default = '/opt/ml/processing/output')
    parser.add_argument('--item', type = str, default = 'FCPOc3')
    parser.add_argument('--target', type = str, default = 'y')
    parser.add_argument('--metric', type = str, default = 'MAPE')    
    parser.add_argument('--quality', type = str, default = 'fast_training')    
    return parser.parse_args()

def create_tarfile(source_dir, output_filename=None):
    ''' create a tarfile from a source directory'''
    if output_filename == None:
        output_filename = "%s/tmptar.tar" %(tempfile.mkdtemp())
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return output_filename 

def make_tarfile(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    return os.path.join(source_dir, output_filename)

if __name__ == "__main__":
    ############################################
    ########## 필요 라이브러리 설치  ###########
    ########################################### 
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'autogluon==0.6.1'])
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    
    ######################################
    ## 커맨드 인자, Hyperparameters 처리 ##
    ######################################
    logger.info("######### Argument Info ####################################")
    logger.info("### start training code")    
    logger.info("### Argument Info ###")
    args = parse_args()
        
    logger.info(f"args.train_dir: {args.train_dir}")   
    logger.info(f"args.test_dir: {args.test_dir}")   
    logger.info(f"args.output_dir: {args.output_dir}")    
    logger.info(f"args.item: {args.item}")   
    logger.info(f"args.target: {args.target}")    
    logger.info(f"args.metric: {args.metric}")   
    logger.info(f"args.quality: {args.quality}")   
    
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir
    prediction_dir = os.path.join(output_dir, 'prediction')
    leaderboard_dir = os.path.join(output_dir, 'leaderboard')
    model_dir = os.path.join(output_dir, 'model')
    
    for path in [prediction_dir, leaderboard_dir, model_dir]:
        if not os.path.exists(path):
            os.mkdir(path)
    item = args.item
    target = args.target
    metric = args.metric
    quality = args.quality
    
    trlist = sorted(os.listdir(train_dir))
    telist = sorted(os.listdir(test_dir))
    
    logger.info(f"the list of train data {trlist}")
    logger.info(f"the list of train data {telist}")
    
    for train_file, test_file in zip(trlist, telist):
        logger.info("### Reading input data")
        logger.info(f"### train data: {train_file}")
        logger.info(f"### test data: {test_file}")
        
        df_train = pd.read_csv(os.path.join(train_dir, train_file))
        df_test = pd.read_csv(os.path.join(test_dir, test_file))      
        # df_train = df_train[df_train['ric'] != 'MCCc3']
        # df_test = df_test[df_test['ric'] != 'MCCc3']
        
        logger.info("### Convert TimeSeriesDataFrame")
        df_train.loc[:, "ds"] = pd.to_datetime(df_train.loc[:, "ds"])
        df_test.loc[:, "ds"] = pd.to_datetime(df_test.loc[:, "ds"])

        tdf_train = TimeSeriesDataFrame.from_data_frame(
            df_train,
            id_column="ric",
            timestamp_column="ds",
        )
        tdf_test = TimeSeriesDataFrame.from_data_frame(
            df_test,
            id_column="ric",
            timestamp_column="ds",
        )

        logger.info("### Show the range of date for training and test")    
        logger.info('Item:', item)
        logger.info('Target:', target)   
        logger.info('Train:',tdf_train.loc[item][target].index.min(),'~',tdf_train.loc[item][target].index.max())
        logger.info('Test:',tdf_test.loc[item][target].index.min(),'~',tdf_test.loc[item][target].index.max())
        logger.info('The number of test data:',len(tdf_test.loc[item][target]))

        logger.info("### Training AutoGluon Model")    
        predictor = TimeSeriesPredictor(
            path = model_dir,
            target = target,
            prediction_length = len(tdf_test.loc[item][target]),
            eval_metric = metric,
        )
        predictor.fit(
            train_data = tdf_train,
            presets = quality
        )
        logger.info("the list of data in model_dir {}".format(os.listdir(model_dir)))
        tar_file_path = make_tarfile(model_dir, f'{model_dir}/model.tar.gz')
        logger.info("Saving model to {}".format(tar_file_path))

        predictor_leaderboard = predictor.leaderboard(tdf_test, silent = True)
        predictor_leaderboard = predictor_leaderboard.sort_values(by = ['score_val', 'score_test'],
                                                                  ascending = False)
        predictor_leaderboard.to_csv(os.path.join(leaderboard_dir,
                                                  f'leaderboard-{test_file}'),
                                     index = False)
        logger.info(f"predictor_leaderboard sample: head(2) \n {predictor_leaderboard.head(2)}")
        
        top_model_name = predictor_leaderboard.loc[0, 'model']
        # second_model_name = predictor_leaderboard.loc[1, 'model']
        
        prediction_ag_model_01 = predictor.predict(data = tdf_train,
                                                   model = top_model_name)
#         prediction_ag_model_02 = predictor.predict(data = tdf_train,
#                                                    model = second_model_name)
        pred_result_01 = pd.merge(tdf_test.loc['FCPOc3']['y'], prediction_ag_model_01.loc['FCPOc3'],
                                  left_index = True, right_index = True, how = 'left')
        pred_result_01.to_csv(os.path.join(prediction_dir,
                                           f'pred-{top_model_name}-{test_file}'))   
        # pred_result_02 = pd.merge(tdf_test.loc['FCPOc3']['y'], prediction_ag_model_02.loc['FCPOc3'],
        #                           left_index = True, right_index = True, how = 'left')
        # pred_result_02.to_csv(os.path.join(prediction_dir,
        #                                    f'pred-{second_model_name}-{test_file}'))   
