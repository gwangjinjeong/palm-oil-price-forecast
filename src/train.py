
import os
import sys
import pickle

import argparse
import pandas as pd
import json

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import joblib # from sklearn.externals import joblib

import logging
import logging.handlers

from dateutil.relativedelta import *
from datetime import datetime as dt

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


if __name__ == "__main__":
    ###################################
    ## 커맨드 인자, Hyperparameters 처리 ##
    ###################################        

    logger.info(f"### start training code")    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type = str, default = os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--output_data_dir', type = str, default = os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type = str, default = os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_dir', type = str, default = os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test_dir', type = str, default = os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--item', type = str, default = 'FCPOc3')
    parser.add_argument('--target', type = str, default = 'y')
    parser.add_argument('--metric', type = str, default = 'MAPE')    
    parser.add_argument('--quality', type = str, default = 'low_quality')
    args = parser.parse_args()     

    logger.info("### Argument Info ###")
    logger.info(f"args.output_dir: {args.output_dir}")
    logger.info(f"args.output_data_dir: {args.output_data_dir}")    
    logger.info(f"args.model_dir: {args.model_dir}")        
    logger.info(f"args.train_dir: {args.train_dir}")   
    logger.info(f"args.test_dir: {args.test_dir}")   
    logger.info(f"args.item: {args.item}")   
    logger.info(f"args.target: {args.target}")    
    logger.info(f"args.metric: {args.metric}")   
    logger.info(f"args.quality: {args.quality}")   
    
    output_dir = args.output_dir
    output_data_dir = args.output_data_dir
    model_dir = args.model_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    item = args.item
    target = args.target
    metric = args.metric
    quality = args.quality
    
    logger.info("### Reading input data")
    df_train= pd.read_csv(os.path.join(train_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(test_dir, 'test.csv'))        
    
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
    logger.info('Item:\t', item)
    logger.info('Target:\t', target)   
    logger.info('Train:\t',tdf_train.loc[item][target].index.min(),'~',tdf_train.loc[item][target].index.max())
    logger.info('Test:\t',tdf_test.loc[item][target].index.min(),'~',tdf_test.loc[item][target].index.max())
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
    logger.info("Saving model to {}".format(model_dir))
    
    # 원래라면 Validation dataset이 input으로 들어와서 leaderboard와 prediction을 해야한다.
    # 근데, 여기서는 아니다. 이번 사이클에서는 test data까지 모두 산출한다음에 넣는것으로 진행하자.
    predictor_leaderboard = predictor.leaderboard(tdf_test, silent = True)
    predictor_leaderboard.to_csv(os.path.join(output_data_dir,'leaderboard.csv'), index = False)
