"""
Script for unit testing the functionality in churn_library.py
Test by running 'ipython churn_script_logging_and_tests.py'

Author: Aaron Ward
Date: December 2021
"""

import os
import logging
import churn_library as cls
from constants import cat_columns

logging.basicConfig(
    filename='./logs/test_churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        return data_frame
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_perform_eda(perform_eda, data_frame):
    '''
    test perform eda function

    '''

    perform_eda(data_frame)
    eda_image_path = "./images/eda"

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        assert len(os.listdir(eda_image_path)) != 0
        logging.info("SUCCESS: test_perform_eda passed")
    except AssertionError as err:
        logging.error("ERROR: Images not found in EDA folder after running perform_eda()")
        raise err

def test_encoder_helper(encoder_helper, data_frame):
    '''
    test encoder helper
    '''
    num_cols = len(data_frame.columns)
    data_frame = encoder_helper(data_frame, cat_columns)

    try:
        # Test length of returned columns
        assert len(data_frame.columns) == len(cat_columns) + num_cols
        logging.info("SUCCESS: test_encoder_helper passed")
        return data_frame
    except AssertionError as err:
        logging.error("ERROR: encoder_helper() return incorrect number of columns.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data_frame):
    '''
    test perform_feature_engineering
    '''
    data_list = []
    [data_list.append(item) for item in perform_feature_engineering(data_frame)]

    # ensure no empty data is returned
    try:
        for item in data_list:
            assert item.shape[0] > 0
        logging.info("SUCCESS: test_perform_feature_engineering passed")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return data_list[0], data_list[1], data_list[2], data_list[3]


def test_train_model(train_model, model):
    '''
    test train_models
    '''
    train_model()
    try:
        model.save_model()
        assert os.path.exists(os.path.join(model.model_path,
                                           f"{model.model_name}.pkl"))
        logging.info("SUCCESS: Successfully trained model and saved")
    except AssertionError:
        logging.error("ERROR: Model failed to train correctly, couldn't load.")

if __name__ == "__main__":
    os.environ['QT_QPA_PLATFORM']='offscreen'
    DATA_FRAME = test_import_data(cls.import_data)
    DATA_FRAME['Churn'] = DATA_FRAME['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    test_perform_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)

    MODEL = cls.LinearRegressionModel(X_TRAIN,
                               X_TEST,
                               Y_TRAIN,
                               Y_TEST,
                               model_name="test_lr_model",
                               model_path='./test_output/')
    test_train_model(MODEL.train_model, MODEL)
