from imports import *
from utils import *

from utils_setup import *
from utils_dataset_load_preprocess import *
from utils_active_learning import *

# from utils_TS2NI import *
# from utils_finetuning import *
# from utils_model import *
# from utils_evaluation import *

def main(base_dir='datasets', num_trials=30, num_epochs=1000, seed=42, refining=True):
    """
    ======================================================================
    gpu랑 gpu strategy랑 로그 설정 ---- utils_setup.py
    ======================================================================
    """
    set_seed(seed)
    print("Setup Start.\n"+"="*100)
    setup_gpu(device_id='0')
    setup_logging(tf_log_level='3')
    strategy = setup_strategy('MirroredStrategy')
    print("Setup done successfully.\n"+"="*100)

    """
    ======================================================================
    데이터 로딩 및 전처리---- utils_dataset_load_preprocess.py
    ======================================================================
    """
    print("Dataset loading Start.\n"+"="*100)
    TRAIN_SET_DIR = os.path.join(base_dir, 'Train.csv')
    TEST_SET_DIR = os.path.join(base_dir, 'Test.csv')
    X_labeled, y_labeled = load_csv_dataset(TRAIN_SET_DIR, has_label=True)
    X_unlabeled = load_csv_dataset(TEST_SET_DIR, has_label=False)

    print("Datasets loaded successfully.\n"+"="*100)

    # (X_labeled, y_labeled) 를 8:2로 나누어서 evaluation set을 따로 빼야 할듯.
    INITIAL_SET_RATIO = 0.01
    TRAIN_SET_RATIO = 0.59
    VAL_SET_RATIO = 0.2
    EVAL_SET_RATIO = 0.2

    # (X_label, y_label)을 train set과 evaluation set으로 8:2로 split
    X_labeled_train, X_labeled_eval, y_labeled_train, y_labeled_eval = train_test_split(X_labeled, y_labeled, test_size=EVAL_SET_RATIO, stratify=y_labeled,shuffle=True, random_state=seed)
    # (X_train, y_train)을 train set과 validation set으로 3:1로 split (train:val:eval = 6:2:2)
    X_labeled_train, X_labeled_val, y_labeled_train, y_labeled_val = train_test_split(X_labeled_train, y_labeled_train, test_size=VAL_SET_RATIO, stratify=y_labeled_train,shuffle=True, random_state=seed)


    X_labeled_train_num, X_labeled_train_cat = split_features(X_labeled_train)
    X_labeled_val_num, X_labeled_val_cat = split_features(X_labeled_val)
    X_labeled_eval_num, X_labeled_eval_cat = split_features(X_labeled_eval)

    X_labeled_train_num_scaled = scaling_features(numerical_features=X_labeled_train_num, scaler_type='MinMaxScaler', verbose=False)
    
    X_labeled_train_num_scaled, X_labeled_eval_num_scaled= scaling_features(X_labeled_train_num, X_labeled_eval_num, scaler_type= 'MinMaxScaler', verbose=True)
    X_labeled_val_num_scaled = scaling_features(X_labeled_val_num, None, scaler_type = 'MinMaxScaler')

    X_labeled_train_combined = concatenate_features(X_labeled_train_num_scaled, X_labeled_train_cat)

    X_labeled_train_selected = select_important_features(X_labeled_train_combined, y_labeled_train,model_type='RandomForest',num_features=10, verbose=True) # 근데 사실 feature selection은 필요없을지도.

    # initial training 위해서 데이터셋 조금 split 하기
    X_labeled_train, X_labeled_val, y_labeled_train, y_labeled_val = train_test_split(X_labeled_train, y_labeled_train, test_size=VAL_SET_RATIO, stratify=y_labeled_train,shuffle=True, random_state=seed)
    
    print("Datasets preprocessed successfully.\n"+"="*100)

    """
    어떻게 data split을 해야할지 생각 좀 해보자.
    semi-supervised를 어떻게 생각할지도 생각해보자.
    """

    

    initial_model = train_initial_model(X_labeled_selected, y_labeled, 'XGBoost', seed, num_epochs, verbose=True)

    
    """
    ======================================================================
    Active Learning 루프 ---- utils_active_learning.py
    ======================================================================
    """

    

    """
    코드 해야 하는 것 좀 정리
    1. dataset loading

    2. 전처리
        2.1. numerical features + categorical features 로 나누기
        2.2. numerical features를 minmaxscaler로 전처리하기
        2.3. 나눴던 feature 합치기
        2.4. feature dimension이 너무 크니까 feature selection하기 (method로, feature selectiton 방법까지도 e.g., random forest, feature importance, 이건 할지 안할지 비교해보기)

    3. Active Learning 루프
        3.0. 모델 build 한다음
        3.1. 초기 모델 학습 (initial training)
        3.2. 불확실성 평가 (uncertainty evaluation)
        3.3. 레이블을 요구할 데이터 포인트 선택 data point selection 
        3.4. 라벨링 및 모델 업데이트 (labeling and model retraining)
        3.5. iteration

    4. Classification

    목표가 뭐야?
    human-in-the-loop을 통해 제조 데이터 labeling cost 낮춤으로써 제조시스템 effeciency, performance 향상

    제조에서는 unstructured data보단, structured data인 tabular data가 많음

    data 특성= 어렵다: tabular data(structured data), high-dimensional, numerical features + categorical features(one-hot encoded)

    다양한 labeling cost를 낮추기 위해 active learning을 가져옴


    """
    
    # """
    # ======================================================================
    # 시계열 데이터를 나이브 이미지로 바꾸는 과정(TS2NI) ---- utils_TS2NI.py
    # ======================================================================
    # """
    # print("="*100+"\nPreprocessing Start.\n"+"="*100)
    # # Image resizing ratio
    # R=1
    # beta_list = [3,4,5,6]


    # # The directory to save the preprocessed datasets
    # TS2NI_DIR = f"datasets_TS2NI/{dataset_name}"

    # # Preprocess and save datasets
    # save_TS2NI_pickle(TS2NI_DIR, X_train, max_length, num_variables, beta_list, R, 'train', seed)
    # save_TS2NI_pickle(TS2NI_DIR, X_test, max_length, num_variables, beta_list, R, 'test', seed) # 이거 나중에 refining에서 trial때 생긴 beta로 하는 게 좋은데 그냥 지금 여기서 만드는 게 나을 듯 생각해보니

    # """
    # 지금은 뒤에 fine_tuning 코드 쓰고 있어서
    # validation set을 설정하지 않고 그냥 train set과 test set으로 하지만
    # 나중에 다르게 피드백 오면
    # validation set도 TS2NI 하도록 고쳐야 함
    # """

    # print("Preprocessing completed successfully.\n"+"="*100)

    


    # """
    # ======================================================================
    # 하이퍼파라미터 fine-tuning 하는 과정 ---- utils_finetuning.py
    # ======================================================================
    # """
    # SEEDS_REFINING = [1,2,3]
    # IMAGE_SIZE = (max_length, max_length, num_variables) # train set의 이미지의 input size. cnn model의 input shape으로 들어감

    # #얘도 나중에 다르게 피드백 오면 dataset 별로 statistic이 다 다르니까 (max_length가 train set, validation set, test_set 마다 다 다름) 조정해야 할 수 있음

    # if refining == True:
    #     print("Fine-tuning Start.\n"+"="*100)
    #     study = optuna.create_study(direction='maximize')

    #     try:
    #         # Ensure the optimization is done within the strategy scope
            
    #         study.optimize(lambda trial: objective_multi(trial, input_shape=IMAGE_SIZE, num_class=num_classes, seeds_refining=SEEDS_REFINING, dataset_dir=TS2NI_DIR, num_epochs=num_epochs,y_train=y_train, y_test=y_test, strategy=strategy),
    #                     n_trials=num_trials, catch=(Exception,), gc_after_trial=True)
    #     except Exception as e:
    #         print(f"An unexpected error occurred during the study optimization: {e}")
    #     finally:
    #         tf.keras.backend.clear_session()
    #         gc.collect()
        
    #         print("Preprocessing completed successfully.\n"+"="*100)
    #     best_params = study.best_trial.params
    #     print("Best trial parameters:", best_params)

    #     """
    #     ======================================================================
    #     refined model로 evaluation 하는 과정 ---- utils_evaluation.py
    #     ======================================================================
    #     """
    
    #     # 마지막에 seed랑 모델이랑 데이터셋 csv에 작성하도록하자.
    #     print("Evaluation Start.\n"+"="*100)
    #     results = evaluate_model_with_best_params(best_params, input_shape=IMAGE_SIZE, num_class=num_classes, seed=seed, dataset_dir=TS2NI_DIR, num_epochs=num_epochs, y_train=y_train, y_test=y_test, strategy=strategy)
    #     print("Results:", results)

    #     save_results_to_excel(results, best_params, excel_dir='results_xlsx/TS2NI', dataset_name=dataset_name)

    # else:
    #     # 여기에 model loading 하고, train set이랑 test set 바로 넣어서 하는 게 좋을 거 같은데
    #     pass
        

if __name__ == "__main__":
    base_dir = 'datasets'
    num_trials = 150
    num_epochs = 500
    seed = 42
    refining = True

    main(base_dir,num_trials,num_epochs,seed,refining)