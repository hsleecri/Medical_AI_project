from alpbench.benchmark.ActiveLearningScenario import ActiveLearningScenario #이건 시나리오 설정하고 데이터 불러오는 거
from ActiveLearnigSceanrio_Kaggle import ActiveLearningScenarioKaggle # 이건 내가 위에 ActiveLearningScenario 클래스에서 openml 데이터셋 대신 캐글 데이터셋 놓는 거
from alpbench.benchmark.ActiveLearningSetting import ActiveLearningSetting
from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Oracle import Oracle
from alpbench.pipeline.QueryStrategy import MarginQueryStrategy
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score



SCENARIO_ID = 1              # 시나리오의 고유 식별자
OPENML_ID = 31               # OpenML에서 데이터셋의 고유 식별자
TEST_SPLIT_SEED = 42         # 테스트 분할을 위한 랜덤 시드
TRAIN_SPLIT_SEED = 43        # 학습 분할을 위한 랜덤 시드
SEED = 44                    # 전체 시나리오의 재현성을 위한 일반적인 시드

SETTING_ID = 1337            # 설정의 고유 식별자
SETTING_NAME = "TestSetting" # 설정의 이름
SETTING_TRAIN_SIZE = 10      # 라벨이 지정된 학습 데이터의 크기
SETTING_TRAIN_TYPE = "absolute" # 학습 데이터 크기의 유형 (절대 크기)
SETTING_TEST_SIZE = 0.3      # 전체 데이터셋에 대한 테스트 데이터의 비율
NUMBER_OF_IT = 10            # 반복 횟수
NUMBER_OF_QUERIES = 5        # 반복 당 쿼리할 샘플 수
FACTOR = -1                  # 작업 종속적 요소 (여기서는 -1로 설정)


# Active Learning에서 제약 조건하고 설계 선택
alsetting = ActiveLearningSetting(
    setting_id=SETTING_ID,
    setting_name=SETTING_NAME,
    setting_labeled_train_size=SETTING_TRAIN_SIZE,
    setting_train_type=SETTING_TRAIN_TYPE,
    setting_test_size=SETTING_TEST_SIZE,
    number_of_iterations=NUMBER_OF_IT,
    number_of_queries=NUMBER_OF_QUERIES,
    factor=FACTOR,
)

# Active Learning 시나리오 정의 + 데이터셋 설정
alscenario = ActiveLearningScenarioKaggle(
    scenario_id=SCENARIO_ID,
    test_split_seed=TEST_SPLIT_SEED,
    train_split_seed=TRAIN_SPLIT_SEED,
    seed=SEED,
    setting=alsetting,
)

X_l, y_l, X_u, y_u, X_test, y_test = alscenario.get_data_split()

# we choose a **random forest** as learning algorithm and **margin sampling** as query strategy

print("define query strategy")
query_strategy = MarginQueryStrategy(42)
print("setup learner")
learner = RF(n_estimators=100)


ALP = ActiveLearningPipeline(
    learner=learner,
    query_strategy=query_strategy,
    init_budget=SETTING_TRAIN_SIZE,
    num_iterations=NUMBER_OF_IT,
    num_queries_per_iteration=NUMBER_OF_QUERIES,
)

oracle = Oracle(X_u, y_u)
print("fit active learning pipeline")
ALP.active_fit(X_l, y_l, X_u, oracle)

y_hat = ALP.predict(X=X_test)
print("final test acc", accuracy_score(y_test, y_hat))