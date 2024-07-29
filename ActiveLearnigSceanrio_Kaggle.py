import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from alpbench.benchmark.ActiveLearningSetting import ActiveLearningSetting

def create_dataset_split(
    X, y, test_split_seed, test_split_size: float, train_split_seed, train_split_size, train_split_type, factor
):
    """데이터를 라벨된 데이터, 라벨되지 않은 데이터 및 테스트 데이터로 나눕니다. 
    분할 유형은 절대 값(즉, 고정된 라벨된 데이터 포인트 수) 또는 상대 값(즉, 학습 데이터의 고정 비율)일 수 있습니다.
    분할은 라벨에 따라 계층화됩니다. 라벨된 데이터는 각 클래스의 인스턴스를 최소 하나씩 포함하도록 보장됩니다.
    또한, 주어진 경우 요소(factor)에 따라 라벨된 데이터 포인트의 수가 클래스 수 곱하기 요소로 결정됩니다.

    Args:
        X (numpy.ndarray): 데이터
        y (numpy.ndarray): 라벨
        test_split_seed (int): 테스트 분할을 위한 시드
        test_split_size (float): 테스트 데이터의 크기
        train_split_seed (int): 학습 분할을 위한 시드
        train_split_size (float): 라벨된 학습 데이터의 크기
        train_split_type (str): 크기 매개변수의 유형: 데이터 포인트 수 또는 (학습) 데이터셋의 비율
        factor (int): 작업 종속 요소

    Returns:
        labeled_indices (list): 라벨된 데이터의 인덱스
        test_indices (list): 테스트 데이터의 인덱스
    """

    # 인덱스 목록 초기화
    indices = np.arange(0, len(X))

    # 데이터를 학습 및 테스트로 분할하고 나중에 반환할 테스트 인덱스를 가져옵니다.
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=test_split_size, random_state=test_split_seed, stratify=y
    )
    
    # 라벨되지 않은 데이터의 비율을 결정합니다.
    unlabeled_size = 1 - train_split_size
    if train_split_type == "absolute":
        if factor != -1:
            train_split_size = factor * len(np.unique(y))
            unlabeled_size = 1 - train_split_size / len(X_train)
        else:
            unlabeled_size = 1 - train_split_size / len(X_train)

    # 데이터를 라벨된 데이터와 라벨되지 않은 데이터로 분할합니다.
    X_l, X_u, y_l, y_u, labeled_indices, unlabeled_indices = train_test_split(
        X_train, y_train, train_indices, test_size=unlabeled_size, random_state=train_split_seed, stratify=y_train
    )

    # 라벨된 데이터에 각 클래스가 최소 하나 포함되도록 보장합니다.
    if len(np.unique(y[labeled_indices])) != len(np.unique(y)):
        for i in np.unique(y):
            if i not in y_l:
                ids = np.where(y_u == i)[0]
                np.random.seed(train_split_seed)
                idx_in_yu = np.random.choice(ids)
                idx = unlabeled_indices[idx_in_yu]
                labeled_indices = np.append(labeled_indices, idx)

    assert len(np.unique(y[labeled_indices])) == len(
        np.unique(y)
    ), "Not all classes are represented in the labeled data"

    return labeled_indices.tolist(), test_indices.tolist()

class ActiveLearningScenarioKaggle:
    """능동 학습 시나리오

    능동 학습 시나리오는 하나의 능동 학습 설정의 데이터와 설정을 정의합니다.
    시나리오는 데이터셋의 OpenML ID, 테스트 분할, 학습 분할 및 재현성을 위한 시드, 설정, 그리고 선택적으로 라벨된 데이터와 테스트 인덱스로 초기화됩니다.

    Args:
        scenario_id (int): 데이터베이스에서 시나리오의 고유 식별자
        test_split_seed (int): 테스트 분할을 위한 시드
        train_split_seed (int): 학습 분할을 위한 시드
        seed (int): 재현성을 위한 시드
        setting (ActiveLearningSetting): 능동 학습 설정
        labeled_indices (list): 라벨된 데이터의 인덱스
        test_indices (list): 테스트 데이터의 인덱스

    Attributes:
        scenario_id (int): 데이터베이스에서 시나리오의 고유 식별자
        test_split_seed (int): 테스트 분할을 위한 시드
        train_split_seed (int): 학습 분할을 위한 시드
        seed (int): 재현성을 위한 시드
        setting (ActiveLearningSetting): 능동 학습 설정
        labeled_indices (list): 라벨된 데이터의 인덱스
        test_indices (list): 테스트 데이터의 인덱스
    """

    def __init__(
        self,
        scenario_id,
        test_split_seed,
        train_split_seed,
        seed,
        setting: ActiveLearningSetting,
        labeled_indices: list = None,
        test_indices: list = None,
    ):
        self.scenario_id = scenario_id
        self.test_split_seed = test_split_seed
        self.train_split_seed = train_split_seed
        self.seed = seed
        self.labeled_indices = labeled_indices
        self.test_indices = test_indices
        self.setting = setting


        # 대체 데이터셋 사용 (수정된 코드)
        # 사용자 데이터셋을 여기에 로드합니다 (예: CSV 파일 로드)
        import pandas as pd
        df = pd.read_csv('datasets/Train.csv')
        X = df.iloc[:, :-1].values  # 마지막 열을 제외한 모든 열을 X로 사용
        y = df.iloc[:, -1].values  # 마지막 열을 y로 사용
        
        # 라벨 인코딩
        if y.dtype != int:
            y_int = np.zeros(len(y)).astype(int)
            vals = np.unique(y)
            for i, val in enumerate(vals):
                mask = y == val
                y_int[mask] = i
            y = y_int
        X = OrdinalEncoder().fit_transform(X)

        # 중복 제거
        _, unique_indices = np.unique(X, axis=0, return_index=True)

        self.X = X[unique_indices]
        self.y = LabelEncoder().fit_transform(y)[unique_indices]

        if test_indices is None or labeled_indices is None:
            self.labeled_indices, self.test_indices = create_dataset_split(
                self.X,
                self.y,
                test_split_seed,
                setting.setting_test_size,
                train_split_seed,
                setting.setting_labeled_train_size,
                setting.setting_train_type,
                setting.factor,
            )

    def get_scenario_id(self):
        """
        시나리오 ID 가져오기
        """
        return self.scenario_id


    def get_setting(self):
        """
        설정 가져오기
        """
        return self.setting

    def get_seed(self):
        """
        시드 가져오기
        """
        return self.seed

    def get_labeled_instances(self):
        """
        라벨된 인스턴스 가져오기
        """
        return self.labeled_indices

    def get_test_indices(self):
        """
        테스트 인덱스 가져오기
        """
        return self.test_indices

    def get_labeled_train_data(self):
        """
        라벨된 학습 데이터 가져오기
        """
        return self.X[self.labeled_indices], self.y[self.labeled_indices]

    def get_unlabeled_train_data(self):
        """
        라벨되지 않은 학습 데이터 가져오기 (X 및 y)
        """
        combined_train_labeled_test = self.labeled_indices + self.test_indices
        mask = np.array([True] * len(self.X))
        mask[combined_train_labeled_test] = False
        return self.X[mask], self.y[mask]

    def get_test_data(self):
        """
        테스트 데이터 가져오기
        """
        return self.X[self.test_indices], self.y[self.test_indices]

    def get_data_split(self):
        """
        라벨된 데이터, 라벨되지 않은 데이터 및 테스트 데이터 가져오기
        """
        X_l, y_l = self.get_labeled_train_data()
        X_u, y_u = self.get_unlabeled_train_data()
        X_test, y_test = self.get_test_data()
        return X_l, y_l, X_u, y_u, X_test, y_test

    def __repr__(self):
        params = dict(self.__dict__)
        params.pop("X")
        params.pop("y")
        return "<ActiveLearningScenario> " + str(params)
