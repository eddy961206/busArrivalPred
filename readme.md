# 딥러닝 기반 통행 시간 예측 프로젝트

## 프로젝트 개요

본 프로젝트는 딥러닝 및 머신러닝 기술을 활용하여 대중교통, 특히 버스의 구간 통행 시간을 정확하게 예측하는 것을 목표로 합니다.  도시 교통 체증 완화, 버스 운행 효율성 증대, 그리고 궁극적으로 시민들의 대중교통 이용 편의성 향상에 기여하고자 합니다. LSTM (Long Short-Term Memory) 순환 신경망과 XGBoost 머신러닝 모델을 사용하여 시간 순서에 따른 데이터의 패턴을 학습하고, 다양한 요인들을 고려하여 통행 시간 예측 모델을 개발했습니다.

**주요 목표:**

* **통행 시간 예측 정확도 향상:**  LSTM 및 XGBoost 모델을 통해 기존 예측 모델 대비 높은 정확도의 통행 시간 예측 시스템 구축.
* **실시간 교통 정보 활용:**  실시간 버스 운행 데이터 및 교통 정보를 모델에 반영하여 예측 정확도를 실시간으로 개선.
* **다양한 교통 상황 고려:**  요일, 시간, 날씨, 교통 혼잡도 등 다양한 요인을 모델 학습에 반영하여 실제 교통 상황에 더욱 근접한 예측 수행.
* **확장 가능한 예측 시스템 구축:**  향후 지하철, 택시 등 다른 대중교통 수단에도 적용 가능한 확장성 있는 예측 시스템 개발 기반 마련.

## 코드 구조

프로젝트 코드는 기능별 및 모델별로 체계적으로 구성되어 있습니다.

- **`src/train/route/`**:  주요 통행 시간 예측 모델 학습 코드가 위치합니다.
    - `travel_LSTM.py`: 기본적인 LSTM 모델을 사용하여 통행 시간을 예측하는 파이썬 스크립트입니다. 데이터셋 클래스 정의, LSTM 모델 구조 설계, 학습 및 검증, 모델 평가 과정을 포함합니다.
    - `travel_LSTM_label.py`: `travel_LSTM.py`를 개선한 버전으로, 범주형 변수 (버스 노선 ID, 구간 ID 등)에 Label Encoding을 적용하여 모델의 입력 데이터 처리 방식을 최적화합니다. 시퀀스 데이터 학습을 위한 데이터셋 구성 방식이 구현되어 있습니다.
    - `travel_XGBoost.py`: XGBoost 모델을 활용하여 통행 시간을 예측하는 코드입니다. LSTM 모델과의 성능 비교를 위해 개발되었으며, 전통적인 머신러닝 기법의 적용 가능성을 탐색합니다.

- **`src/train/ML/`**: 머신러닝 모델 학습 관련 코드를 포함합니다.
    - `travel_XGBoost.py` (위와 동일): XGBoost 모델 학습 코드가 위치합니다.

- **`legacy/`**:  과거 버전 코드, 실험적 코드, 더 이상 사용되지 않는 코드 등이 보관되어 있습니다.
    - `travel_LSTM_old.py`: 이전 버전의 LSTM 모델 코드입니다.
    - `travel_LSTM.py`: `legacy` 디렉토리에도 LSTM 모델 코드가 존재하며, 오래된 버전일 가능성이 있습니다.
    - `travel_time.py`: 초기 단계의 통행 시간 예측 모델 코드일 수 있습니다.
    - `travel_time_optuna.py`: Optuna를 사용한 하이퍼파라미터 최적화 실험 관련 코드입니다.
    - `travel_time_optuna_hash.py`: 해시 함수를 적용한 Optuna 하이퍼파라미터 최적화 실험 코드입니다.
    - `holding_time_optuna_hash.py`:  정류장 **정차 시간** 예측 관련 Optuna 실험 코드일 수 있습니다 (파일명으로 추정).

- **`dataset/`**: (`.gitignore`에 의해 버전 관리에서 제외) 실제 모델 학습 및 평가에 사용된 데이터셋 CSV 파일들이 위치합니다.  이 디렉토리에는 실제 데이터 파일이 포함되어야 하지만, 버전 관리에서 제외되어 있으므로 별도로 준비해야 합니다.
    - 예시 파일명: `소통시간학습_수요일_filtered_train.csv`, `travel소통시간학습_수요일_filtered_train.csv`, `travel소통시간학습_수요일_filtered_val.csv`, `travel소통시간학습_수요일_filtered_test.csv`

- **`.gitignore`**: 버전 관리에서 제외할 파일 및 디렉토리 목록을 정의합니다. `dataset/`, `__pycache__/`, `.idea/` 등이 제외되어 있습니다.
- **`README.md`**: 프로젝트 개요, 코드 구조, 실행 방법 등을 설명하는 문서 (현재 파일).

## 데이터셋

본 프로젝트는 실제 버스 운행 데이터를 기반으로 통행 시간 예측 모델을 학습하고 평가합니다.

**데이터 구성:**

데이터셋은 CSV (Comma Separated Values) 파일 형태로 구성되어 있으며, 주요 컬럼은 다음과 같습니다.

* **DAY_TYPE**: 요일 (1: 월요일, 2: 화요일, ..., 7: 일요일). 범주형 변수이며, 모델 학습 시 One-Hot Encoding 또는 Label Encoding 등의 전처리 과정을 거칩니다.
* **BUSROUTE_ID**: 버스 노선 ID.  버스 노선을 식별하는 고유 ID입니다. 범주형 변수이며, Label Encoding 등을 통해 수치형 데이터로 변환될 수 있습니다.
* **BUSINFOUNIT_ID**: 구간 ID. 버스 노선 내의 특정 구간을 식별하는 ID입니다.  범주형 변수이며, Label Encoding 또는 Hash Encoding 등을 통해 수치형 데이터로 변환될 수 있습니다.
* **LEN**: 구간 길이 (km 또는 m 단위).  구간의 물리적인 길이를 나타냅니다. 수치형 변수이며, Min-Max Scaling 등의 표준화 과정을 거칠 수 있습니다.
* **DEP_TIME**: 출발 시간 (`HH:MM:SS` 또는 `HH:MM` 형식). 버스가 해당 구간을 출발한 시각입니다. 시간 정보는 시계열 특성을 나타내는 중요한 변수이며, 시간, 분, 주기성 특징 (sin, cos 변환) 등으로 변환되어 모델 입력으로 사용될 수 있습니다.
* **TIME_GAP**: 실제 통행 시간 (초 또는 분 단위).  모델이 예측해야 하는 목표 변수 (target variable)입니다.  해당 구간을 실제로 통행하는 데 걸린 시간입니다.

**데이터 전처리:**

* **결측치 처리:** 데이터에 결측치가 존재할 경우, 제거, 평균/중앙값 대체, 또는 다른 통계적 기법을 사용하여 처리합니다.
* **이상치 처리:**  통계적 방법 또는 도메인 지식을 활용하여 이상치를 탐지하고, 제거 또는 대체합니다.
* **범주형 변수 인코딩:** `DAY_TYPE`, `BUSROUTE_ID`, `BUSINFOUNIT_ID` 와 같은 범주형 변수를 모델이 학습할 수 있도록 수치형으로 변환합니다 (One-Hot Encoding, Label Encoding, Hash Encoding 등).
* **수치형 변수 스케일링/정규화:** `LEN`, 시간 관련 변수 등을 Min-Max Scaling, StandardScaler 등을 사용하여 스케일링하거나 정규화하여 모델 학습 효율성을 높입니다.
* **특징 공학 (Feature Engineering):**
    * **주기성 특징:** 출발 시간 (`DEP_TIME`) 및 요일 (`DAY_TYPE`) 정보를 sin, cos 함수를 사용하여 주기적인 특징으로 변환합니다. (예: 하루 24시간 주기, 일주일 7일 주기).
    * **시간대 특징:**  출퇴근 시간, 심야 시간 등 시간대별 교통 혼잡도를 반영하는 특징을 생성합니다.
    * **거리 유형 특징:** 구간 길이 (`LEN`)에 따라 단거리, 중거리, 장거리 구간을 구분하는 특징을 생성합니다.
    * **로그 변환:** 구간 길이 (`LEN`) 변수를 로그 변환하여 데이터 분포를 조정하고 모델 성능을 개선할 수 있습니다.
    * **주말/평일 특징:** `DAY_TYPE` 정보를 활용하여 주말 여부를 나타내는 특징을 생성합니다.
    * **혼잡 시간대 특징:**  출퇴근 시간 등 혼잡 시간대를 나타내는 특징을 생성합니다.

## 모델

본 프로젝트에서는 통행 시간 예측을 위해 다음과 같은 두 가지 주요 모델을 사용했습니다.

1. **LSTM (Long Short-Term Memory) 모델:**

    * **모델 구조:**  LSTM은 순환 신경망 (RNN) 의 한 종류로, 시계열 데이터의 장기 의존성을 학습하는 데 효과적인 모델입니다.  본 프로젝트에서는 버스 운행 데이터의 시간 순서 패턴을 학습하기 위해 LSTM 모델을 사용합니다.  모델 구조는 입력층, LSTM 층, 완전 연결층 (Fully Connected Layer), 출력층 등으로 구성될 수 있습니다.  `travel_LSTM.py` 및 `travel_LSTM_label.py`에 다양한 LSTM 모델 구조가 구현되어 있을 수 있습니다.
    * **입력 특징:**  요일 정보 (One-Hot Encoding), 구간 길이, 출발 시간 (시간, 분, 주기성 특징),  혼잡 시간대 여부, 거리 유형 등 다양한 특징을 입력으로 사용합니다.
    * **학습 과정:**  과거 버스 운행 데이터를 사용하여 LSTM 모델을 학습합니다.  손실 함수 (Loss function)로는 평균 제곱 오차 (Mean Squared Error, MSE) 또는 평균 절대 오차 (Mean Absolute Error, MAE) 등을 사용하며, Optimizer로는 Adam, SGD 등을 사용하여 손실 함수를 최소화하는 방향으로 모델 파라미터를 업데이트합니다.  학습 과정에서 검증 데이터셋을 사용하여 과적합 (overfitting) 을 방지하고, Early Stopping 기법을 적용하여 최적의 학습 epoch 수를 결정합니다.
    * **하이퍼파라미터:** LSTM 층의 hidden unit 수, layer 수, learning rate, batch size, dropout rate 등 다양한 하이퍼파라미터를 조정하여 모델 성능을 최적화합니다.  Optuna와 같은 하이퍼파라미터 최적화 도구를 사용하여 효율적인 탐색을 수행했을 수 있습니다.

2. **XGBoost (Extreme Gradient Boosting) 모델:**

    * **모델 구조:** XGBoost는 Gradient Boosting 알고리즘 기반의 머신러닝 모델로, 정형 데이터 예측 문제에서 뛰어난 성능을 보이는 모델입니다.  트리 기반 모델이며, 여러 개의 결정 트리 (Decision Tree) 를 앙상블 (Ensemble) 하여 예측 성능을 높입니다. `travel_XGBoost.py`에 XGBoost 모델 학습 코드가 구현되어 있습니다.
    * **입력 특징:** LSTM 모델과 유사하게, 요일 정보, 구간 길이, 출발 시간, 혼잡 시간대 여부, 거리 유형 등 다양한 특징을 입력으로 사용합니다.
    * **학습 과정:**  과거 버스 운행 데이터를 사용하여 XGBoost 모델을 학습합니다.  손실 함수로는 Mean Squared Error 등을 사용하며, Gradient Boosting 알고리즘을 통해 손실 함수를 최소화하는 방향으로 모델을 학습합니다.  학습 과정에서 검증 데이터셋을 사용하여 Early Stopping 기법을 적용하고, 최적의 트리 개수를 결정합니다.
    * **하이퍼파라미터:**  `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree` 등 다양한 하이퍼파라미터를 조정하여 모델 성능을 최적화합니다.

## 실행 방법

프로젝트 코드를 실행하고 모델을 학습하기 위한 상세한 방법은 다음과 같습니다.

1. **개발 환경 설정:**

    * **Python 버전:** Python 3.7 이상 버전 권장.
    * **필요 라이브러리 설치:**  프로젝트 실행에 필요한 Python 라이브러리를 설치합니다. `pip` 또는 `conda` 를 사용하여 설치할 수 있습니다.  명령 프롬프트 또는 터미널을 열고 다음 명령어를 실행합니다.

      ```bash
      pip install torch pandas scikit-learn numpy tqdm xgboost optuna
      ```
      (GPU를 사용하는 경우 CUDA 및 cuDNN 설치 필요. PyTorch 설치 시 CUDA 버전에 맞는 버전을 선택하여 설치)

2. **데이터셋 준비:**

    * **데이터셋 다운로드/준비:**  `dataset/` 디렉토리에 학습, 검증, 테스트 CSV 파일을 위치시킵니다.  실제 데이터 파일은 레포지토리에 포함되어 있지 않으므로, 별도로 데이터셋을 준비해야 합니다.  데이터 파일명은 `config` 변수에 설정된 파일 경로와 일치하도록 합니다.
    * **데이터셋 구조 확인:**  CSV 파일의 컬럼 구조 (DAY_TYPE, BUSROUTE_ID, BUSINFOUNIT_ID, LEN, DEP_TIME, TIME_GAP 등) 및 데이터 형식을 확인합니다.

3. **모델 학습:**

    * **LSTM 모델 학습:** `src/train/route/travel_LSTM.py` 또는 `src/train/route/travel_LSTM_label.py` 스크립트를 실행하여 LSTM 모델을 학습합니다.  명령 프롬프트 또는 터미널에서 해당 스크립트가 위치한 디렉토리로 이동한 후 다음 명령어를 실행합니다.

      ```bash
      python src/train/route/travel_LSTM.py
      ```
      또는
      ```bash
      python src/train/route/travel_LSTM_label.py
      ```

    * **XGBoost 모델 학습:** `src/train/ML/travel_XGBoost.py` 스크립트를 실행하여 XGBoost 모델을 학습합니다.

      ```bash
      python src/train/ML/travel_XGBoost.py
      ```

    * **학습 설정 조정:** 각 스크립트 파일 (`travel_LSTM.py`, `travel_LSTM_label.py`, `travel_XGBoost.py`) 내의 `config` 딕셔너리 변수를 수정하여 학습 파라미터 (epochs, batch size, learning rate, patience 등) 및 데이터셋 파일 경로를 조정할 수 있습니다.

4. **학습 결과 확인:**

    * **모델 저장 경로:** 학습된 모델은 `config` 변수에 설정된 `model_save_path` 에 저장됩니다.  (예: `model/travel_lstm_model.pth`, `model/travel_xgboost_model.joblib`).
    * **학습 로그:** 학습 과정은 콘솔에 출력되며, 추가적으로 TensorBoard, 로그 파일 등을 사용하여 학습 과정을 모니터링할 수 있습니다. (TensorBoard 사용법은 별도 검색 필요).

## 추가 정보

* **하이퍼파라미터 최적화:**  `legacy/travel_time_optuna.py`, `legacy/travel_time_optuna_hash.py`, `legacy/holding_time_optuna_hash.py` 등의 코드를 참고하여 Optuna를 사용한 하이퍼파라미터 최적화 실험을 수행할 수 있습니다.  최적화된 하이퍼파라미터를 모델 학습에 적용하여 성능을 향상시킬 수 있습니다.
* **모델 평가:** 학습된 모델을 테스트 데이터셋으로 평가하고, 예측 성능 지표 (RMSE, MAE 등) 를 계산하여 모델 성능을 객관적으로 평가합니다.  평가 코드는 각 모델 학습 스크립트 내에 포함되어 있거나, 별도의 평가 스크립트를 작성할 수 있습니다.
* **TensorBoard 시각화:**  TensorBoard를 사용하여 학습 과정 (loss, accuracy 등) 을 시각화하고, 모델 구조 및 학습 결과를 분석할 수 있습니다.
* **모델 개선 방향:**
    * **더욱 다양한 특징 활용:** 날씨 정보, 교통 혼잡도 정보, 공휴일 정보 등 외부 데이터를 추가적으로 활용하여 모델 입력 특징을 확장합니다.
    * **모델 구조 개선:**  LSTM 모델의 layer 수, hidden unit 수, attention mechanism 적용 등 모델 구조를 개선하여 성능을 향상시킵니다.  Transformer 모델 등 최신 딥러닝 모델을 적용해볼 수도 있습니다.
    * **앙상블 모델:** LSTM 모델과 XGBoost 모델, 또는 다른 여러 모델을 앙상블 하여 예측 성능을 더욱 높일 수 있습니다.
    * **실시간 데이터 반영:**  실시간 버스 운행 데이터 및 교통 정보를 모델에 반영하여 실시간 예측 시스템을 구축합니다.

## 연락처

[skykum2004@gmail.com]