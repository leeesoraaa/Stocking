import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import FinanceDataReader as fdr
import joblib

# 기업 코드 → 모델 파일명, 이름 매핑
COMPANY_MODEL_NAME = {
    '005930': ('samsung.h5', '삼성전자'),
    # 필요한 기업들 추가 가능
}

# 주요 입력변수
INPUT_FEATURES = ['Open', 'High', 'Low', 'Close', 'DOW', 'NASDAQ', 'AMD', 'KOSPI']

# 예측 함수 (오늘 기준 5일 예측)
def predict_stock(code='005930'):
    # 기업 정보 조회
    model_file, company_name = COMPANY_MODEL_NAME.get(code, COMPANY_MODEL_NAME['005930'])

    # 모델 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'ml', model_file)
    scaler_x_path = os.path.join(current_dir, '..', 'ml', 'scaler_X.pkl')
    scaler_y_path = os.path.join(current_dir, '..', 'ml', 'scaler_y.pkl')

    # 모델 & 스케일러 불러오기
    model = load_model(model_path, compile=False)
    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # 날짜 계산
    today = datetime.today().date()
    start_date = today - timedelta(days=40)  # 학습용 20일 확보를 위한 여유

    # 주가 데이터 불러오기
    # 데이터 수집
    stock = fdr.DataReader(code, start_date, today)[['Open', 'High', 'Low', 'Close']]
    dow = fdr.DataReader('DJI', start_date, today)[['Close']].rename(columns={'Close': 'DOW'})
    nas = fdr.DataReader('IXIC', start_date, today)[['Close']].rename(columns={'Close': 'NASDAQ'})
    amd = fdr.DataReader('AMD', start_date, today)[['Close']].rename(columns={'Close': 'AMD'})
    kospi = fdr.DataReader('KS11', start_date, today)[['Close']].rename(columns={'Close': 'KOSPI'})

    # 병합 및 전처리
    stock = stock.join([dow.shift(1), nas.shift(1), amd.shift(1), kospi.shift(1)])
    stock = stock.fillna(method='ffill').dropna()
    stock = stock[INPUT_FEATURES]

    # 입력 데이터 생성
    test_scaled = scaler_X.transform(stock)  # shape: (n, 8)
    x_input = np.array([test_scaled[-20:]])  # shape: (1, 20, 8)

    # 예측
    pred_scaled = model.predict(x_input).reshape(-1, 1)  # shape: (5, 1)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()

    # 결과 포맷
    forecast_days = 5
    last_date = stock.index[-1]
    date_range = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

    predict_df = pd.DataFrame({'Close': pred.astype(int)}, index=date_range)

    return company_name, predict_df.round(2)