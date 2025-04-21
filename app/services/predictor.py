import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import FinanceDataReader as fdr
import joblib

# 기업 코드 → 모델 파일명, 이름 매핑
COMPANY_MODEL_NAME = {
    '005930': ('stock_samsung.h5', '삼성전자'),
    # 필요한 기업들 추가 가능
}

# 주요 입력변수
INPUT_FEATURES = ['Open', 'High', 'Low', 'Close', 'DOW', 'NAS', 'AMD', 'KS11']

# 예측 함수 (오늘 기준 5일 예측)
def predict_stock(code='005930'):
    print("load_model 정상 작동")
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
    start_date = today - timedelta(days=40) # 학습용 20일 확보를 위한 여유
    print(today, start_date)

    # 주가 데이터 불러오기
    # 데이터 수집
    stock = fdr.DataReader(code, start_date)[['Open', 'High', 'Low', 'Close']]
    print('stock 데이터 수집 완료')
    dow = fdr.DataReader('^DJI', start_date)[['Close']].rename(columns={'Close': 'DOW'})
    print("✅ DOW 데이터 수집 성공")
    nas = fdr.DataReader('^IXIC', start_date)[['Close']].rename(columns={'Close': 'NAS'})
    print("✅ nas 데이터 수집 성공")
    amd = fdr.DataReader('AMD', start_date)[['Close']].rename(columns={'Close': 'AMD'})
    print("✅ amd 데이터 수집 성공")
    ks11 = fdr.DataReader('^KS11', start_date)[['Close']].rename(columns={'Close': 'KS11'})
    print("✅ kospi 데이터 수집 성공")
    # 병합 및 전처리
    stock = stock.join([dow.shift(1), nas.shift(1), amd.shift(1), ks11.shift(1)])
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
    predict_df.index = predict_df.index.strftime('%Y-%m-%d')

    #시각화를 위한 실제값 + 예측값 연결
    last_20_close = stock['Close'].iloc[-20:]
    full_dates = list(last_20_close.index.strftime('%Y-%m-%d')) + list(date_range.strftime('%Y-%m-%d'))
    full_prices = list(last_20_close.values) + list(pred)

    return company_name, predict_df.round(2), full_dates, full_prices