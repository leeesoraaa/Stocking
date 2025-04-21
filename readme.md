# 👑 Stock + King = 주식왕

5일간의 미래 주가를 예측하는 웹 서비스입니다.  
삼성전자 등 주요 기업의 데이터를 기반으로, LSTM 딥러닝 모델을 통해 종가(Close) 예측 결과를 제공합니다.

---
![stocking_mainpage](https://github.com/user-attachments/assets/aa8c09e6-173a-444d-8db8-c67068e1c72d)

## 📌 주요 기능

- ✅ 실시간 데이터 기반 주가 예측
- ✅ 5일간 종가(Close) 예측
- ✅ LSTM + 외부 지표(DOW, NASDAQ, KOSPI 등) 활용
- ✅ Django 기반 웹 인터페이스 제공
- ✅ 기업별 모델 확장 가능

---

## 🧠 기술 스택

| 구성 요소     | 기술                         |
|--------------|------------------------------|
| 백엔드       | Django                   |
| 예측 모델     | LSTM (Keras/Tensorflow)      |
| 데이터 수집   | FinanceDataReader            |
| 정규화       | scikit-learn (StandardScaler)|
| 시각화 (옵션) | matplotlib / Chart.js        |

---
📘 모델 학습 과정은 [`notebooks/train_model.ipynb`](notebooks/train_model.ipynb) 참고

## 📈 앞으로 할 일

- [ ] 사용자 입력으로 기업 선택 기능
- [ ] 실시간 예측 차트 시각화
- [ ] 주요 기업 모델 추가
- [ ] Docker로 패키징 및 배포 준비
