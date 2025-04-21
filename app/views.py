from django.shortcuts import render

# Create your views here.

from django.shortcuts import render
from .services.predictor import predict_stock
import json

def home(request):
    code = request.GET.get('code', '005930')  # 기본값은 삼성전자
    company_name, predict_df, full_dates, full_prices = predict_stock(code)
    full_prices = [float(p) for p in full_prices]
    context = {
        'company_name': company_name,
        'code': code,
        'predictions': predict_df.to_dict('index'),  # 날짜: {'Close': 값}
        'full_dates': json.dumps(full_dates),
        'full_prices': json.dumps(full_prices),
    }
    return render(request, 'home.html', context)


