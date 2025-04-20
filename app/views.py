from django.shortcuts import render

# Create your views here.

from django.shortcuts import render

# 기업코드 - 모델 파일 맵
COMPANY_MODEL = {
    '005930':'samsung.h5'
}
# 기업코드 - 이름 맵
COMPANY_NAME = {
    '005930':'삼성전자'
}
def home(request):
    return render(request, 'home.html')

