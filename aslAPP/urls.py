from django.urls import path
from aslAPP import views

urlpatterns = [
    path("", views.home, name="home"),
    path("send", views.send, name="send"),
    path("prediction", views.prediction, name='prediction')
]