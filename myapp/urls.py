from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('slr/', views.slr, name='slr'),
    path('mlr/', views.mlr, name='mlr'),
    path('logistic/', views.logistic, name='logistic'),
    path('polynomial/', views.polynomial, name='polynomial'),
    path('knn/', views.knn, name='knn'),
]

