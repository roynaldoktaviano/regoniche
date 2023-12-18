from django.urls import path
from . import views

urlpatterns = [
    path('', views.indexs, name='pages'),
    path('about', views.about, name='pages'),
    path('diagnose', views.diagnose, name='diagnose'),

]
