from django.urls import path,include
from . import views

urlpatterns = [
    path('classification/',views.input_features.as_view()),
]
