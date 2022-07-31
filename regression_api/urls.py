from django.urls import path,include
from . import views

urlpatterns = [
    path('regression/',views.input_features.as_view()),
]
