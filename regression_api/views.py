from django.shortcuts import render
from rest_framework.generics import CreateAPIView,ListCreateAPIView,GenericAPIView
from .models import Features
from .serializers import model_serializer
from rest_framework.response import Response
import numpy as np
from rest_framework import status
import pickle
import time
import joblib

def model_output(X):
    # model = pickle.load(open('./rf_regression.sav','rb'))
    with open('rf_regression.sav', 'rb') as ifp:
        model=pickle.load(ifp)
    # model = joblib.load("./rf_regression.joblib")
    return model.predict(X)



class input_features(ListCreateAPIView):
    queryset  = Features.objects.all()
    serializer_class = model_serializer

    def post(self, request, *args, **kwargs):
        start_time = time.time()
        data_json = request.data
        try:
            data = np.array(list(data_json.values())[1::],dtype=np.float64)
            data = data.reshape(-1, 1)
            data = data.T
            

        except ValueError as ve:
            end_time = time.time()
            return Response( {
                'error_code' : '-1',
                "info": str(ve),
                'respons_time':round(end_time-start_time,4)
            },status=status.HTTP_502_BAD_GATEWAY)

        try:
            output = model_output(data)
            end_time = time.time()
            return Response({'BorrowerRate is: ':output[0],'respons_time':round(end_time-start_time,4)},status=status.HTTP_200_OK)
        
        except ValueError as ve:
            end_time = time.time()
            return Response( {
                'error_code' : '-1',
                "info": str(ve),
                'respons_time':round(end_time-start_time,4)
            },status=status.HTTP_502_BAD_GATEWAY)



