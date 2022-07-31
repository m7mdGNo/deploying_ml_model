from rest_framework import serializers
from .models import Features

class model_serializer(serializers.ModelSerializer):
    class Meta:
        model = Features
        fields = '__all__'