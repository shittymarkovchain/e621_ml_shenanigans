from django.urls import path

from .views import userFindView

urlpatterns = [
        path('', userFindView, name='find')
        ]
