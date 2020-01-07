from django.urls import path

from .views import recommenderView

urlpatterns = [
        path('recommender', recommenderView, name='recommend')
        ]
