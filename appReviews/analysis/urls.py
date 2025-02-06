from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path('registration', views.register, name='register'),
    path('', views.login, name = 'login'),
    path('upload-file/', views.upload_file, name = 'upload_file'),
    path('bert_model', views.bert_model_analysis, name= 'bert_model_analysis'),
    path('show_sentiment_distribution', views.show_sentiment_distribution, name = "show_sentiment_distribution"),
    path('nlp_model_analysis', views.nlp_model_analysis, name="nlp_model_analysis"),
    path('logout', views.logout_user, name = "logout_user"),
    path('prediction', views.prediction, name="prediction"),
    path('show_comaprison', views.show_sentiment_comparison, name = 'sentiment_comparison'),
    path('llm-model/', views.llm_model_analysis, name='llm_model_analysis'),  # Add this for LLM-based analysis

]
