from django.urls import path
from . import views
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html')),
    # path('services', TemplateView.as_view(template_name='service.jsx')),
    # path('contact', TemplateView.as_view(template_name='../src/contact.jsx')),

    path('api/getdata/', views.getData, name="getData"),
    path('api/postdata/', views.postData, name="postData"),
    path('api/contactsubmit/', views.handleContactData, name="handleContactData"),
    path('api/handleaidata/', views.handleAIData, name="handleAIData"),
]