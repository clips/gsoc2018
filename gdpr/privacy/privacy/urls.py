"""privacy URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from app import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^test/$', views.preprocess),
    url(r'register/$', views.register_user),
    url(r'login/$', views.login_user),
    url(r'form-test/$', views.test),
    url(r'add_attribute/$', views.add_attribute),
    url(r'add_suppression_configuration/(\d+)/$',
        views.add_suppression_configuration),
    url(r'add_deletion_configuration/(\d+)/$',
        views.add_deletion_configuration),
    url(r'dashboard/$',
        views.show_dashboard),
    url(r'add_alias/(\d+)/$',
        views.add_alias),
    url(r'anonymize', views.anonymize)
]
