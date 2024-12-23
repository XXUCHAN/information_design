from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ItemViewSet
from .views import run_script,run_top_script,run_init
router = DefaultRouter()
router.register(r'items', ItemViewSet)

urlpatterns = [
    path('',include(router.urls)),
    path('strategies/', run_script),
    path('top/',run_top_script),
    path('init/',run_init)
]
