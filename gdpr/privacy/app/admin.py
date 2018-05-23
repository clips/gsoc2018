from django.contrib import admin
from .models import Attribute_Configuration, Supression_Configuration, Deletion_Configuration
# Register your models here.
admin.site.register(Attribute_Configuration)
admin.site.register(Supression_Configuration)
admin.site.register(Deletion_Configuration)
