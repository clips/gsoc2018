from django.contrib import admin
from .models import Attribute_Configuration, Supression_Configuration,\
    Deletion_Configuration, Attribute_Alias, Regex_Pattern

# Register your models here.
admin.site.register(Attribute_Configuration)
admin.site.register(Supression_Configuration)
admin.site.register(Deletion_Configuration)
admin.site.register(Attribute_Alias)
admin.site.register(Regex_Pattern)
