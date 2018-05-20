from django.db import models

# Create your models here.
from django.db import models
# Create your models here.


class Attribute_Configuration(models.Model):
    # Contains the primary name of the attribute.
    # Will create a one to many mapping for aliases and action details
    attribute_name = models.CharField(max_length=150)
    ATTRIBUTE_ACTION_CHOICES = (
        ('supp', 'supression'),
        ('gen', 'generalization'),
        ('del', 'deletion'),
    )
    attribute_action = models.CharField(
        max_length=5, choices=ATTRIBUTE_ACTION_CHOICES)
