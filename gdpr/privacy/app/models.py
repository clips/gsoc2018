from django.db import models

# Create your models here.
from django.db import models
# Create your models here.


class Attribute_Configuration(models.Model):
    # Contains the primary name of the attribute.
    # Will create a one to many mapping for aliases
    # Will create a one to one mapping for action details
    attribute_title = models.CharField(max_length=150)
    ATTRIBUTE_ACTION_CHOICES = (
        ('supp', 'supression'),
        ('gen', 'generalization'),
        ('del', 'deletion'),
    )
    attribute_action = models.CharField(
        max_length=5, choices=ATTRIBUTE_ACTION_CHOICES)


class Supression_Configuration(models.Model):
    attribute = models.OneToOneField(Attribute_Configuration, primary_key=True)
    # The field below gives number of characters to suppress. Can also use %
    suppress_number = models.IntegerField()
    # The field below specifies the percentage of characters to suppress
    suppress_percent = models.FloatField()
    # Will check if atleast one of the two is provided, if no, will throw an error

    def clean(self, *args, **kwargs):
        if not self.suppress_number and not self.suppress_percent:
            raise Exception(
                'You need to mention either supress number of bits or percentage')


class Deletion_Configuration(models.Model):
    # Linking It to the attribute
    attribute = models.OneToOneField(Attribute_Configuration, primary_key=True)
    # The title you want to replace it with
    replacement_name = models.TextField()


class Attribute_Alias(models.Model):
    # Linking it to the attribute
    attribute = models.ForeignKey(Attribute_Configuration)
    # The alias the attribute/entity name
    alias = models.TextField()
