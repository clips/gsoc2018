from django.db import models

# Create your models here.
from django.db import models
# Create your models here.
from django.contrib.auth.models import User


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
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def clean(self, *args, **kwargs):
        print('CHECKINGGG')
        if self.attribute_action != 'supp' and self.attribute_action != 'gen' and self.attribute_action != 'del':
            raise Exception(
                'Illegal value entered')

    def __str__(self):
        return self.attribute_title + ' - ' + self.attribute_action


class Supression_Configuration(models.Model):
    attribute = models.OneToOneField(
        Attribute_Configuration, primary_key=True, on_delete=models.CASCADE)
    # The field below gives number of characters to suppress. Can also use %
    suppress_number = models.IntegerField(null=True, blank=True, default=None)
    # The field below specifies the percentage of characters to suppress
    suppress_percent = models.FloatField(null=True, blank=True, default=None)
    # Will check if atleast one of the two (percent or number) is provided
    # if no, will throw an error
    # the field below allows to set the replacement character
    replacement_character = models.CharField(max_length=1, default='*')

    def clean(self, *args, **kwargs):
        if not self.suppress_number and not self.suppress_percent:
            raise Exception(
                'You need to mention either supress number of bits or percentage')


class Deletion_Configuration(models.Model):
    # Linking It to the attribute
    attribute = models.OneToOneField(
        Attribute_Configuration, primary_key=True, on_delete=models.CASCADE)
    # The title you want to replace it with
    replacement_name = models.TextField()


class Attribute_Alias(models.Model):
    # Linking it to the attribute
    attribute = models.ForeignKey(
        Attribute_Configuration, on_delete=models.CASCADE)
    # The alias the attribute/entity name
    alias = models.TextField()
    # Adding user for faster DB lookups
    user = models.ForeignKey(User, null=True, on_delete=models.CASCADE)


class Regex_Pattern(models.Model):
    # Linking it to the attribute
    attribute = models.ForeignKey(
        Attribute_Configuration, on_delete=models.CASCADE)
    # Store the raw string of the pattern
    regular_expression = models.CharField(max_length=500)
    # Adding user for faster DB lookups
    user = models.ForeignKey(User, null=True, on_delete=models.CASCADE)
