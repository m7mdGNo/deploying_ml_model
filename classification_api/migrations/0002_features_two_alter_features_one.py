# Generated by Django 4.0.6 on 2022-07-31 11:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('classification_api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='features',
            name='two',
            field=models.FloatField(default=0),
        ),
        migrations.AlterField(
            model_name='features',
            name='one',
            field=models.FloatField(default=0),
        ),
    ]
