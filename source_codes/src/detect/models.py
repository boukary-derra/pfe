from django.db import models
from datetime import datetime
from django.template.defaultfilters import slugify


class SaveModel(models.Model):
    name = models.CharField(max_length=100, unique=True, blank=False, null=False)
    model = models.CharField(max_length=100, blank=False, null=False)
    video = models.FileField(blank=True, null=True, upload_to='videos')
    slug = models.SlugField(max_length=255, blank=True)
    created_on = models.DateTimeField(blank=True, default=datetime.now)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['-created_on']

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)

        super().save(*args, **kwargs)