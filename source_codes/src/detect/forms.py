from django import forms

model_choices = [
    ("Standard models", (
        ("bs", "Background Subtraction"),
        ("dof", "dense Optical Flow"),
        ("sof", "sparse Optical Flow"),
        ("fd", "Frame Differencing"),
        ("td", "Temporal Differencing"),
        ("infrared", "Infrared")
    )),
    ("Bio-inspired models", (
         ("estmd", "ESTMD"),
         ("fstmd", "FSTMD"),
         ("dstmd", "DSTMD")
    ))
]


class UploadForm(forms.Form):
    model = forms.ChoiceField(choices=model_choices, label="Choose a model :")
    video = forms.FileField(required=True, widget=forms.FileInput, label="Choose a video")
    number = forms.IntegerField(label="Camera number :")