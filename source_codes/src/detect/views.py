from django.shortcuts import render
from django.http import HttpResponseNotFound
from django.core.files.storage import FileSystemStorage
from .forms import *
from technics.bs import bs_fct
from technics.sparseOpticaleFlow import sof
from technics.denseOpticalFlow import dof
from technics.frame_differencing import fd_fct
from technics.infrared import infrared
from technics.temporal_differencing import td_fct


def home(request):
    if request.method == "POST":
        form = UploadForm(request.POST, request.FILES)
        model = request.POST.get('model')
        video = form["video"].value()
        number = form["number"].value()

        if video is not None:
            fs = FileSystemStorage()
            video = str(video)
            file = fs.path(video)
        else:
            try:
                file = int(number)
            except:
                return HttpResponseNotFound("ERROR")

        if model == "bs":
            bs_fct(file)
        elif model == "dof":
            dof(file)
        elif model == "sof":
            sof(file)
        elif model == "fd":
            fd_fct(file)
        elif model == "td":
            td_fct(file)
        elif model == "infrared":
            infrared(file)
        else:
            pass
    return render(request, "home.html")