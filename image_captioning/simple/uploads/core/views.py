from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from pathlib import Path

import os

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })
def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        print(uploaded_file_url[1:])
        
        sentence=os.popen("python -W ignore attention/caption.py --img="+uploaded_file_url[1:]+" --beam_size=5 --model=attention/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=attention/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json").read()
        sentence=sentence.replace(",","")
        sentence=sentence.replace("'","")
        print(sentence)
        sentence=sentence[:-1]

        
        my_file = Path(uploaded_file_url)
        if my_file.is_file():
            print('file exists')
        else:
            print('file does not exist')

        return render(request, 'core/simple_upload.html', {
            'uploaded_file_url':uploaded_file_url,
            'sentence':sentence
        })
    return render(request, 'core/simple_upload.html')
def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
