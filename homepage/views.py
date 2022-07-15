from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def homepageCode(request):
    return render(request, 'interface.html')