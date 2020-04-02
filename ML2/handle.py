from django.http import HttpResponse
import os
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response

from ML2.settings import BASE_DIR
from ML2.workers.worker_start import run


@api_view(['GET', 'POST'])
def do(request):
    return Response("hello")


@csrf_exempt
@api_view(['POST'])
def upload(request):
        obj = request.FILES.get('file')
        f = open(os.path.join(BASE_DIR, 'static', 'pic', obj.name), 'wb')
        for chunk in obj.chunks():
            f.write(chunk)
        f.close()

        re = run(os.path.join(BASE_DIR, 'static', 'pic', obj.name))
        return Response(re)