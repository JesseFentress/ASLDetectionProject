from django.shortcuts import render
from django.views.decorators import gzip
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
import copy
import itertools
import numpy as np
import json
import tensorflow as tf

# Create your views here.

def home(request):
    context = {}
    return render(request, 'home.html', context)

def g(request):
    print(request.session['prediction'], 'okay prediction')
    yield request.session['prediction']

def prediction(request):
    return StreamingHttpResponse(g(request))

def send(request):
    if  request.method == 'POST':
        labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        if 'data' in request.POST:
            context = {}
            array = request.POST['data']
            array = json.loads(array)
            preprocessed_landmarks = preprocess_hand(array)
            model_interpreter = tf.lite.Interpreter(model_path="aslAPP/static/tflite_model/tflite_classifier.tflite")
            model_interpreter.allocate_tensors()
            input_details = model_interpreter.get_input_details()
            output_details = model_interpreter.get_output_details()
            model_interpreter.set_tensor(input_details[0]["index"], np.reshape(np.array(preprocessed_landmarks, dtype=np.float32), (1, 42)))
            model_interpreter.invoke()
            prediction = model_interpreter.get_tensor(output_details[0]["index"])
            id = np.argmax(np.squeeze([prediction]))
            array = []
            request.session['prediction'] = labels[id]
            return JsonResponse({'prediction': labels[id]})
        else:
            print('Error - Empty POST')
    else:
        return JsonResponse({'prediction': request.session['prediction']})

def preprocess_hand(landmark):
    l = copy.deepcopy(landmark)
    base_x, base_y = 0, 0
    for index, point in enumerate(l):
        if index == 0:
            base_x, base_y = point[0], point[1]
        l[index][0] = l[index][0] - base_x
        l[index][1] = l[index][1] - base_y
    l = list(itertools.chain.from_iterable(l))
    max_value = max(list(map(abs, l)))
    def normalize(n):
        return n / max_value
    l = list(map(normalize, l))
    return l

