from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
import heapq

from .forms import UserFindForm
import pickle
import numpy as np

users_data = pickle.load(open("users_preprocess.p", "rb"))

@csrf_exempt
def userFindView(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = UserFindForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            username = form.cleaned_data['username']

            if username not in users_data:
                return HttpResponse("I don't know this username")

            baseline = users_data[username]

            l = []
            for u in users_data:
                diff = ((baseline - users_data[u]) ** 2).mean()
                l.append((diff, u))
            res = heapq.nsmallest(17, l)
            res.sort()
            response = HttpResponse()
            response.write("<p>most similar users to {}:</p>".format(username))
            for dist, u in res[1:]:
                response.write('<p> <a href="https://e621.net/post?tags=fav:' + u + '">' + u + '</a> </p>')


            return response

    else:
        form = UserFindForm()

    return render(request, 'user.html', {'form': form})
