from django.http import HttpResponse
from django.shortcuts import render
import heapq

from .forms import RecommenderForm
from .network import Network
import pickle
import numpy as np
import time
import requests
import json
import torch
import sklearn
import pickle
import re
import pandas as pd
import string
import time
import datetime
from django.views.decorators.csrf import csrf_exempt

url = "http://e621.net"
username = "shitty_markov_chain"
project_name = "Recommender/1.0"
headers = {
    'User-Agent': f"{project_name} {username}"
}


class Recommender:
    def __init__(self):
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(__name__, name)
                except AttributeError:
                    return super().find_class(module, name)
        #model, posts_encoded, posts_mapping = pickle.load(open("deploy_data", "rb"))
        print("loading static data...", flush=True)
        self.next_request = time.time()
        self.df = pd.read_csv("updated_posts.csv")
        


        
        self.df["tags"] = self.df["tags"].apply(lambda s: self.clean(s))
        
        blacklist = set(["scat", "gore", "cub", "loli", "feral", "young"])
        self.df["whitelist"] = self.df.tags.apply(lambda x: not bool(set(x.split()) & blacklist))
        
        whitelist = set(self.df[(self.df["score"] > 75) & self.df["whitelist"]].id.values)
        
        self.img_map = dict()
        for id, link in self.df[["id", "sample_url"]].values:
            if id in whitelist:
                self.img_map[id] = link
        self.description_map = dict()
        for id, score, fav_count, tags in self.df[["id", "score", "fav_count", "tags"]].values:
            if id in whitelist:
                description = f"score:{score} fav_count:{fav_count} {tags}"
                self.description_map[id] = description

        del self.df

        self.model, self.posts_encoded, self.posts_mapping = CustomUnpickler(open('deploy_data', 'rb')).load()
        self.model.eval()

        self.all_ids = set(self.posts_mapping) & whitelist

        print("loading static data: done", flush=True)




    def clean(self, x):
        if type(x) == int:
            return x
        if type(x) == str:
            printable = set(string.printable)
            return ''.join(filter(lambda c: c in printable, x))
        return x.encode('ascii', 'ignore').strip()

    def request(self, url):
        while time.time() < self.next_request:
            time.sleep(self.next_request - time.time() + 0.1)
        self.next_request = time.time() + 1
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            string = r.content
            return json.loads(string)
        else:
            raise RuntimeError(f'Error {r.status_code} while getting {url}')

    def list_for_user(self, name, max=960):
        res = []
        current_id = 2105591
        while True:
            post_url = url + "/posts.json?tags=fav:{} id:<{}&limit=320".format(name, current_id)
            content = self.request(post_url)
            n_posts = len(content)
            for post in content["posts"]:
                res.append(post["id"])
                current_id = min(current_id, post["id"])
            if n_posts < 320 or len(res) >= max:
                break
        return res

    def chunks(self, l, n):
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))

    def convert(self, id):
        return self.posts_encoded[self.posts_mapping[id]]
    def convert_list(self, l):
        for x in l:
            if x in self.posts_mapping:
                yield self.convert(x)

    def score_posts(self, user, to_test, fingerprint=None):
        favs = list(self.convert_list(user))
        favs = torch.Tensor(np.array(favs).astype(float))
        to_test = list([torch.tensor(np.array(self.convert(i)).astype(float)) for i in to_test])
        to_test = torch.stack(to_test)
        res, fingerprint = self.model(favs.transpose(0, 1).unsqueeze(0), to_test.transpose(0, 1).unsqueeze(0))
        return res.data[0], fingerprint

    def __call__(self, fav_list):
        results = []
        to_test = list(self.all_ids - set(fav_list))
        fingerprint = None
        for batch in self.chunks(to_test, 512):
            scores, fingerprint = self.score_posts(fav_list, batch, fingerprint)
            for i, score in zip(batch, scores):
                results.append((score.item(), i))
        results.sort(reverse=True)
        return results[:512]

recommender = Recommender()



@csrf_exempt
def recommenderView(request):
    begin = time.time()
    if request.method == 'POST':
        form = RecommenderForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            username_regex = r"^[\w~'-]{0,20}$"
            if not re.match(username_regex, username):
                return HttpResponse("Usename is not valid", status=400)

            posts = recommender.list_for_user(username)
            if not posts:
                return HttpResponse(f"No post was favorited by {username}", status=404)
            
            recommendations = recommender(posts)

            response = HttpResponse()
            
            for _, p in recommendations:
                display = False
                if display:
                    img_link = recommender.img_map[p]
                    embedding = f'<img src="{img_link}"  title="title test"/>'
                    response.write(f'<p> <a href="https://e621.net/post/show/{p}">{embedding}</a> </p>')
                else:
                    desc = recommender.description_map[p]
                    response.write(f'<p> <a href="https://e621.net/post/show/{p}">{p}</a> <small>{desc}</small></p>')

            with open("log", "a") as f:
                print(datetime.datetime.now().isoformat(), username, time.time() - begin, file=f)
            return response

    else:
        form = RecommenderForm()

    return render(request, 'recommend.html', {'form': form})
