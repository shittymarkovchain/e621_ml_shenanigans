from django import forms

class RecommenderForm(forms.Form):
    username = forms.CharField(label='username', max_length=100)
