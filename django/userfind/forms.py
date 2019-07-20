from django import forms

class UserFindForm(forms.Form):
    username = forms.CharField(label='username', max_length=100)
