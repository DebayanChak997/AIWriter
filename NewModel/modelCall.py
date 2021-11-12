# from NewModel import GPT2

# head = "" 
# out = ""

# def aicall(title, category, subcategory, keywords, abstract):
    
#     global head
#     global out
#     STRING_SEED = ""

#     head = title
#     out = GPT2.GPT2MainFunc(title, subcategory, category, abstract)

#     for items in out :

#         STRING_SEED = STRING_SEED + items

#     with open('Blog.txt', 'w') as fp:

#         fp.write(STRING_SEED)

# def emailbody():

#     return out

# def emailsub():

#     return head 

from NewModel import GPT2
import warnings
warnings.filterwarnings('ignore')

head = ""
out = ""

def aicall(title, category, subcategory, keywords, abstract):
    global head
    global out
    head = title
    out = GPT2.GPT2MainFunc(title, subcategory, category, abstract)
    with open('Blog.txt', 'w', errors='ignore') as fp:
        fp.writelines(out)


def emailbody():
    return out

def emailsub():
    return head

#
#
# ###################################################### 08-10-2021 #######################################################
# from NewModel import GPT2
# import pyrebase
# import random
# import string
#
# config = {
#     'apiKey': "AIzaSyBOlwCWjEPhZzN7GtoBEAWcw4Se_aXOsXU",
#     'authDomain': "aiwriter-6ae7e.firebaseapp.com",
#     'databaseURL': "https://aiwriter-6ae7e-default-rtdb.asia-southeast1.firebasedatabase.app",
#     'projectId': "aiwriter-6ae7e",
#     'storageBucket': "aiwriter-6ae7e.appspot.com",
#     'messagingSenderId': "287933654027",
#     'appId': "1:287933654027:web:8b81b112211766864c930e",
#     'measurementId': "G-CHE0BPR3PQ"
# }
#
# firebase = pyrebase.initialize_app(config)
# authe = firebase.auth()
# database = firebase.database()
# head = ""
# out = "empty"
#
# def aicall(title, subtitle, category, subcategory, keywords, abstract, email, uname):
#     global head
#     global out
#     head = title
#     output_seed = "false"
#     data = {"title":title, "subtitle":subtitle, "category":category, "subcategory":subcategory, "keywords":keywords, "abstract":abstract, "email":email, "output" : out, "output_seed":output_seed}
#     database.child("AICalldata").child("UID").child(uname).child(title).set(data)
#     first_list_output = GPT2.GPT2MultipleGenerationFunction(title, subcategory, category, abstract)
#     out = GPT2.genOut(first_list_output)
#     output_seed = "true"
#     database.child("AICalldata").child("UID").child(uname).child(title).update({"output":out, "output_seed":output_seed})
#     with open('Blog.txt', 'w') as fp:
#         fp.write(out)
#
# def emailbody():
#     return out
#
# def emailsub():
#     return head
