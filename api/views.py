# from django.shortcuts import render
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# import multiprocessing
# from NewModel import modelCall
# import smtpserver
# import warnings
# warnings.filterwarnings('ignore')


# # Create your views here.

# @api_view(['GET'])
# def getData(request):
#     data=[
#         {
#             'Name': 'Surya',
#             'Age':20
#         },
#         {
#             'Name': 'LambLord',
#             'Age':69
#         }
#     ]
#     return Response(data)

# @api_view(['POST'])
# def postData(request):
#     postdata = request.data
#     print(postdata)
#     return Response(postdata)

# @api_view(['POST'])
# def handleContactData(request):
#     contactData = request.data
#     name = contactData.get('name')
#     email = contactData.get('email')
#     message = contactData.get('message')
#     return Response("Contact Data Recieved")

# @api_view(['POST'])
# def handleAIData(request):
#     inputAIData = request.data
#     title = inputAIData.get('title')
#     subtitle = inputAIData.get('subtitle')
#     category = inputAIData.get('category')
#     subcategory = inputAIData.get('subcategory')
#     keywords = inputAIData.get('keywords')
#     abstract = inputAIData.get('abstract')
#     email = inputAIData.get('email')
#     uname = inputAIData.get('uname')

#     print(str(title)+" "+str(subtitle)+" "+str(category)+" "+str(subcategory)+" "+str(abstract)+" "+str(keywords)+" "+str(email)+" "+str(uname))

#     process = multiprocessing.Process(target=processes, args=(title, subtitle, category, subcategory, keywords, abstract, email, uname))
#     process.start()

#     return Response("AI Input Recieved")

# def processes(title, subtitle, category, subcategory, keywords, abstract, email):
#     Beta = 10
#     Gamma = 20
#     X1 = len(title)
#     X2 = len(category)
#     X3 = len(subcategory)
#     X4 = len(abstract)
#     if X1 > Beta and X2 > Beta and X3 >= Beta and X4 >= Gamma:
#         print("Successful Inputs for AI Writer provided now wait !!!!")
#         modelCall.aicall(title, subtitle, category, subcategory, keywords, abstract)
#         smtpserver.sendmail(email)
#         print("Blog Email sent successfully user to the provided email address in AI Page")
#     else:
#         print("Improper inputs provided by user to the AI Writer !!!!")
#         smtpserver.warningmail(email)
#         print("Warning mail sent successfully to the provided email address in AI Page")
    

from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import multiprocessing
from NewModel import modelCall
import smtpserver

# Create your views here.

@api_view(['GET'])
def getData(request):
    data=[
        {
            'Name': 'Surya',
            'Age':20
        },
        {
            'Name': 'LambLord',
            'Age':69
        }
    ]
    return Response(data)

@api_view(['POST'])
def postData(request):
    postdata = request.data
    print(postdata)
    return Response(postdata)

@api_view(['POST'])
def handleContactData(request):
    contactData = request.data
    name = contactData.get('name')
    email = contactData.get('email')
    message = contactData.get('message')
    print(name, email, message)
    return Response("Contact Data Recieved")

@api_view(['POST'])
def handleAIData(request):
    inputAIData = request.data
    title = inputAIData.get('title')
    subtitle = inputAIData.get('subtitle')
    category = inputAIData.get('category')
    subcategory = inputAIData.get('subcategory')
    keywords = inputAIData.get('keywords')
    abstract = inputAIData.get('abstract')
    email = inputAIData.get('email')

    print(str(title)+" "+str(subtitle)+" "+str(category)+" "+str(subcategory)+" "+str(abstract)+" "+str(keywords)+" "+str(email))

    process = multiprocessing.Process(target=processes, args=(title, category, subcategory, keywords, abstract, email))
    process.start()

    return Response("AI Input Recieved")

def processes(title, category, subcategory, keywords, abstract, email):
    
    if len(title) >= 10 and len(abstract) >= 20 and len(subcategory) >= 10 and len(category) >= 10:
        print("SUCCESS USER !!")
        modelCall.aicall(str(title), str(category), str(subcategory), str(keywords), str(abstract))
        smtpserver.sendmail(email)
        print("EMAIL SENT SUCCESSFULLY")
    else:
        print("FAILURE USER !!")
        smtpserver.warningmail(email)
        print("WARNING MAIL SENT SUCCEFULLY")

