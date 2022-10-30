from crypt import methods
import re
import requests
from strsimpy.levenshtein import Levenshtein
import flask
from flask import request

app = flask.Flask(__name__)

def lang_type(text):
    en_dict = 'abcdefghijklmnopqrstuvwxyz'
    ru_dict = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    ru = 0
    en = 0
    for char in text.lower():
        if char in en_dict:
            en += 1
        elif char in ru_dict:
            ru += 1
    return (0 if ru > en else 1)
        
# print(lang_type('fuckмо'))

def register(text, lang):
    
    if text.title() == text:
        return 'title'
    
    up_ru_dict = 'АБЕЁРУФЦЩ'
    low_ru_dict = 'абеёруфцщ'
    
    low_en_dict = 'abdefghijklmnpqrtuy'
    up_en_dict = 'ABDEFGHIJKLMNPQRTUY'
    up = 0
    low = 0
    
    if lang == 0:
        for char in text:
            if char in up_ru_dict:
                up += 1
            elif char in low_ru_dict:
                low += 1
                
    if lang == 1:               
        for char in text:
            if char in up_en_dict:
                up += 1
            elif char in low_en_dict:
                low += 1
    if up == 0 and low == 0:
        return 'low'
    
    return ('low' if low > up else 'up')

# print(register('FiLM', 1))   
    
def email(text, lang): 
    if lang == 0:
        regex = re.compile(r"^[A-Za-z0-9а-яА-Я\._-]+@([A-Za-z0-9а-яА-Я]{1,2}|[A-Za-z0-9а-яА-Я]((?!(\.\.))[A-Za-z0-9а-яА-Я.-])+[A-Za-z0-9а-яА-Я])\.[A-Za-zа-яА-Я]{2,}$")
        if re.fullmatch(regex, text):
            return text
        else:
            if text == text.lower():
                return text.replace('@', 'а')
            else:
                return text.replace('@', 'А') 
            
    if lang == 1:
        regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
        if re.fullmatch(regex, text):
            return text
        else:
            if text == text.lower():
                return text.replace('@', 'a')
            else:
                return text.replace('@', 'A')           


def defis(text, lang):
    return text.replace('-', '')

    
def tilda(text, lang):
    return text.replace('~', '')
    
def bracket(text, lang):
        
    if text.startswith('(') and text.endswith(')'):
        return text
    else:
        return text.replace('(', '').replace(')', '')
    
def bracket_dict(text, lang):
    
    if text.startswith('{') and text.endswith('}'):
        return text
    else:
        return text.replace('{', '').replace('}', '')
        
def star(text, lang):
    return text.replace('*', '')  

# def drop_letter(line, lang):
#     new_line = ''
#     if lang == 0:
#         dict_norm =['а', 'в', 'и', 'к', 'о', 'с', 'у', 'я']
#         for word in line:
#             if len(word) == 1 and word.lowwer() not in dict_norm:
#                 continue
#             new_line += word + ' '
#         return new_line[:-1]
            
#     if lang == 1:
#         en_dict = 'abcdefghijklmnopqrstuvwxyz' 
#         dict_norm =['a', 'n']
#         for word in line:
#             if len(word) == 1 and word.lowwer() not in dict_norm:
#                 continue
#             new_line += word + ' ' 
#         return new_line[:-1]
    
def spell_checker(interval, buffer_lang):
    new_line= ''
    for i in range(len(buffer_lang)):
        text = {"x": [interval[i]]}
        
        if buffer_lang[i] == 0:
            ru_model = requests.post("http://192.168.50.84:8081/model", json=text)
            new_line += ru_model.json()[0][0] + ' '
        else: 
            en_model = requests.post("http://192.168.50.84:8082/model", json=text)
            new_line += en_model.json()[0][0] + ' '
    return new_line[:-1]


# boxes
def post_processing(box):
    all_line = []
    lang = 0
    for line in box:
        
        interval = []
        buffer = ''
        buffer_lang = []
        buf_l = ''
        registers = []
        dict_case = {}
        for text in line.split():
            lang = lang_type(text)
            dict_case[text] = register(text, lang)
            text = email(text, lang)
            #text = defis(text, lang)
            text = bracket(text, lang)  
            text = bracket_dict(text, lang)  
            text = star(text, lang)
        
            if buf_l == '' or buf_l == lang:
                buffer += text + ' '
                buf_l = lang
            else:
                interval.append(buffer[:-1])
                buffer = text + ' '
                buffer_lang.append(lang)
                buf_l = ''

        interval.append(buffer)
        buffer_lang.append(lang)
        
        clean_line = spell_checker(interval, buffer_lang)
        clean_line = clean_line.replace(' @ ', '@')
        new_line = ''
        for text in clean_line.split():
            levenshtein = Levenshtein()
            min_dist = 999999
            best_key = ''
            for key in dict_case.keys():
                if levenshtein.distance(key, text) < min_dist:
                    min_dist = levenshtein.distance(key, text)
                    best_key = key
                    
            
            if dict_case[best_key] == 'up':
                new_line += text.upper() + ' '
            elif dict_case[best_key] == 'low':
                new_line += text.lower() + ' '
            elif dict_case[best_key] == 'title':
                new_line += text.title() + ' '
            else:
                new_line += text + ' '
        all_line.append(new_line[:-1])
    return all_line

@app.route('/get_spell_check', methods=['POST'])
def post():
    box = request.get_json()['box']
    result = post_processing(box)
    return {'result':result}


app.run(host='0.0.0.0', port=8083)