from flask import Flask, redirect, url_for, request,render_template,jsonify
import json,os,sys
import numpy as np
import pandas as pd

sys.path.append(r'./BotProject')
from BotLink import bot_follow

bf=bot_follow()
app = Flask('to_follow')
#app.config['JSON_AS_ASCII'] = False

def to_follow_main(json_data):
    df=pd.DataFrame()
    #filename = 'output.csv'
    bot_name = json_data['bot']
    name_list=[x.strip() for x in json_data['name_list']]#list of str
    score={}
    embs=bf.get_embs('embs.npy')
    user_map=bf.get_name2node()
    print("bot id=",user_map[bot_name])
    result={}
    if bf.check_name(bot_name,embs,user_map):
        score=bf.get_score_dict(embs,user_map,bot_name,name_list)
        result['score']=score
        result=bf.to_jsonstr(result)
        return result
    else:
        result["Error"]="Bot without embedding, pls choose another bot's name"
        result['score']={}
        return bf.to_jsonstr(result)

def to_follow_debug(json_data):
    df=pd.DataFrame()
    #filename = 'output.csv'
    bot_name = json_data['bot']
    name_list=[x.strip() for x in json_data['name_list']]#list of str
    score={}
    embs=bf.get_embs('embs.npy')
    user_map=bf.get_name2node()
    print("bot id=",user_map[bot_name])
    result={}
    if bf.check_name(bot_name,embs,user_map):
        score=bf.get_score_dict(embs,user_map,bot_name,name_list)
        result['score']=score
        result=bf.to_jsonstr(result)
        return result
    else:
        result["Error"]="Bot without embedding, pls choose another bot's name"
        result['score']={}
        return bf.to_jsonstr(result)

@app.route('/')
def index():
    #return '<h1>Hello World!</h1>'
    msg={}
    msg['test_link']='success'
    return jsonify(msg)

@app.route('/to_follow',methods = ['POST','GET'])
def to_follow():
    print('----- Server Start-----')
    print(request.headers)
    print(request.form)
    print(type(request.json))
    print(request.json)
    return to_follow_main(request.json)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010,debug=True)