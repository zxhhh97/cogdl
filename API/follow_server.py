from flask import Flask, redirect, url_for, request,render_template,jsonify
import json,os,sys
import numpy as np
import pandas as pd

sys.path.append(r'./BotProject')
from BotLink import bot_follow

bf=bot_follow()
app = Flask('to_follow')
#app.config['JSON_AS_ASCII'] = False
app.secret_key='\x03\x8do\xea\xbfuQ\xbbW\xdd\xace\x85\xa6\x8bV\x1b\x1fg\xf4\xe3Ke\x0f'

def to_follow_main(json_data):
    file_mid = 'nameid.csv'
    bot_name = json_data['bot']
    name_list=[x.strip() for x in json_data['name_list']]#list of str
    score={}
    embs=bf.get_embs('embs.npy')
    user_map=bf.get_name2node()
    
    # write input name list 
    index=['bot']+['target_'+str(i) for i in range(len(name_list))]
    df=pd.DataFrame([bot_name]+name_list,index=index,columns=['name'])
    df.to_csv(file_mid)
    
    # get result
    result={}
    if bf.check_name(bot_name,embs,user_map):
        score=bf.get_score_dict(embs,user_map,bot_name,name_list)
        df_score=pd.DataFrame(pd.Series(score),columns=['Score'])
        df_score=df_score.sort_index(by = 'Score',axis = 0,ascending = False)
        df_score.to_csv('score_out.csv')
        result['score']=score
        result=bf.to_jsonstr(result)
        return result
    else:
        result["Error"]="Bot without embedding, pls choose another bot's name"
        result['score']={}
        return bf.to_jsonstr(result)

def to_follow_traindata(json_data):
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