from flask import Flask, redirect, url_for, request,render_template,jsonify
import json,os
import numpy as np
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    #return '<h1>Hello World!</h1>'
    di={}
    di['a']=2
    return jsonify(di)

@app.route('/to_follow',methods = ['POST','GET'])
def login():
    json_data=request.json
    bot_name = json_data['bot']
    name_list=json_data['name_list']#str
    name_list=name_list.strip().split('\n')[:]
    user_map=get_name2node()  # return dict:{'aaa'=123}
    embs=get_embs()
    #score_dict={}
    #score_dict[]={}
    for name in name_list:
        score[name]=get_score(embs,user_map[bot_name],user_map[name])
    di={}
    di['a']=2
    return jsonify(score)

def get_name2node():
    file1='user_map.txt'
    file2='user_list.txt'
    #file1=os.path.join()
    name2node={}
    id2node={}
    with open(file2,'r') as f:
        i=0
        while True:
            line=f.readline()
            if not line:
                break
            id2node[line.strip()[0]]=i
            i+=1
    with open(file1,'r') as f:
        while True:
            line=f.readline()
            if not line:
                break
            tmp=line.strip().split(' ')
            if tmp[0] in id2node:
                name2node[tmp[1]]=id2node[tmp[0]]
            else:
                name2node[tmp[1]]=None
    return name2node #{"name"='node_id'}
    
def get_embs():
    file='embs.npy'
    pwd=file
    embs=np.load(pwd,allow_pickle=True)
    embs=embs.item()
    returm embs

def get_score(embs, node1, node2):
    vector1 = embs[int(node1)]
    vector2 = embs[int(node2)]
    return np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)