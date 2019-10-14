'''
author:xiaohan 

'''
import pandas as pd
import numpy as np
import warnings, time
from sklearn import metrics
import json,os


class bot_follow(object):
    def __init__(self):
        pass

    def to_jsonstr(self, json_data):
        json_obj = {}
        json_obj['data'] = json_data
        json_obj['code'] = 0
        json_obj['msg'] = 'ok'
        json_str = json.dumps(json_obj, ensure_ascii=False)
        return json_str

    def check_name(self,name,embs,user_map):
        #user_map=self.get_name2node()  # return dict:{'aaa'=123
        embs=self.get_embs()
        if  user_map[name] in embs:
            return True
        else:
            return False

    def get_score_dict(self,embs,user_map,bot_name,name_list):
        #user_map=self.get_name2node()  # return dict:{'aaa'=123
        scoredict={}
        for name in name_list:
            if self.check_name(name,embs,user_map):
                scoredict[name]=round(float(self.get_score(embs,user_map[bot_name],user_map[name])),4)
            else:
                scoredict[name]=-1
        return scoredict

    def get_name2node(self):
        file1='user_map.txt'
        file2='user_list.txt'
        print(os.path.abspath('.'))
        file1=os.path.join('./BotProject/pre_data',file1)
        file2=os.path.join('./BotProject/pre_data',file2)
        name2node={}
        id2node={}
        with open(file2,'r') as f:
            i=0
            while True:
                line=f.readline()
                if not line:
                    break
                id2node[line.strip()]=i
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
        
    def get_embs(self,filename='embs.npy'):
        #filename='embs.npy'
        pwd=os.path.join('./BotProject/pre_data/',filename)
        embs=np.load(pwd,allow_pickle=True)
        embs=embs.item()
        return embs

    def get_score(self,embs, node1, node2):
        vector1 = embs[int(node1)]
        vector2 = embs[int(node2)]
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2))
        
        