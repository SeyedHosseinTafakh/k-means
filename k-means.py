import os
import werkzeug
from flask import Flask, jsonify , send_from_directory,send_file
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)
import mysql.connector
import json
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from joblib import dump, load

x_global=None

class elbow(Resource):
    def get(self):
        return "salam"
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('number_of_max_cluster',required=True)
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage,location='files')
        args = parser.parse_args()
        file = args['file']
        file.save("file.csv")
        dataset = pd.read_csv('file.csv')
        wcss = []
        for i in range(1,int(args['number_of_max_cluster'])):
            kmeans = KMeans(n_clusters = i , init = 'k-means++',max_iter = 300 ,n_init = 10,random_state = 0)
            kmeans.fit(dataset)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1,int(args['number_of_max_cluster'])),wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        # plt.show()
        plt.savefig('files/plot.png')
        seaborn_file = sns.pairplot(dataset)
        seaborn_file.savefig('files/seaborn.png')

class Train(Resource):
    def get(self):
        return "dont know currently"

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('k')
        args = parser.parse_args()
        dataset = pd.read_csv('file.csv')
        kmeans = KMeans(n_clusters=int(args['k']) , init='k-means++',max_iter = 300 , n_init=10,random_state = 0)
        y_kmeans = kmeans.fit_predict(dataset)
        # print(type(y_kmeans))
        dataset['labels'] = y_kmeans
        # print (dataset)
        seaborn_file =sns.pairplot(dataset,hue='labels',diag_kind='hist')
        seaborn_file.savefig('files/trained.png')
        dataset = dataset.drop(columns='labels')
        x = dataset.columns.to_list()
        global x_global
        x_global = x
        ret = {}
        ret['columns'] = x
        ret['image'] = 'trained.png'
        dump(kmeans, 'k-means.joblib')
        return ret

class predict(Resource):
    def post(self):
        global x_global
        parser = reqparse.RequestParser()
        for x in x_global:
            parser.add_argument(name=x,required=True)
        args = parser.parse_args()
        kmeans = load('k-means.joblib')
        # print (kmeans.get_params())
        values = []
        for ar in args:
            values.append(args[ar])
        y = kmeans.predict([values])
        return y.tolist()[0]


api.add_resource(elbow, '/')
api.add_resource(predict, '/predict')
api.add_resource(Train, '/train')

class add_resource:
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def add_argument(self,name):
        self.parser.add_argument(name,required=True)
    def return_arg(self):
        args = self.parser.parse_args()
        return args
@app.route('/download', methods=['GET', 'POST'])
def download():
    parser = reqparse.RequestParser()
    parser.add_argument('file_name',required=True)
    args = parser.parse_args()
    path = 'files/'+ args['file_name']
    return send_file(path,as_attachment=True)











# app.run(host='192.168.43.7',debug=True)
app.run(debug=True)


