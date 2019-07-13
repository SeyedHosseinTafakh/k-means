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



api.add_resource(elbow, '/')

@app.route('/download', methods=['GET', 'POST'])
def download():
    parser = reqparse.RequestParser()
    parser.add_argument('file_name',required=True)
    args = parser.parse_args()
    path = 'files/'+ args['file_name']
    return send_file(path,as_attachment=True)
# app.run(host='192.168.43.7',debug=True)
app.run(debug=True)
