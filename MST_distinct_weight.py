#!/usr/bin/env python
# coding: utf-8


import time
from pyspark import SparkContext

#sc = SparkContext('spark://54.208.255.5:7077')
sc = SparkContext('yarn')
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Boruvka_MST_algorithm").getOrCreate()
sc.addPyFile("s3a://rogerzhuo/graphframes-0.6.0-spark2.3-s_2.11.jar")

from graphframes import *
from pyspark.sql.functions import *


path_to_file="s3a://rogerzhuo/edges_generate_6.csv"
edge_dataframe = sc.textFile(path_to_file)

header = edge_dataframe.first()#第一行 
edge_dataframe = edge_dataframe.filter(lambda row:row != header)#删除第一行
edge_dataframe = edge_dataframe.map(lambda line: line.split(',')).map(lambda edge:(edge[0], int(edge[1]),int(edge[2])))

begin_vex = edge_dataframe.map(lambda line: (line[0], line[0])).distinct()


v = spark.createDataFrame(begin_vex.collect(), ["id", "label"])
e = spark.createDataFrame(edge_dataframe.collect(), ["src", "dst", "weight"])

# Create a GraphFrame
g = GraphFrame(v, e)




mst = spark.createDataFrame([['', '', '']], ["src", "dst", "weight"])



class QuickFind(object):
    id=[]
    count=0
    
    def __init__(self,n):
        self.count = n
        i=0
        while i<n:
            self.id.append(i)
            i+=1
            
    def connected(self,p,q):
        return self.find(p) == self.find(q)
    
    def find(self,p):    
        return self.id[p]
    
    def union(self,p,q):
        idp = self.find(p)
        if not self.connected(p,q):
            for i in range(len(self.id)):
                if self.id[i]==idp: # 将p所在组内的所有节点的id都设为q的当前id
                    self.id[i] = self.id[q] 
            self.count -= 1


start = time.clock()
while g.vertices.select('label').distinct().count() > 1:
    filter_df = g.find("(a)-[e]->(b)").filter("a.label != b.label").select("e.*")
    filter_df.cache()
    
    inter_graph = GraphFrame(g.vertices, filter_df)
    
    min_edges = inter_graph.triplets.groupBy('src.label').agg({'edge.weight': 'min'}).withColumnRenamed('min(edge.weight AS `weight`)', 'min_weight')

    final_edges = min_edges.join(inter_graph.triplets, (min_edges.label == inter_graph.triplets.src.label)                              & (min_edges.min_weight == inter_graph.triplets.edge.weight))                            .select(col('src.id').alias('src'), col('dst.id').alias('dst'), col('min_weight').alias('weight'))
    final_edges.cache()
    
    edges_rdd = final_edges.rdd.map(lambda item: (int(item.src), int(item["dst"])))
    
    mst = mst.union(final_edges).distinct().filter("src != ''")

    num_edges = mst.select('src').distinct().count()
    qf = QuickFind(num_edges)
    for item in edges_rdd.collect():
        qf.union(item[0],item[1])

    connected_rdd = sc.parallelize([str(x) for x in qf.id])
    id_rdd = sc.parallelize(range(num_edges))
    vertis = list(zip(id_rdd.collect(), connected_rdd.collect()))
    new_vertis = spark.createDataFrame(vertis, ['id', 'label'])

    g = GraphFrame(new_vertis, g.edges)
    g.cache()

elapsed = (time.clock() - start)
#data = [elapsed]  
#time = sc.parallelize(data) 
#time.rdd.saveAsTextFile('time')
print(elapsed)
#mst.rdd.saveAsTextFile('mst')


