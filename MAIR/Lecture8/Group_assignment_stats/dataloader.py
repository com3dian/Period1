import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

def readData(dataPath, keys):
    '''
    import dataset by sheet names
    '''
    ans = dict()
    for i in keys:
        ans[i] = pd.read_excel(dataPath, sheet_name = keys.index(i))
    return ans

class dataloader:
    '''
    import excel file, transform into networkx, run SIR
    '''    
    def __init__():  
        self.graph = nx.Graph() # network
        
        
    def path2nx(path, sheetName, graphRelation, attributeName):
        '''
        define netwrok with number of followers as attribution
        sheetName: 'tweets', 'tweeters'
        graphRelation: 'replied', 'quoted', 'retweeted'
        attributeName: 'numFollower'
        '''
        dfDict = readData(dataPath, sheetName)
        
        dfTweet = df['tweets']
        dfTweeter = df['tweeters']
        
        timeDifference = list(np.full([len(dfTweet)], np.nan))
        # initial nan list
        for i in trange(len(timeDifference)):
            if dfTweet.loc[i, graphRelation + '_id'] in dfTweet.tweet_id.values:
                # if target node is in graph
                time1 = datetime.strptime(dfTweet.loc[i, 'posted_on'][:-4], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(dfTweet[(dfTweet['tweet_id'] == 
                                           dfTweet.loc[i, graphRelation + '_id'])].posted_on.values[0][:-4], 
                                          '%Y-%m-%d %H:%M:%S')
                
                timeDifference[i] = np.exp(-(time2 - time1).total_seconds()/600)
                # weight scaled
            
        dfTweet['timeDifference'] = timeDifference
        dfSub = dfTweet[[graphRelation + '_id', 'tweet_id', 'timeDifference']]
        
        edgeList = dfSub.dropna(axis = 0, how = 'any').values.tolist()
        self.graph.add_edges_from(edgeList) # add edges with weight
        
        # set node attributes
        attributeDict = dict()
        for node in self.graph.nodes:
            if len(dfTweeter[(dfTweeter.tweeter_id == node)]) > 0:
                attributeDict[node] = dfTweeter[(dfTweeter.tweeter_id == node)].followers_count.values[0]
            else:
                attributeDict[node] = 1
        nx.set_node_attributes(self.graph, attributeDict, name = attributeName)
        
        return self.graph
        
            
        
    def runSIR(epoches, tau, gamma, influencialNodes, tmax, transmissionWeight):
        '''
        epoches: epoches of SIR simulation
        tau, gamma: parameter of SIR, 
                    tau represents transmission rate per edge, 
                    gamma represents recovery rate per node
        tmax:  if 0 then runing to infinity, else the maximum time after which the simulation will stop
        transmissionWeight: a string.
                            the label for a weight given to the edges. 
                            transmission rate is G.adj[i][j][transmission_weight]*tau
        '''
        infectedRate = []
        for i in range(epoches):
            t, S, I, R = EoN.fast_SIR(self.graph, tau, gamma, tmax = tmax, 
                                      initial_infecteds = influencialNodes, 
                                      transmission_weight = transmissionWeight)
            infectedRate.append(R[-1])
        return np.mean(infectedRate)/self.graph.number_of_nodes()
        