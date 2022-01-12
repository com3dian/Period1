import networkx as nx
import numpy as np

def leaderRank(Graph, alpha, weightName, withWeight):
    '''
    Graph: networkx graph with edgeWeight
    alpha: damping parameter for PageRank, default=0.85.
    weightName: weight name
    withWeight: if True then consider weight
    return: pagerank dictionary, dictionary of nodes with PageRank as value
    '''
    Graph = Graph.copy()
    num_nodes = Graph.number_of_nodes()
    nodes = Graph.nodes()
    # get number of nodes
    Graph.add_node('extraLeaderNode')
    # add node 0
    color=nx.get_edge_attributes(G, weightName)
    meanEdgeWeight = np.mean(list(color.values()))
    # get mean edge weight of graph
    for node in nodes:
        nx.add_path(Graph, [0, node, 0], edgeWeight = meanEdgeWeight)
        
    if withWeight:
        return nx.pagerank(Graph, alpha = alpha, weight = weightName)
    else:
        return nx.pagerank(Graph, alpha = alpha, weight = None)

def mixedDegreeDecomposition(Graph, Lambda, withWeight):
    '''
    When λ = 0, the MDD method coincides with the k-shell method. 
    When λ = 1, the MDD method is equivalent to the degree centrality method.
    -----------------
    Graph: directed networkx graph with edgeWeight
    Lambda: tenable parameter between 0 and 1
    withWeight: if True then consider weight
    '''
    
    graphToRemove = Graph.to_undirected()
    
    shellRank = dict()
    MDDRank = dict()
    degreeAll = dict(graphToRemove.degree)

    kShell = 0
    minKmValue = 0
    while len(graphToRemove.nodes) > 0:
        shellRank[kShell] = []
        while minKmValue <= kShell:
            minKmValue = len(Graph.nodes)
            for node in graphToRemove.nodes:
                
                kmValue = graphToRemove.degree(node) + Lambda*(degreeAll[node] - graphToRemove.degree(node))
                # compute kmValue
                if kmValue < minKmValue:
                    minKmValue = kmValue
                    # update minKmValue
                
                if kmValue <= kShell:
                    # check for nodes to remove
                    shellRank[kShell].append(node)
                    MDDRank[node] = kmValue
        
            graphToRemove.remove_nodes_from(shellRank[kShell])
            # remove nodes
        kShell += 1
    
    return shellRank, MDDRank
    
def randomizedInfluencePathsSelection(Graph, beta, T, theta, withWeight, withUniformWeight):
    '''
    Graph: directed graph with weight
    beta: probability of selecting random edge in Graph
    T: threshold of random beta-graph
    theta: rounds of iteration
    withWeight: if True then consider weight
    withUniformWeight: with uniform weight
    '''
    
    RIPSRank = dict()
    for node in Graph.nodes:
        RIPSRank[node] = 0
    
    edgeList = list(Graph.edges.data("weight"))
    for i in range(theta):
        # select beta edge
        newEdges = []
        for j in range(len(Graph.edges)):
            randomNumber = np.random.rand()
            if randomNumber < beta * (withWeight * Graph.edges[edgeList[j]]['weight'] + 1 - withWeight):
                newEdges.append(edgeList[j])
        betaGraph = nx.Graph()
        betaGraph.add_weighted_edges_from(newEdges)
        
        connectedComponentGraph = [betaGraph.subgraph(c).copy() for c in nx.connected_components(betaGraph)]
        for connectedComponent in connectedComponentGraph:
            connectedDirectedComponent = nx.DiGraph(connectedComponent)
            if len(connectedComponent.nodes) > T:
                for node in connectedComponent.nodes:
                    if withUniformWeight:
                        RIPSRank[node] += 1
                    else:
                        RIPSRank[node] += \
                        len(connectedComponent.nodes) * beta * connectedComponent.out_degree(node)
    
    return RIPSRank

def kShellIterationFactor(Graph):
    '''
    Graph: networkx graph without edgeWeight
    '''
    shellRank = dict()
    KSiFRank = dict()
    graphToRemove = Graph.to_undirected()
    coreNumber = nx.core_number(graphToRemove)
    # core number
    
    kShell = 0
    
    while len(graphToRemove.nodes) > 0:
        minDegree = 0
        mTurn = 0
        shellRank[kShell] = dict()
        # do kshell
        
        while minDegree <= kShell:
            minDegree = len(Graph.nodes)
            shellRank[kShell][mTurn] = []
            # do iteration
            for node in graphToRemove.nodes:
                if graphToRemove.degree(node) <= kShell:
                    # check for nodes to remove
                    shellRank[kShell][mTurn].append(node)
                    
                if graphToRemove.degree(node) < minDegree:
                    # update minimum degree
                    minDegree = graphToRemove.degree(node) 
                KSiFRank[node] = mTurn
            
            graphToRemove.remove_nodes_from(shellRank[kShell][mTurn]) # remove nodes
            mTurn += 1
            
        for nturn in range(mTurn):
            for node in shellRank[kShell][nturn]:
                KSiFRank[node] = (1 + KSiFRank[node]/mTurn) * coreNumber[node]
        
        kShell += 1
    
    return shellRank, KSiFRank

def Decomposition(Graph, beta, attributeName):
    '''
    Graph: networkx graph with edgeWeight
    beta: scale rate
    attributeName: string, node attribute
    '''
    DRank = dict()
    shellRank = dict()
    for node in Graph.nodes:
        DRank[node] = 0
    
    graphToRemove = nx.DiGraph(Graph)
    kShell = 0
    while len(graphToRemove.nodes) > 0:
        shellRank[kShell] = []
        for node in graphToRemove.nodes:
            # check for nodes to remove
            if graphToRemove.out_degree(node) == 0:
                shellRank[kShell].append(node)
                
                # update weight
                for nodePredecessors in graphToRemove.predecessors(node):
                    DRank[nodePredecessors] += graphToRemove.nodes[node][attributeName] * \
                                               (graphToRemove.edges[nodePredecessors,node]['weight']) ** beta
                    
        graphToRemove.remove_nodes_from(shellRank[kShell]) # remove nodes
        kShell += 1
    return DRank, kShell