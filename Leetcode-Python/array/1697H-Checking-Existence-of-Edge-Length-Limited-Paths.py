'''
Problem Discription:
An undirected graph of n nodes is defined by edgeList, where edgeList[i] = [ui, vi, disi] denotes an edge between nodes ui and vi with distance disi. Note that there may be multiple edges between two nodes.
Given an array queries, where queries[j] = [pj, qj, limitj], your task is to determine for each queries[j] whether there is a path between pj and qj such that each edge on the path has a distance strictly less than limitj .
Return a boolean array answer, where answer.length == queries.length and the jth value of answer is true if there is a path for queries[j] is true, and false otherwise.

Example:
Input: n = 3, edgeList = [[0,1,2],[1,2,4],[2,0,8],[1,0,16]], queries = [[0,1,2],[0,2,5]]
Output: [false,true]
Explanation: The above figure shows the given graph. Note that there are two overlapping edges between 0 and 1 with distances 2 and 16.
For the first query, between 0 and 1 there is no path where each distance is less than 2, thus we return false for this query.
For the second query, there is a path (0 -> 1 -> 2) of two edges with distances less than 5, thus we return true for this query.
'''

### 约束限制(排序)+将点边关系与union结合
def distanceLimitedPathsExist(self, n: int, edgeList: List[List[int]], queries: List[List[int]]) -> List[bool]:
    res = [None] * len(queries)
    edgeList.sort(key=lambda x: x[2])
    queries = sorted([q + [i] for i, q in enumerate(queries)], key=lambda x: x[2]) # 因为要排序，所以存储原有索引
    
    root = list(range(n))
    def find(x): # 判断连通性
        if root[x] != x:
            root[x] = find(root[x])
        return root[x]
    
    def union(x, y): # 构建连通性
        rx, ry = find(x), find(y)
        if rx != ry:
            root[ry] = rx
            return 
    
    i = 0
    for a, b, limit, idx in queries:
        while i < len(edgeList) and edgeList[i][2] < limit:
            x, y, d = edgeList[i]
            union(x, y)
            i += 1
        res[idx] = find(a) == find(b)   
        
    return res