# %%
parent = [i for i in range(n)]
def find_parent(x,parent):
    if parent[x]!= x:
        parent[x] = find_parent(parent[x],parent)
    return parent[x]
def union(a,b):
    a = find_parent(a,parent)
    b = find_parent(b,parent)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

# %%
def solution(n, computers):
    answer = 0
    visited = [0]*n
    number = 1
    for i in range(n):
        dfs(i,computers,visited,number)
        number+=1
        print(visited)
    answer = len(set(visited))
    return answer

def dfs(start,graph,visited,number):
    now = start
    print(now)
    
    
    for i in graph[now]:
        if visited[i] != 0:
            continue
        else:
            visited[i] = number
            dfs(i,graph,visited,number)
            
    
           

# %%
def solution(begin, target, words):
    now = begin
    answer = 0
    number = 1
    visited = [0]*len(words)
    result=change_check(now,words,number,visited)
    while True:
        
        number+=1
        if target in result:
            return max(visited)
        for i in result:
            new_result = change_check(i,words,number,visited)
        result = new_result
    

def change_check(word,words,number,visited):
    now = word
    result = []
    
    for idx,i in enumerate(words):
        diff=0
        if visited[idx]!=0:
            continue
        for j in range(len(i)):
            if now[j] !=i[j]:
                diff+=1
        if diff==1:
            result.append(i)
            visited[idx] = number
    return result
    
solution("hit",	"cog"	,["hot", "dot", "dog", "lot", "log", "cog"]	)

# %%
graph = [["ICN", "SFO"], ["ICN", "ATL"], ["SFO", "ATL"], ["ATL", "ICN"], ["ATL","SFO"]]
graph.sort()
graph

# %%
arr = ['1234','2']
a = '1234'
b = '1'

print(arr[0][:len(b)])

# %%
arr = ["12","123","1235","567","88"]
result = sorted(arr)
result

# %%

def solution(phoneBook):
    phoneBook = sorted(phoneBook)

    for a in zip(phoneBook, phoneBook[1:]):
        print(a)
        
            
    return True
solution(["12","123","1235","567","88"])

# %%
a = [1,2,3,4]
b = [6,7,8,9]

print(list(zip(a,b)))

# %%
def solution(clothes):
    dic = dict()
    
    for name, kind in clothes:
        if dic.get(kind) == None:
            dic[kind] = 1
            
        else:
            dic[kind]+=1
            
    return dic
solution([["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]])

# %%
result = solution([["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]]).values()
print(result)
answer = 1
for i in result:
    answer*=(i+1)

print(answer)

# %%

def solution(clothes):
    from collections import Counter
    from functools import reduce
    cnt = Counter([kind for name, kind in clothes])
    return cnt

solution([["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]])


# %%

def solution(numbers):
    from itertools import permutations
    answer = ''
    n = len(numbers) 
    
    arr = [i for i in range(n)]
    seq=list(permutations(arr,n))
    print(seq)
    for i in seq:
        for j in i:
            answer+=str(j)
        result.append()
    result.sort(reverse=True)
    return result[0]



# %%
a = ['6', '910', '2']
b = sorted(a,key= lambda x: x[0])
print(b)

# %%
def solution(triangle):
    dp = [0]*10
    for i in range(len(triangle)):
        new_dp = [0]*(len(triangle[i])+1)
        for j in range(len(triangle[i])):
            
            if j==0:
                new_dp[j] = dp[j]+triangle[i][j]
            elif j==len(triangle[i])-1:
                new_dp[j] = dp[j-1]+triangle[i][j]
            else:
                new_dp[j] = max(dp[j-1]+triangle[i][j] , dp[j]+triangle[i][j])
        dp = new_dp
        
    answer = max(dp)
    return answer
solution([[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]])

# %%
from collections import deque

q = deque()

sum(q)

# %%
from collections import deque
def solution(progresses, speeds):
    queue = deque(progresses)
    queue_speed = deque(speeds)
    time = 0
    exit_number = 0
    answer = []
    while queue:
        exit_number = 0
        while queue:
            
            if queue[0]>=100:
                queue.popleft()
                queue_speed.popleft()
                exit_number+=1
            else:
                break
        if exit_number !=0:
            answer.append(exit_number)
        if len(queue)==0:
            break
        if (100-queue[0])%speeds[0]==0:
            time+= (100-queue[0])//queue_speed[0]
        if (100-queue[0])%speeds[0]!=0:
            time+= (100-queue[0])//queue_speed[0] +1
        for idx, i in enumerate(queue):
            queue[idx] += (time*queue_speed[idx])
        
    return answer

# %%
from collections import deque


def solution(progresses, speeds):
    q = deque(progresses)
    
    speed_q = deque(speeds)
    answer = []
    while q:
        
        if (100-q[0])%speed_q[0]:
            time = (100 - q[0])//speed_q[0]
        else:
            time = (100 - q[0])//speed_q[0]+1
        for i in range(len(q)):
            q[i]+=time*speed_q[i]
        result=0
        while q:
            
            if q[0]<100:
                break
            else:
                q.popleft()
                speed_q.popleft()
                result+=1
        answer.append(result)
    return answer
solution([93, 30, 55],	[1, 30, 5])

# %%
a = [1,2,1,3,4]
del a[0]
a

# %%
import heapq
def solution(n, edge):
    visited = [False]*(n+1)
    start = 1
    INF = int(1e9)
    distance = [0]*(n+1)
    
    visited[start] = True
    q=[start]
    answer = 0
    while q:
        now = heapq.heappop(q)
        for i in range(len(edge)):
            if now in edge[i]:
                
                if edge[i][1]== now and visited[edge[i][0]]==False:
                    visited[edge[i][0]]=True
                    distance[edge[i][0]]=distance[now]+1
                    heapq.heappush(q,edge[i][0])
                    
                elif edge[i][0] == now and visited[edge[i][1]]==False:
                    visited[edge[i][1]]=True
                    distance[edge[i][1]]=distance[now]+1
                    heapq.heappush(q,edge[i][1])
                    
    distance.sort(reverse=True)
    
    for i in range(len(distance)):
        if distance[i]==distance[1]:
            answer+=1
    return answer

# %%
a = [[1,2],[3,4],[1,2]]
a.remove([1,2])
a