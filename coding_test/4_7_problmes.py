# %%
n = int(input())
m = int(input())
INF = int(1e9)
graph = [[INF]*(n+1) for _ in range(n+1)]
for i in range(1,n+1):
    graph[i][i] = 0
for i in range(m):
    a,b,c = map(int,input().split())
    graph[a][b]= c


# %%
for k in range(1,n+1):
    for i in range(1,n+1):
        for j in range(1,n+1):
            graph[i][j] = min(graph[i][j],graph[i][k]+graph[k][j])

for i in range(1,n+1):
    for j in range(1,n+1):
        if graph[i][j] == INF:
            graph[i][j]=0
            print(graph[i][j],end = " ")
        else:
            print(graph[i][j],end = " ")
    print("")


# %%
import heapq
n, m = map(int,input().split())
start = int(input())
INF = int(1e9)
graph= [[] for _ in range(n+1)]
for i in range(m):
    a,b,c = map(int,input().split())
    graph[a].append((c,b))
dist = [INF] *(n+1)
dist[start]= 0
q = []
heapq.heappush(q,(0,start))
while q:
    cost,now = heapq.heappop(q)
    if dist[now]< cost:
        continue
    for j in graph[now]:
        if j[0]+cost<dist[j[1]]:
            dist[j[1]]= j[0]+cost
            heapq.heappush(q,(j[0]+cost,j[1]))
for i in range(1,len(dist)):
    print(dist[i], end=" ")




# %%
import heapq
N = int(input())
graph = []
for i in range(N):
    graph.append(list(map(int,input().split())))
INF = int(1e9)
dist = [[INF]*N for _ in range(N)]
dist[0][0] = graph[0][0]
dx = [-1,0,1,0]
dy = [0,1,0,-1]
q = []
heapq.heappush(q,(dist[0][0],0,0))
while q:
    cost,x,y = heapq.heappop(q)
    if dist[x][y]<cost:
        continue
    for i in range(4):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0<=nx < N and 0<=ny < N:
            if dist[nx][ny] >cost+graph[nx][ny]:
                dist[nx][ny] = cost+graph[nx][ny]
                heapq.heappush(q,(dist[nx][ny],nx,ny))

print(dist[-1][-1])


# %%
a = [[1,2,3,4],[5,6,7,8,],[9,10,11,12]]
a[-1]

# %%
N, M = map(int,input().split())
graph = []
for i in range(N):
    graph.append(list(map(int,input().split())))

tour_list = list(map(int,input().split()))

# %%
parent = [i for i in range(N)]
def find_parent(x,parent):
    if parent[x]!=x:
        parent[x] = find_parent(parent[x],parent)
    return parent[x]
def union(a,b,parent):
    a = find_parent(a,parent)
    b = find_parent(b,parent)
    if a< b:
        parent[b]=a
    else:
        parent[a]=b

for i in range(N):
    for j in range(N):
        if graph[i][j]==1:
            union(i,j,parent)

for i in tour_list:
    if find_parent(tour_list[0],parent)!=find_parent(i,parent):
        print("NO")
        break
print("YES")

# %%
G = int(input())
P = int(input())
graph = [0]*P
for i in range(P):
    graph[i]=int(input())

parent = [i for i in range(N)]
def find_parent(x,parent):
    if parent[x]!=x:
        parent[x] = find_parent(parent[x],parent)
    return parent[x]
def union(a,b,parent):
    a = find_parent(a,parent)
    b = find_parent(b,parent)
    if a< b:
        parent[b]=a
    else:
        parent[a]=b

for i in range(P):
    result = 0
    print(parent)
    if find_parent(i,parent)==i:
        
        union(i,i-1,parent)
    else:
        if find_parent(i,parent)==0:
            break
        else : union(i,find_parent(i,parent),parent)
    result+=1
print(result)

# %%
N,M = map(int,input().split())
import heapq
q = []
parent = [i for i in range(N)]
total_result = 0
result = 0
for i in range(M):



    X,Y,Z = map(int,input().split())
    heapq.heappush(q,(Z,X,Y))
    total_result+=Z


# %%
while q:
    cost,a,b=heapq.heappop(q)
    if find_parent(a,parent)==find_parent(b,parent):
        continue
    else:
        union(a,b,parent)
        result+=cost
        print(a,b)
print(total_result - result)
print(total_result)

# %%
from collections import deque

n = int(input())
score = list(map(int,input().split()))
m = int(input())
change= []
graph = [[0]*n for _ in range(n)]
for i in range(m):
    a,b = map(int,input().split())
    change.append((a,b))


# %%
indegree = [0]*n
for i in range(n):
    for j in range(n):
        if i==j:
            continue
        if score[i]<score[j]:
            graph[i][j] = 1
            
        elif score[i]>score[j]:
            graph[j][i] = 1
            
for i in range(m):
    a,b=change[i]
    if graph[a][b]==1:
        graph[b][a] =1
        
    elif graph[b][a] ==1:
        graph[a][b]=1
for i in range(n):
    for j in range(n):
        if graph[i][j]==1:
            indegree[j]+=1
q = deque()
for i in range(n):
    if indegree[i]==0:
        q.append(i)
result = []
while q:
    if len(q)==0:
        print("Impossible")
        break
    elif len(q)>=2:
        print("?")
        break
    else:
        now=q.popleft()
        result.append(now)
        for i in range(n):
            if graph[now][i]==1:
                indegree[i]-=1
                if indegree[i]==0:
                    q.append(now)
print(result)

# %%
print(q)

# %%
print(graph)

# %%
N, x = map(int,input().split())
arr = list(map(int,input().split()))


def get_right_idx(array,target):
    start = 0
    end = len(array)
    mid = (start+end)//2
    while start<=end:
        mid = (start+end)//2
        if array[mid]<target:
            if mid +1 <len(array):
                start = mid+1
            else:
                return False
        elif array[mid]>target:
            if mid-1 >=0:
                end = mid-1
            else: return False
        elif array[mid]==target:
            if mid+1 < len(array):
                if array[mid+1]>target:
                    return mid
                else:
                    start = mid+1
            else:
                return mid

def get_left_idx(array,target):
    start = 0
    end = len(array)
    mid = (start+end)//2
    while start<=end:
        mid = (start+end)//2
        if array[mid]<target:
            if mid +1 <len(array):
                start = mid+1
            else:
                return False
        elif array[mid]>target:
            if mid-1 >=0:
                end = mid-1
            else: return False
        elif array[mid]==target:
            if mid-1 >=0:
                if array[mid-1]<target:
                    return mid
                else:
                    end = mid-1
            else:
                return mid
if get_right_idx(arr,x) ==False or get_left_idx(arr,x) == False:
    print("-1")
else:
    print(get_right_idx(arr,x)-get_left_idx(arr,x)+1)


# %%
print(right_idx - left_idx +1)

# %%
N = int(input())
arr = list(map(int,input().split()))
def bisect(arr):
    start = 0
    end = N-1
    mid = (start+end)//2
    while start<= end:
        mid = (start+end)//2
        if arr[mid]>mid:
            end = mid-1
        elif arr[mid]<mid:
            start = mid+1
        else:
            return mid
    return -1

print(bisect(arr))

# %%
N,C = map(int,input().split())
arr = []
for i in range(N):
    arr.append(int(input()))


# %%
def install(arr,target,number):
    start = arr[0]
    
    result=1
    while True:
        change = False
        for i in range(idx,len(arr)):
            if arr[i]>=start+target:
                start = arr[i]
                
                result+=1
        if result>=number:
            return True
                
        if result<number:
            return False
        
arr.sort()
start = arr[0]
end = arr[-1]

while start<=end:
    mid = (start+end)//2
    if install(arr,mid,C)==False:
        end = mid -1
    elif install(arr,mid,C)==True:
        start = mid+1
        result = mid
print(result)

# %%
def solution(words, queries):
    
    answer = [0]*len(queries)
    direction = True
    for i,query in enumerate(queries):
        if query[0]=="?":
            direction = True
            right_idx=get_right_idx(query)
            
        else:
            left_idx = get_left_idx(query)
            direction = False
            
        for word in words:
            if direction == True:
                if len(word) != len(query):
                    continue
                for j in range(right_idx+1,len(query)):
                    
                    if word[j] != query[j]:
                        
                        break
                    if j == len(query)-1:
                        answer[i]+=1
                        
            if direction == False:
                if len(word) != len(query):
                    continue
                for j in range(left_idx):
                    if word[j] != query[j]:
                        
                        break
                    if j == left_idx - 1:
                        answer[i]+=1
                        
    return answer

def get_right_idx(query):
    start = 0
    end = len(query)-1
    mid = int((start+end)/2)
    while start<= end:
        if query[mid]!="?":
            end = mid-1
        else:
            if mid==len(query)-1 or query[mid+1]!="?" :
                return mid
            else: start = mid+1
        mid = int((start+end)/2)
    return mid

def get_left_idx(query):
    start = 0
    end = len(query)-1
    mid = int((start+end)/2)
    while start<= end:
        if query[mid]!="?":
            start = mid+1
        else:
            if mid == 0 or query[mid-1]!="?"  :
                return mid
            else: end = mid-1
        mid = int((start+end)/2)
    return mid












    

# %%
a = [[1,2,3,4],[5,6,7,8]]
[1,2,4,3] in a

# %%
#m은 열, n은 행
#목표지점은 n-1,m-1
from collections import deque
def solution(m, n, puddles):
    visited=[[False]*m for _ in range(n)]
    q = deque()
    dp = [[0]*m for _ in range(n)]
    q.append([0,0])
    while q:
        row,col = q.popleft()
        move(row,col,puddles,dp,q,m,n,visited)
        print(dp)
    max_dp = max(dp)
    answer = 0
    for i in dp:
        if i == max_dp:
            answer+=1
    return answer

def move(row,col,puddles,dp,q,m,n,visited):
    dx = [1,0]
    dy = [0,1]
    for i in range(2):
        ny,nx = row+dx[i],col+dy[i]
        if 0<=nx<m and 0<=ny<n and [nx,ny] not in puddles and visited[ny][nx]==False:
            dp[ny][nx] = dp[row][col]+1
            q.append([ny,nx])
            visited[ny][nx]=True