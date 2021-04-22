# %%
N,M,K,X = map(int,input().split())
graph = [[] for _ in range(M+1)]
for _ in range(M):
    A,B = map(int,input().split())
    graph[A].append(B)

from collections import deque

def bfs(start,graph,M,K):
    dist = [0]*(M+1)
    q = deque()
    q.append(start)
    visited=[False]*(M+1)
    while q:
        now = q.popleft()
        for i in graph[now]:
            if visited[i]==False:
                dist[i] = dist[now]+1
                q.append(i)
                visited[i]=True

    result = []
    for idx,i in enumerate(dist):
        if i == K:
            result.append(idx)
    if len(result)==0:
        print("-1")
    else:
        for i in range(len(result)):

            print(result[i])
bfs(X,graph,M,K)

# %%
from itertools import combinations
from copy import deepcopy
N, M = map(int,input().split())
graph = [[] for _ in range(N)]
for i in range(N):
    graph[i] = list(map(int,input().split()))



# %%
def move(graph,row,col):
    dx = [0,1,0,-1]
    dy = [-1,0,1,0]
    for i in range(4):
        nx = col + dx[i]
        ny = row + dy[i]
        if 0<=nx<M and 0<=ny<N and graph[ny][nx]==0:
            graph[ny][nx]= 2
            move(graph,ny,nx)
zero_list = []
for i in range(N):
    for j in range(M):
        if graph[i][j] == 0:
            zero_list.append((i,j))

block_candidate = list(combinations(zero_list,3))
max_number = 0
for cases in block_candidate:
    new_graph = deepcopy(graph)
    for case in cases:
        new_graph[case[0]][case[1]] =1
    #여기서 다시 시작할 것
    
    for i in range(N):
        for j in range(M):
            if new_graph[i][j]==2:
                move(new_graph,i,j)
    answer = 0
    for i in range(N):
        for j in range(M):
            if new_graph[i][j]==0:
                answer+=1
    max_number = max(answer,max_number)
print(max_number)

# %%
N, K = map(int,input().split())
graph = []
for i in range(N):
    graph.append(list(map(int,input().split())))
S,X,Y = map(int,input().split())


# %%
def spread(graph,row,col,number):
    dx = [0,1,0,-1]
    dy = [-1,0,1,0]

    for i in range(4):
        nx = dx[i] + col
        ny = dy[i] + row
        if 0<=nx < N and 0<=ny <N and graph[ny][nx]==0:
            graph[ny][nx] = number

from copy import deepcopy
time =0
while time<=S:
    time+=1
    new_graph = deepcopy(graph)
    for k in range(1,K+1):
        for i in range(N):
            for j in range(N):
                if graph[i][j]==k:
                    spread(new_graph,i,j,k)
    graph = new_graph
print(graph[X-1][Y-1])


# %%
N = int(input())
number_arr = list(map(int,input().split()))
op_arr = list(map(int,input().split()))
add_list = [0]*op_arr[0]
sub_list = [1]*op_arr[1]
mul_list = [2]*op_arr[2]
div_list = [3]*op_arr[3]
op_list = add_list + sub_list + mul_list + div_list

# %%
from itertools import permutations
max_answer = -int(1e9)
min_answer = int(1e9)
for cases in list(permutations(op_list,N-1)):
    answer = number_arr[0]
    for idx,op in enumerate(cases):
        if op==0:
            answer+=number_arr[idx+1]
        elif op==1:
            answer-=number_arr[idx+1]
        elif op==2:
            answer*=number_arr[idx+1]
        elif op==3:
            if answer<0:
                answer = -(-answer//number_arr[idx+1])
            else:
                answer = answer//number_arr[idx+1]
    max_answer = max(max_answer,answer)
    min_answer = min(min_answer,answer)
print(max_answer)
print(min_answer)
    

# %%
N, L, R = map(int,input().split())
graph = [[] for _ in range(N)]
for i in range(N):
    graph[i]=list(map(int,input().split()))


# %%
print(graph)

# %%
def check(row,col,graph,union,coordinate,visited,change):
    dx = [0,1,0,-1]
    dy = [-1,0,1,0]
    
    for i in range(4):
        print(row,dy[i])
        nx = col + dx[i]
        ny = row + dy[i]

        if 0<=nx<N and 0<=ny<N and not visited[ny][nx] and L<=abs(graph[ny][nx]-graph[row][col])<=R:
            union.append(graph[ny][nx])
            coordinate.append([ny,nx])
            visited[ny][nx]=True
            check(ny,nx,graph,union,coordinate,visited,change)
            change = True

time = 0
while True:
    visited= [[False]*N] *N
    total_change = False
    for i in range(N):
        for j in range(N):
            change = False
            union = []
            coordinate = []
            check(i,j,graph,union,coordinate,visited,change)
            if change == True: 
                total_change= True


            for i in coordinate:
                y,x = i
                graph[y][x]=int(sum(union)/len(union))
    if total_change == False:
        break
    time+=1
print(time)

# %%
a = True
print(a or False)