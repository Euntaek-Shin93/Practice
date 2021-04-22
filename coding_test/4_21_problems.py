

n = int(input())
dp = [0]*(n+1)
dp[1]=1
dp[2]=3
for i in range(3,n+1):
    dp[i]=dp[i-1]+(2*dp[i-2])
    dp[i]=dp[i]
print(dp)

#가희와 3단 고음
N,A,D = map(int,input().split())
arr = list(map(int,input().split()))
answer =0
for number in arr:
    if number == A:
        answer+=1
        A+=D
print(answer)

N = int(input())
arr = list(map(int,input().split()))
INF = int(1e6)
dist = [INF]*len(arr)
dist[-1]=0
for i in range(len(arr)-2,-1,-1):
    now=arr[i]
    for j in range(i,i+now+1):
        if j==len(arr):
            break
        dist[i]=min(dist[j]+1,dist[i])
if dist[0]>=INF:
    print(-1)
else:
    print(dist[0])

# 우리집엔 도서관이 있어
# 부분증가수열

N = int(input())
arr =[]
for i in range(N):
    arr.append(int(input()))
max_number = arr[0]
answer =1
for i in range(1,len(arr)):
    if max_number<arr[i]:
        if max_number+1==arr[i]:
            answer+=1
            max_number = arr[i]
        else:
            max_number = arr[i]
            answer =1
print(len(arr)-answer)

# 아기 상어2
# 한 방향으로만 가는 버전
def move(row,col,dir,graph,value):
    
    dx = [0,1,0,-1,1,1,-1,-1]
    dy = [-1,0,1,0,-1,1,1,-1]

    for i in range(8):
        if dir == i:
            nx = col+dx[i]
            ny = row+dy[i]
            if 0<=nx<M and 0<=ny<N:
                if graph[ny][nx]==0:
                    graph[ny][nx]=value
                else:
                    graph[ny][nx]=min(value,graph[ny][nx])
                move(ny,nx,dir,graph,value+1)
N, M = map(int,input().split())
graph = []
target = []
for _ in range(N):
    graph.append(list(map(int,input().split())))
for i in range(N):
    for j in range(M):
        if graph[i][j]==1:
            target.append((i,j))
value = 1
direction = ['U','R','D','L','UR','DR','DL','UL']
for i,j in target:
    for k in range(8):
        move(i,j,k,graph,value)

answer = 0
for i in range(N):
    for j in range(M):
        answer = max(answer,graph[i][j])
print(answer)



# 자유자재로 갈 수 있는 버전
from collections import deque
def move(row,col,graph,value,visited):
    
    dx = [0,1,0,-1,1,1,-1,-1]
    dy = [-1,0,1,0,-1,1,1,-1]

    for i in range(8):
        
        nx = col+dx[i]
        ny = row+dy[i]
        if 0<=nx<M and 0<=ny<N and visited[ny][nx] ==False:
            if graph[ny][nx]==0:
                graph[ny][nx]=value
            else:
                graph[ny][nx]=min(value,graph[ny][nx])
            visited[ny][nx]=True
            q.append((ny,nx,value+1))
N, M = map(int,input().split())
graph = []
target = []
for _ in range(N):
    graph.append(list(map(int,input().split())))
for i in range(N):
    for j in range(M):
        if graph[i][j]==1:
            target.append((i,j))

value = 1
for i,j in target:
    q = deque()
    q.append((i,j,value))
    visited = [[False]*(M) for _ in range(N)]
    visited[i][j]=True
    while q:
        now_row,now_col,now_value = q.popleft()
        move(now_row,now_col,graph,now_value,visited)    

answer = 0
for i in range(N):
    for j in range(M):
        answer = max(answer,graph[i][j])
print(answer)

# 자원 캐기
# bfs를 사용하니 deque에 메모리를 많이 잡음
# 이럴 경우는 dp로 되는지 확인
from collections import deque
def move(row,col,graph,value):
    dx = [1,0]
    dy = [0,1]
    for i in range(2):
        nx = col + dx[i]
        ny = row + dy[i]
        if 0<=nx<M and 0<=ny<N:
            if graph[ny][nx]==1:
                q.append((ny,nx,value+1))
            else:
                q.append((ny,nx,value))
N , M = map(int,input().split())
graph = []
for _ in range(N):
    graph.append(list(map(int,input().split())))

q = deque()
if graph[0][0]==0:
    q.append((0,0,0))
else:
    q.append((0,0,1))
while q:
    now_row,now_col,now_value = q.popleft()
    move(now_row,now_col,graph,now_value)


print(now_value)

from copy import deepcopy
N , M = map(int,input().split())
graph = []
for _ in range(N):
    graph.append(list(map(int,input().split())))
dp = [0]*M
new_dp= [0]*M
for i in range(N):
    for j in range(M):
        if j-1>=0:
            new_dp[j]= max(dp[j],new_dp[j-1])+graph[i][j]
        else:
            new_dp[j]= dp[j]+graph[i][j]
    dp = deepcopy(new_dp)
    new_dp [0]*M
print(max(new_dp))