# %%
def rotate_matrix(arr):
    n = len(arr)
    new_arr = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            new_arr[j][n-i-1] = arr[i][j]
    return new_arr

def new_lock(arr):
    n = len(arr)
    new_arr = [[0]*(3*n) for _ in range(3*n)]
    for i in range(n,2*n):
        for j in range(n,2*n):
            new_arr[i][j] = arr[i-n][j-n]
    return new_arr

def check(arr):
    n = len(arr)
    for i in range(n//3,2*n//3):
        for j in range(n//3,2*n//3):
            if arr[i][j]!=1:
                return False
    return True
def solution(key,lock):
    
    big_lock = new_lock(lock)
    for i in range(4):
        key=rotate_matrix(key)
        for i in range(len(big_lock)-len(key)+1):
            for j in range(len(big_lock)-len(key)+1):
                for m in range(len(key)):
                    for n in range(len(key)):
                        big_lock[i+m][j+n]+=key[m][n]
                if check(big_lock)==True:
                    return True
                for m in range(len(key)):
                    for n in range(len(key)):
                        big_lock[i+m][j+n]-=key[m][n]
    return False

# %%
def solution(key, lock):
    
    
    for rotate_number in range(4):
        expand_lock = expand(lock)
        key = rotate(key)
        for m in range(len(lock)*2):
            for n in range(len(lock)*2):
                for i in range(len(key)):
                    for j in range(len(key)):
                        expand_lock[m+i][n+j]+=key[i][j]
                
                if check(expand_lock)==True:
                    return True
                for i in range(len(key)):
                    for j in range(len(key)):
                        expand_lock[m+i][n+j]-=key[i][j]
    
    return False

def rotate(array):
    rotate_graph =[[0]*len(array) for _ in range(len(array))]
    for i in range(len(array)):
        for j in range(len(array[0])):
            rotate_graph[j][len(array)-i-1] = array[i][j]
            
    return rotate_graph

def expand(array):
    expand_graph = [[0]*(3*len(array)) for _ in range(3*len(array))]
    for i in range(len(array),2*len(array)):
        for j in range(len(array),2*len(array)):
            expand_graph[i][j] = array[i-len(array)][j-len(array)]
            
    return expand_graph

#확장된 열쇠와 key와의 check
def check(array):
    for i in range(len(array)//3,2*len(array)//3):
        for j in range(len(array)//3,2*len(array)//3):
            if array[i][j]!=1:
                return False
    return True

# %%
from collections import deque
N = int(input())
K = int(input())
graph = [[0]*N for _ in range(N)]
for i in range(K):
    m,n = map(int,input().split())
    graph[m-1][n-1]=1

L = int(input())
conversion_arr =[]
for i in range(L):
    X , C = input().split()
    conversion_arr.append((int(X),C))


# %%

def rotate(now,direction):
    left_arr = ['up','right','down','left']
    right_arr = ['up','left','down','right']
    if direction == 'D': 
        for idx,i in enumerate(left_arr):
            if now == i:
                
                now = left_arr[(idx+1)%4]
                return now
    elif direction == 'L':
        for idx,i in enumerate(right_arr):
            if now == i:
                now = right_arr[(idx+1)%4]
                return now
    return now

def move(row,col,toward):
    direction_arr = ['up','right','down','left']
    if toward == 'up':
        row -=1
    elif toward == 'right':
        col+=1
    elif toward == 'down':
        row+=1
    elif toward == 'left':
        col-=1

    return row,col

time= 0
row = 0
col = 0
index=0
direction = 'right'
q = deque()
q.append((row,col))
graph[row][col]=2
while True:
    
    time+=1
    row,col = move(row,col,direction)
    if row<0 or row>=N or col<0 or col>=N:
        break
    if graph[row][col]==2:
        break
    elif graph[row][col]==0:
        prev_row,prev_col=q.popleft()
        graph[prev_row][prev_col]=0
        q.append((row,col))
    elif graph[row][col]==1:
        q.append((row,col))
    
    if index <L and time == conversion_arr[index][0]:
        direction = rotate(direction,conversion_arr[index][1])
        index+=1
    graph[row][col]=2
print(time)

# %%
from itertools import combinations

N , M = map(int,input().split())
graph = []
for i in range(N):
    graph.append(list(map(int,input().split())))

chicken_list =[]
home_list = []
INF = int(1e9)
for i in range(N):
    for j in range(N):
        if graph[i][j] == 2:
            chicken_list.append((i,j))
        if graph[i][j] == 1:
            home_list.append((i,j))
min_total_dist = INF
city_dist = 0

def get_dist(a,b):
    row_a,col_a = a
    row_b,col_b = b
    return abs(row_a-row_b)+abs(col_a-col_b)
for cases in list(combinations(chicken_list,M)):
    city_dist = 0
    for home in home_list:
        home_dist = INF
        for case in cases:
        
            home_dist = min(get_dist(home,case),home_dist)
        print(home,home_dist)
        city_dist+=home_dist
        #각각의 치킨집에 대해서 집마다 치킨 거리 구하는 식
        #city_dist
    # 도시의 치킨 거리 구하기
    min_total_dist = min(min_total_dist,city_dist)

print(min_total_dist)

# %%
def solution(n, arr1, arr2):
    answer = []
    graph1 = [[0]*n for _ in range(n)]
    graph2= [[0]*n for _ in range(n)]
    sum_graph = [[0]*n for _ in range(n)]
    for idx,i in enumerate(arr1):
        graph1[idx][n-len(binary(i)):]=binary(i)
    for idx,i in enumerate(arr2):
        graph2[idx][n-len(binary(i)):]=binary(i)    
    for i in range(n):
        result = ""
        for j in range(n):
            sum_graph[i][j] = int(graph1[i][j])+int(graph2[i][j])
            if sum_graph[i][j]>=1:
                result+="#"
            else:
                result+= " "
        answer.append(result)
    return answer

def binary(x):
    binary_x = ""
    while x>=1:
        
        binary_x+=str(x%2)
        x = x//2
    
    return binary_x

solution(5,[9, 20, 28, 18, 11],[30, 1, 21, 17, 28])

# %%
def binary(x):
    binary_x = ""
    while x>=1:
        print(x,"x")
        print(x%2,"x%2")
        binary_x+=str(x%2)
        x = x//2
    
    return binary_x[::-1]

binary(20)

# %%
print(home_list)
print(chicken_list)

# %%
from itertools import permutations

a = [1,2,3,4]
list(permutations(a,3))

# %%
def solution(n, times):
    times.sort()
    start = times[0]
    end = n*times[0]
    mid = (start+end)//2
    while start<=end:
        print(start,mid,end)
        mid = (start+end)//2
        if check(times,mid,n)==False:
            start = mid+1
        elif check(times,mid,n)==True:
            if check(times,mid-1,n)==False:
                return mid
            else:
                end = mid-1
    return mid    
    
def check(arr,target,n):
    num_people = 0
    for element in arr:
        num_people+= target//element
        if num_people >=n:
            return True
    return False

solution(6,	[7, 10])

# %%
def calculate(a,b):
    numerator = 1
    denominator = 1
    for i in range(a,a-b,-1):
        numerator*=i
    print(numerator)
    for j in range(1,b+1):
        denominator*=j
    print(denominator)
    return numerator/denominator

calculate(5,2)