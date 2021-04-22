# %%
a = '1234'
b = int(a)
print(b)

# %%
a = "-1 -2 -3 -4"
b = list(map(int,a.split()))
print(b)

# %%
import heapq
def move(row,col,direction):
    d = ['U','R','D','L']
    index = 0
    for idx,i in enumerate(d):
        if i == direction:
            index = idx
            break
    dx = [0,1,0,-1]
    dy = [-1,0,1,0]
    nx = col + dx[index]
    ny = row + dy[index]
    return ny,nx
def solution(board):
    INF = int(1e9)
    N = len(board)
    answer = 0
    q = []
    dp = [[INF]*N for _ in range(N)]
    d = ['U','R','D','L']
    heapq.heappush(q,[0,'R',0,0])
    heapq.heappush(q,[0,'D',0,0])
    dp[0][0] = 0
    while q:        
        cost,direction,row_now, col_now = heapq.heappop(q)
        
        if dp[row_now][col_now] < cost:
            continue
        for i in d:
            ny,nx = move(row_now,col_now,i)
            if nx<0 or ny <0 or N<=nx or N<= ny or board[ny][nx]==1:
                continue
            if i ==direction:
                if dp[ny][nx]>= cost+100:
                        dp[ny][nx]= cost + 100
                        heapq.heappush(q,[cost+100,i,ny,nx])
            if i!=direction:
                if dp[ny][nx]>= cost+600:
                    dp[ny][nx]= cost + 600
                    heapq.heappush(q,[cost+600,i,ny,nx])
        
        
    answer = dp[-1][-1]   
    print(dp)
    return answer

# %%
solution([[0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
[1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
[0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
[0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

# %%
[[0, 100, 1000000000, 1500, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000],
 [100, 700, 800, 900, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000],
 [1000000000, 800, 1400, 1500, 1600, 1000000000, 1000000000, 1000000000, 1000000000, 1000000000],
 [1500, 900, 1500, 1600, 1700, 1800, 1000000000, 1000000000, 1000000000, 1000000000], 
 [1600, 1000, 1600, 1700, 1000000000, 2400, 1000000000, 1000000000, 1000000000, 1000000000]
 [1700, 1100, 1000000000, 1800, 1000000000, 1000000000, 5100, 1000000000, 1000000000, 1000000000],
 [2300, 1000000000, 2500, 1900, 1000000000, 5600, 5000, 5600, 1000000000, 6300],
 [1000000000, 3700, 3100, 1000000000, 4500, 5100, 4900, 5500, 5600, 5700], 
 [3900, 3800, 3200, 3800, 3900, 1000000000, 4800, 1000000000, 6200, 6300],
 [1000000000, 3900, 3300, 3900, 4000, 4100, 4200, 4300, 1000000000, 6400]]

# %%
a = "11:11.22"
b = a.replace(":","")

c = float(b)
c+0.1

# %%
def solution(n, results):
    INF = int(1e9)
    graph = [[INF]*(n+1) for _ in range(n+1)]
    for i in range(0,n+1):
        graph[i][i] = 0
    for result in results:
        a,b = result
        graph[a][b] = 1
    for k in range(1,n+1):
        for i in range(1,n+1):
            for j in range(1,n+1):
                graph[a][b] = min(graph[a][k] + graph[k][b],graph[a][b])
    result = []
    for i in range(1,n+1):
        for j in range(1,n+1):
            if graph[i][j]>=INF and graph[j][j]>= INF:
                break
            if j ==n:
                if graph[i][j]< INF or graph[j][i]<INF:
                    result.append(i)
    
    answer = len(result)                
    return answer



# %%
print(solution(5, [[4, 3], [4, 2], [3, 2], [1, 2], [2, 5]]), 2)
print(solution(7, [[4, 3], [4, 2], [3, 2], [1, 2], [2, 5], [5,6], [6,7]]), 4)
print(solution(6, [[1,2], [2,3], [3,4], [4,5], [5,6]]), 6)
print(solution(5, [[1, 4], [4, 2], [2, 5], [5, 3]]), 5)
print(solution(5, [[3, 5], [4, 2], [4, 5], [5, 1], [5, 2]]), 1)
print(solution(3, [[1,2],[1,3]]), 1)
print(solution(6, [[1,6],[2,6],[3,6],[4,6]]), 0)
print(solution(8, [[1,2],[3,4],[5,6],[7,8]]),0)
print(solution(9, [[1,2],[1,3],[1,4],[1,5],[6,1],[7,1],[8,1],[9,1]]), 1)
print(solution(6, [[1,2],[2,3],[3,4],[4,5],[5,6],[2,4],[2,6]]), 6)
print(solution(4, [[4,3],[4,2],[3,2],[3,1],[4,1], [2,1]]), 4)
print(solution(3,[[3,2],[3,1]]), 1)
print(solution(4, [[1,2],[1,3],[3,4]]), 1)

# %%
2.0 == 2

# %%
def solution(dartResult):
    index = 0
    score_list = []
    numbers= ['0','1','2','3','4','5','6','7','8','9']
    bonus= ['S','D','T']
    option = ['*','#']
    option_list = []
    while index<=len(dartResult)-1:
        print(index)
        print(score_list)
        if dartResult[index:index+2]=='10':
            score_list.append(10)
            index+=1
        elif dartResult[index] in numbers:
            score_list.append(int(dartResult[index]))
            
        elif dartResult[index] in bonus:
            print("?")
            if dartResult[index]=='S':
                score_list[-1]*=1
            elif dartResult[index]=='D':
                score_list[-1]=score_list[-1]**2
                print("?")
            elif dartResult[index]=='T':
                score_list[-1]=score_list[-1]**3
            
        elif dartResult[index] in option:
            if dartResult[index] =='*':
                if len(score_list) == 1:
                    score_list[-1]*=2
                else:
                    score_list[-1]*=2
                    score_list[-2]*=2
            elif dartResult[index] == '#':
                score_list[-1]*=-1
        index+=1
    answer = sum(score_list)
    return answer

solution('10D10D*')
#400

# %%
a = 'abc'
b = list(a)
b
'd' in b

# %%
def solution(n):
    graph = [[0]*n for _ in range(n)]
    row =0
    col = 0
    arr = ['down','right','left-up']
    direction_idx= 0
    direction = arr[direction_idx]
    length = (n*(n+1))//2
    
    for i in range(1,length+1):
        graph[row][col]=i
        
        ny,nx=move(row,col,direction)
        
        if ny<0 or ny>=n or nx<0 or nx>=n:
        
            direction_idx= (direction_idx+1)%3
            direction = arr[direction_idx]
            row,col=move(row,col,direction)
            
        else:
            row,col= ny,nx
    answer = []
    print(graph)
    for i in range(n):
        for j in range(i+1):
            answer.append(graph[i][j])
    return answer

def move(row,col,direction):
    if direction =='down':
        row+=1
    elif direction =='right':
        col+=1
    elif direction =='left-up':
        row-=1
        col-=1
    return row,col

solution(4)