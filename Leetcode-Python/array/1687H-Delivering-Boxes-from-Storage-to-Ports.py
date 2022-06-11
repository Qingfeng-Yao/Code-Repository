'''
Problem Discription:
You have the task of delivering some boxes from storage to their ports using only one ship. However, this ship has a limit on the number of boxes and the total weight that it can carry.
You are given an array boxes, where boxes[i] = [ports​​i​, weighti], and three integers portsCount, maxBoxes, and maxWeight.
ports​​i is the port where you need to deliver the ith box and weightsi is the weight of the ith box.
portsCount is the number of ports.
maxBoxes and maxWeight are the respective box and weight limits of the ship.
The boxes need to be delivered in the order they are given. The ship will follow these steps:
The ship will take some number of boxes from the boxes queue, not violating the maxBoxes and maxWeight constraints.
For each loaded box in order, the ship will make a trip to the port the box needs to be delivered to and deliver it. If the ship is already at the correct port, no trip is needed, and the box can immediately be delivered.
The ship then makes a return trip to storage to take more boxes from the queue.
The ship must end at storage after all the boxes have been delivered.
Return the minimum number of trips the ship needs to make to deliver all boxes to their respective ports.

Example:
Input: boxes = [[1,1],[2,1],[1,1]], portsCount = 2, maxBoxes = 3, maxWeight = 3
Output: 4
Explanation: The optimal strategy is as follows: 
- The ship takes all the boxes in the queue, goes to port 1, then port 2, then port 1 again, then returns to storage. 4 trips.
So the total number of trips is 4.
Note that the first and third boxes cannot be delivered together because the boxes need to be delivered in order (i.e. the second box needs to be delivered at port 2 before the third box).
'''

## other solution
### sliding window(有约束限制) + dp
def boxDelivering(self, boxes: List[List[int]], portsCount: int, maxBoxes: int, maxWeight: int) -> int:
    n = len(boxes)
    need = j = lastj = 0
    dp = [0] + [float('inf')] * n
    for i in range(n):
        ## 关联j和lastj + 条件限制(一次装载)
        while j < n and maxBoxes > 0 and maxWeight >= boxes[j][1]:
            maxBoxes -= 1
            maxWeight -= boxes[j][1]
            if j == 0 or boxes[j][0] != boxes[j - 1][0]:
                lastj = j
                need += 1
            j += 1

        dp[j] = min(dp[j], dp[i] + need + 1)
        dp[lastj] = min(dp[lastj], dp[i] + need)
        
        ## 关联i
        maxBoxes += 1
        maxWeight += boxes[i][1]
        if i == n - 1 or boxes[i][0] != boxes[i + 1][0]:
            need -= 1
    return dp[-1]
    