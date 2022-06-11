'''
Problem Discription:
You are given a 0-indexed integer array nums and an integer k.
You are initially standing at index 0. In one move, you can jump at most k steps forward without going outside the boundaries of the array. That is, you can jump from index i to any index in the range [i + 1, min(n - 1, i + k)] inclusive.
You want to reach the last index of the array (index n - 1). Your score is the sum of all nums[j] for each index j you visited in the array.
Return the maximum score you can get.

Example:
Input: nums = [1,-1,-2,4,-7,3], k = 2
Output: 7
Explanation: You can choose your jumps forming the subsequence [1,-1,4,3] (underlined above). The sum is 7.
'''

## other solution
### dp(dp[i] represents the maxScore if you are at that idx-->dp[i] = nums[i] + max(dp[i-k], dp[i-k+1].............dp[i-1]))  +  deque(hold the maxScore of the k DPs before idx i and each deque element will be a tuple (score, idx))(sliding window maximum variation)
def maxResult(self, nums: List[int], k: int) -> int:
    dp = [0] * len(nums)
    dp[0] = nums[0]
    d = deque([(nums[0],0)])
    for i in range(1, len(nums)):
        dp[i] = nums[i] + d[0][0]

        while d and d[-1][0] < dp[i]:  
            d.pop()                 
        d.append((dp[i],i))            

        if i-k == d[0][1]:              
            d.popleft()              

    return dp[-1]
    