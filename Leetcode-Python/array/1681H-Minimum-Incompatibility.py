'''
Problem Discription:
You are given an integer array nums​​​ and an integer k. You are asked to distribute this array into k subsets of equal size such that there are no two equal elements in the same subset.
A subset's incompatibility is the difference between the maximum and minimum elements in that array.
Return the minimum possible sum of incompatibilities of the k subsets after distributing the array optimally, or return -1 if it is not possible.
A subset is a group integers that appear in the array with no particular order.

Example:
Input: nums = [1,2,1,4], k = 2
Output: 4
'''

## other solution 1: Time Limit Exceeded
### sort+dp+bit
def minimumIncompatibility(self, nums: List[int], k: int) -> int:
    n = len(nums)
    if k == n: return 0
    dp = [[float("inf")] * n for _ in range(1<<n)] 
    nums.sort()
    for i in range(n): dp[1<<i][i] = 0 

    for mask in range(1<<n):
        n_z_bits = [j for j in range(n) if mask&(1<<j)] # 记录mask中1的位置情况
        if len(n_z_bits)%(n//k) == 1: # start new group: choose any index we want, not neccecerily bigger than previous
            for j, l in permutations(n_z_bits, 2):
                dp[mask][l] = min(dp[mask][l], dp[mask^(1<<l)][j])
        else: # continue to build our group: next index should be bigger than previous
            for j, l in combinations(n_z_bits, 2):
                if nums[j] != nums[l]:
                    dp[mask][j] = min(dp[mask][j], dp[mask^(1<<j)][l] + nums[l] - nums[j])
                    
    return min(dp[-1]) if min(dp[-1]) != float("inf") else -1

## other solution 2
### backtracking: 这是一个非常耗时到算法，通常需要early break condition
### sort+函数递归/停止条件
def minimumIncompatibility(self, nums: List[int], k: int) -> int:
    nums.sort(reverse = True) # 从大到小
    upperbound = len(nums) // k
    arr = [[] for _ in range(k)]
    self.res = float('inf')
    def assign(i):
        if i == len(nums):
            self.res = min(self.res, sum(arr[i][0]-arr[i][-1] for i in range(k)))
            return True
        flag = 0
        for j in range(k):
            if not arr[j] or (len(arr[j]) < upperbound and arr[j][-1] != nums[i]):
                arr[j].append(nums[i])
                if assign(i+1):
                    flag += 1
                arr[j].pop()
            if flag >= 2: break
        return flag != 0
    if max(collections.Counter(nums).values()) > k: return -1
    assign(0)
    return self.res