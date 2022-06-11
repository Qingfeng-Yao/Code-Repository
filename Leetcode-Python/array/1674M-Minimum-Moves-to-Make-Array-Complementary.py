'''
Problem Discription:
You are given an integer array nums of even length n and an integer limit. In one move, you can replace any integer from nums with another integer between 1 and limit, inclusive.
The array nums is complementary if for all indices i (0-indexed), nums[i] + nums[n - 1 - i] equals the same number. For example, the array [1,2,3,4] is complementary because for all indices i, nums[i] + nums[n - 1 - i] = 5.
Return the minimum number of moves required to make nums complementary.

Example:
Input: nums = [1,2,4,3], limit = 4
Output: 1
'''

### 二分遍历+Counter计数/分情况讨论
import collections
from typing import List
import math
class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        delta = collections.Counter() # delta[i] = x means we need x more operations when target T change from i - 1 to i
        n = len(nums)
        for i in range(n // 2):
            a, b = nums[i], nums[n - 1 - i]
            delta[2] += 2
            delta[min(a, b) + 1] -= 1
            delta[a + b] -= 1
            delta[a + b + 1] += 1
            delta[max(a, b) + limit + 1] += 1
            
        curr = 0            
        res = math.inf
        for i in range(2, 2 * limit + 1):
            curr += delta[i]
            res = min(res, curr)
        return res   

nums = [1,2,4,3]
limit = 4
s = Solution()
print(s.minMoves(nums,limit))