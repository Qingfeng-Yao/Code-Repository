'''
Problem Discription:
Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.
An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.
We define that a subsequence a is more competitive than a subsequence b (of the same length) if in the first position where a and b differ, subsequence a has a number less than the corresponding number in b. For example, [1,3,4] is more competitive than [1,3,5] because the first position they differ is at the final number, and 4 is less than 5.

Example:
Input: nums = [3,5,2,6], k = 2
Output: [2,6]
'''

## other solution
### 遍历+pop/while
def mostCompetitive(self, nums: List[int], k: int) -> List[int]:
    n = len(nums)
    s = []
    for i, x in enumerate(nums):
        while s and s[-1] > x and len(s) + n - i > k:
            s.pop()
        if len(s) < k:
            s.append(x)
    return s