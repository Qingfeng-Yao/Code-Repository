'''
Problem Discription:
You are given an array of positive integers nums and want to erase a subarray containing unique elements. The score you get by erasing the subarray is equal to the sum of its elements.
Return the maximum score you can get by erasing exactly one subarray.
An array b is called to be a subarray of a if it forms a contiguous subsequence of a, that is, if it is equal to a[l],a[l+1],...,a[r] for some (l,r).

Example:
Input: nums = [4,2,4,5,6]
Output: 17
Explanation: The optimal subarray here is [2,4,5,6].
'''

## other solution
### sliding window(确定两个边界)+遍历
def maximumUniqueSubarray(self, nums: List[int]) -> int:
    ans = float('-inf')
    cur = 0
    # sliding window; current value = [i, j]
    seen = set()
    i = 0
    for j in range(len(nums)):
        while nums[j] in seen:
            cur -= nums[i]
            seen.remove(nums[i])
            i += 1
        seen.add(nums[j])
        cur += nums[j]
        ans = max(ans, cur)
        
    return ans