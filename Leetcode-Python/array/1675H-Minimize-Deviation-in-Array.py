'''
Problem Discription:
You are given an array nums of n positive integers.
You can perform two types of operations on any element of the array any number of times:
If the element is even, divide it by 2.
For example, if the array is [1,2,3,4], then you can do this operation on the last element, and the array will be [1,2,3,2].
If the element is odd, multiply it by 2.
For example, if the array is [1,2,3,4], then you can do this operation on the first element, and the array will be [2,2,3,4].
The deviation of the array is the maximum difference between any two elements in the array.
Return the minimum deviation the array can have after performing some number of operations.

Example:
Input: nums = [1,2,3,4]
Output: 1
'''

## otheer solution
### 动态最值问题/堆
def minimumDeviation(self, nums: List[int]) -> int:
    h = []
    n = len(nums)
    curr_min = float('inf')
    for n in nums:
        if n % 2 == 1:
            n *= 2  # odd value can only be multiplied by 2 once. We do it now.
        curr_min = min(curr_min, n) # keep track of the minimum value
        heappush(h, -n)
    devi = -h[0] - curr_min 
    while h[0] % 2 == 0: # the deviation will stop changing when the maximum value is odd which can not be devided by 2 any more.
        e = -heappop(h)
        while e % 2 == 0 and e >= -h[0]: # we keep track of the deviation while reducing the maximum number.
            devi = min(devi, e - curr_min)
            e = e // 2
        curr_min = min(curr_min, e)
        heappush(h, -e)
        devi = min(devi, -h[0] - curr_min)
    return devi
        
    