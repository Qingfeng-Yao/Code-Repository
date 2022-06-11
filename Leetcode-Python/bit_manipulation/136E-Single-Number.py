'''
Problem Discription:
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
Follow up: Could you implement a solution with a linear runtime complexity and without using extra memory?

Example:
Input: nums = [2,2,1]
Output: 1
'''

# 使用^运算符，相同位为0，不同位为1
def singleNumber(self, nums: List[int]) -> int:
    result = 0
    for i in nums:
        result ^= i
        
    return result