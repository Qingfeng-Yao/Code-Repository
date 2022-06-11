'''
Problem Discription:
Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it.

Example:
Input: nums = [2,2,3,2]
Output: 3
'''

# Assume "ones" and "twos" to be sets that are keeping track of which numbers have appeared once and twice respectively
# consider the (set^val) as one of the following:
## 1. adding "val" to the "set" if "val" is not in the "set" => A^0 = A
## 2. removing "val" from the "set" if "val" is already in the "set" => A^A = 0

def singleNumber(self, nums: List[int]) -> int:
    ones, twos = 0, 0
    for n in nums:
        ones = (ones ^ n) & ~twos
        twos = (twos ^ n) & ~ones
        
    return ones
        