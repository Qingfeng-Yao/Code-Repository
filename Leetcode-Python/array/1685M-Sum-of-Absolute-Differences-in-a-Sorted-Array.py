'''
Problem Discription:
You are given an integer array nums sorted in non-decreasing order.
Build and return an integer array result with the same length as nums such that result[i] is equal to the summation of absolute differences between nums[i] and all the other elements in the array.
In other words, result[i] is equal to sum(|nums[i]-nums[j]|) where 0 <= j < nums.length and j != i (0-indexed).

Example:
Input: nums = [2,3,5]
Output: [4,3,5]
'''

### sort+遍历
def getSumAbsoluteDifference(nums):
    n = len(nums)
    sumBelow, sumAbove = 0, sum(nums)
    result = []
    
    for i, num in enumerate(nums):
        result.append(sumAbove - (n - i) * num + i * num - sumBelow)
        sumAbove -= num
        sumBelow += num

    return result