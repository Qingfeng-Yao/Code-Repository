'''
Problem Discription:
A decimal number is called deci-binary if each of its digits is either 0 or 1 without any leading zeros. For example, 101 and 1100 are deci-binary, while 112 and 3001 are not.
Given a string n that represents a positive decimal integer, return the minimum number of positive deci-binary numbers needed so that they sum up to n.

Example:
Input: n = "32"
Output: 3
Explanation: 10 + 11 + 11 = 32
'''

## self solution
### 分析题意+列表生成式
def minPartitions(self, n: str) -> int:
    return max([int(i) for i in list(n)])

## other solution
### 分析题意+不使用列表生成式
def minPartitions(self, n: str) -> int:
    return int(max(n))