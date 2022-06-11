'''
Problem Discription:
Given an integer n, return true if it is a power of two. Otherwise, return false.
An integer n is a power of two, if there exists an integer x such that n == 2^x.

Example:
Input: n = 1
Output: true
Explanation: 2^0 = 1
'''

# 2的幂的二进制数，最高位为1，其他位为0
## 对于N为2的幂的数，都有 N&(N-1)=0 
def isPowerOfTwo(self, n: int) -> bool:
    return n>0 and n&(n-1)==0