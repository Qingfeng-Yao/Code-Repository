'''
Problem Discription:
Given an integer n, return the decimal value of the binary string formed by concatenating the binary representations of 1 to n in order, modulo 10^9 + 7.

Example:
Input: n = 3
Output: 27
'''

### bin函数+int函数
def concatenatedBinary(self, n: int) -> int:
    c = ""
    for i in range(1, n+1):
        c += bin(i)[2:]
        
    return int(c, 2)%(10**9+7)