'''
Problem Discription:
We are given two strings, A and B.
A shift on A consists of taking string A and moving the leftmost character to the rightmost position. For example, if A = 'abcde', then it will be 'bcdea' after one shift on A. Return True if and only if A can become B after some number of shifts on A.

Example:
Input: A = 'abcde', B = 'cdeab'
Output: true
'''

# 转换成子串包含问题
def rotateString(self, A: str, B: str) -> bool:
    return len(A)==len(B) and ((A+A).find(B)+1) # B in A + A