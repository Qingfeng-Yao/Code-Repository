'''
Problem Discription:
Given a string s consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return 0.
A word is a maximal substring consisting of non-space characters only.

Example:
Input: s = "Hello World"
Output: 5
Input: s = " "
Output: 0
'''

# 先去掉末尾的空格
def lengthOfLastWord(self, s: str) -> int:
    return len(s.strip().split()[-1]) if s.strip().split() else 0