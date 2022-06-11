'''
Problem Discription:
Given a string, find the first non-repeating character in it and return its index. If it doesn't exist, return -1.

Example:
Input: "leetcode"
Output: 0
'''

# 两次遍历，第一次记录字符索引，重复出现的字符索引覆盖；第二次比较索引，索引一致则返回该索引，不一致需要记录为-1
def firstUniqChar(self, s: str) -> int:
    d = {}
    for i,c in enumerate(s):
        d[c]=i
    print(d)
    for i,c in enumerate(s):
        if i == d[c]:
            return i
        else:
            d[c]=-1
    return -1
