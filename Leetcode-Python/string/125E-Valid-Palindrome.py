'''
Problem Discription:
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
Note: For the purpose of this problem, we define empty string as valid palindrome.

Example:
Input: "A man, a plan, a canal: Panama"
Output: true
'''

# filter+str.isalnum+[::-1] 
def isPalindrome(self, s: str) -> bool:
    s = list(filter(str.isalnum, s.lower()))
    return s == s[::-1] 