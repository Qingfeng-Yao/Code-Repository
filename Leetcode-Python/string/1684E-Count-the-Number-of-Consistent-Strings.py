'''
Problem Discription:
You are given a string allowed consisting of distinct characters and an array of strings words. A string is consistent if all characters in the string appear in the string allowed.
Return the number of consistent strings in the array words.

Example:
Input: allowed = "ab", words = ["ad","bd","aaab","baa","badab"]
Output: 2
'''

## self solution
### 集合运算
def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
    count = 0
    t_l = set(allowed)
    for w in words:
        if (t_l | set(w)) == t_l:
            count += 1
    return count

## other solution
### 列表生成式+all
def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
    return sum(all(c in allowed for c in w) for w in words)