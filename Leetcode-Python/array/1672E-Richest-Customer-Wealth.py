'''
Problem Discription:
You are given an m x n integer grid accounts where accounts[i][j] is the amount of money the i​​​​​​​​​​​th​​​​ customer has in the j​​​​​​​​​​​th​​​​ bank. Return the wealth that the richest customer has.
A customer's wealth is the amount of money they have in all their bank accounts. The richest customer is the customer that has the maximum wealth.

Example:
Input: accounts = [[1,2,3],[3,2,1]]
Output: 6
'''

## self solution
### 一层循环，使用已有数组空间
def maximumWealth(self, accounts: List[List[int]]) -> int:
    for c in accounts:
        if sum(c) > accounts[0][0]:
            accounts[0][0] = sum(c)
            
    return accounts[0][0]