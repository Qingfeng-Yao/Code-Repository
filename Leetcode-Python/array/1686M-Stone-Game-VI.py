'''
Problem Discription:
Alice and Bob take turns playing a game, with Alice starting first.
There are n stones in a pile. On each player's turn, they can remove a stone from the pile and receive points based on the stone's value. Alice and Bob may value the stones differently.
You are given two integer arrays of length n, aliceValues and bobValues. Each aliceValues[i] and bobValues[i] represents how Alice and Bob, respectively, value the ith stone.
The winner is the person with the most points after all the stones are chosen. If both players have the same amount of points, the game results in a draw. Both players will play optimally. Both players know the other's values.
Determine the result of the game, and:
If Alice wins, return 1.
If Bob wins, return -1.
If the game results in a draw, return 0.

Example:
Input: aliceValues = [1,3], bobValues = [2,1]
Output: 1
'''

### 假设+分析题意(Alice wants the largest difference between her and Bob, then a1 - b2 >= a2 - b1, i.e., a1 + b1 >= a2 + b2)
def stoneGameVI(self, aliceValues: List[int], bobValues: List[int]) -> int:
    def cmp(a,b):
        if a>b:
            return 1
        elif a<b:
            return -1
        else:
            return 0
    A = sorted(zip(aliceValues, bobValues), key=sum)
    return cmp(sum(a for a, b in A[::-2]), sum(b for a, b in A[-2::-2]))