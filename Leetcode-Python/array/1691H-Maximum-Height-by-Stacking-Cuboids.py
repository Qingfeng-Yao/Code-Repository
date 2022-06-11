'''
Problem Discription:
Given n cuboids where the dimensions of the ith cuboid is cuboids[i] = [widthi, lengthi, heighti] (0-indexed). Choose a subset of cuboids and place them on each other.
You can place cuboid i on cuboid j if widthi <= widthj and lengthi <= lengthj and heighti <= heightj. You can rearrange any cuboid's dimensions by rotating it to put it on another cuboid.
Return the maximum height of the stacked cuboids.

Example:
Input: cuboids = [[50,45,20],[95,37,53],[45,23,12]]
Output: 190
Explanation:
Cuboid 1 is placed on the bottom with the 53x37 side facing down with height 95.
Cuboid 0 is placed next with the 45x20 side facing down with height 50.
Cuboid 2 is placed next with the 23x12 side facing down with height 45.
The total height is 95 + 50 + 45 = 190.
'''

## other solution
### dp+排序+遍历比较
def maxHeight(self, cuboids: List[List[int]]) -> int:
    cuboids = [[0, 0, 0]] + sorted(map(sorted, cuboids))
    dp = [0] * len(cuboids)
    for j in range(1, len(cuboids)):
        for i in range(j):
            if all(cuboids[i][k] <= cuboids[j][k] for k in range(3)):
                dp[j] = max(dp[j], dp[i] + cuboids[j][2])
    return max(dp)