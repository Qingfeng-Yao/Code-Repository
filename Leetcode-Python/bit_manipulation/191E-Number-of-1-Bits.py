'''
Problem Discription:
Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).
Note:
Note that in some languages such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3 above, the input represents the signed integer. -3.

Example:
Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
'''

# 对于任意一个数，将 n 和 n-1 进行 & 运算，我们都可以把 n 中最低位的 1 变成 0
def hammingWeight(self, n: int) -> int:
    count = 0
    while n>0:
        n &= n-1
        count += 1
    return count