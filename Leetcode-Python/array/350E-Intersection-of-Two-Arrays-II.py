'''
Problem Discription:
Given two arrays, write a function to compute their intersection.

Example:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
'''

# 排序+两个数组遍历
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    i,j,k = 0, 0, 0
    nums1.sort()
    nums2.sort()
    
    while i<len(nums1) and j<len(nums2):
        if nums1[i]>nums2[j]:
            j+=1
        elif nums1[i]<nums2[j]:
            i+=1
        else:
            nums1[k]=nums1[i]
            i+=1
            j+=1
            k+=1
            
    return nums1[:k]
