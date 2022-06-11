## asymptotic performance
- 与问题大小`n`有关
	- 运行时间
	- 内存使用
	- Bandwidth/power requirements/logic gates/etc
- 关心`order of growth`，即关心高阶项
	- 当输入大小变大时，是高阶项占主导
- asymptotic notation
	- 上界`big O`
	- 下界`big omega`
	- asymptotic tight bound: `big theta`

### 输入大小
- Sorting: number of input items
- Multiplication: total number of bits
- Graph algorithms: number of nodes & edges
- Etc

### 时间复杂度
- 基本步骤的数目
- 最好情况
- 最坏情况：运行时间的上界
- 平均情况：期望的运行时间(对于排序，则每种序列出现的概率相等)
```
以下对应排序算法最好情况、最坏情况及平均情况的时间复杂度
冒泡排序: O(n)、O(n^2)、O(n^2)
插入排序: O(n)、O(n^2)、O(n^2)
选择排序: O(n)、O(n^2)、O(n^2)
归并排序: O(nlgn)、O(nlgn)、O(nlgn)
堆排序: O(nlgn)、O(nlgn)、O(nlgn)
快排序: O(nlgn)、O(n^2)、O(nlgn)
计数排序: O(n+k)、O(n+k)、O(n+k)；其中k指明数值取值范围
基数排序：O(n*m)、O(n*m)、O(n*m)；其中m是数据位数
```

### 空间复杂度
```
以下对应排序算法的空间复杂度
冒泡排序: O(1) 
插入排序: O(1)
选择排序: O(1)
归并排序: O(n)
堆排序: O(1)
快排序: O(nlgn)
计数排序: O(n+k)
基数排序：O(m)
```

### 稳定性分析
如果待排序的序列中存在两个或两个以上具有相同关键词的数据，排序后这些数据的相对次序保持不变，即它们的位置保持不变，则该算法是稳定的；如果排序后，数据的相对次序发生了变化，则该算法是不稳定的。
```
以下对应排序算法的稳定性
冒泡排序: 稳定
插入排序: 稳定
选择排序: 不稳定
归并排序: 稳定
堆排序: 不稳定
快排序: 不稳定
计数排序: 稳定
基数排序：稳定
```
