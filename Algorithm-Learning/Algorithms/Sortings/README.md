## 排序
- 排序大的分类可以分为两种：内排序和外排序
- 放在内存的称为内排序，需要使用外存的称为外排序；以下涉及的均为内排序
- 交换排序包括冒泡排序和快排序
- 插入排序这里指直接插入排序
- 选择排序包括直接选择排序和堆排序

## 冒泡排序
* 两两比较，若大值沉底，则从后到前逐渐有序

## 插入排序
* 从数组第2位开始，确定插入位置，逐渐使得数组前部有序
* fast on small inputs (less than 50 elements) and fast on nearly-sorted inputs

## 选择排序
* 从数组第1位开始，筛选最值，逐渐使得数组前部有序

## 归并排序
* 分而治之`divide-and-conquer`: 将问题分`divide`成一些小问题然后递归求解，而治`conquer`的阶段则将分的阶段得到的各答案"修补"在一起
  * split array in half
  * recursively sort subarrays
  * linear-time merge step
* doesn’t sort in place(需要分配O(n) space)

### 解决递归的方法
* Substitution method
  * 猜测答案的形式，并用归纳法求出常数以及证明答案的准确性
* Iteration method
  * 扩展递归表达式，得到一个求和形式，然后求这个和
* Master method
  * `f(n)`为每个阶段(划分问题+组合已解决的子问题)的cost
  * 根据`f(n)`、`a`、`b`的值确定算法的时间复杂度

## 堆排序
*  使用堆数据结构
  * (nearly)完全二叉树
  * 数组存储结点值(层序)：各节点的索引与其父结点、孩子结点的索引有关
  * heap属性：父节点的值大于等于孩子节点的值(根节点的值最大)
  * 树中某个结点的高度=从该结点出发到叶子结点的最长路上的边数(根结点的高度=树的高度)
    * 含有n个元素的堆的高度为`O(lgn)`
* 堆操作
  * 基本堆操作所花最多的时间正比于堆的高度`O(h)`
  * `Heapify(A,i)`：保持堆属性，让违反堆属性的结点值下沉
  * `BuildHeap(A)`：从n/2开始直到1，对每个结点调用`Heapify()`
    * 给定一个未排序的数组A，使得A成为一个堆
* 堆排序：借助`BuildHeap(A)`，此时数组首位是最大值，然后首尾数值交换，同时缩短数组长度，此时数组末尾总是存储着准确的值
  * 结合插入排序和归并排序的优点
  * sort in place
* Priority Queues(堆数据结构): 维护一个元素集合S，其中每一个元素与一个`value`或`key`相关
  * 涉及操作：
    * Insert(S, x): 向集合S插入元素x
    * Maximum(S): 返回集合S中有最大key的元素
    * ExtractMax(S): 移除并返回集合S中有最大key的元素

## 快速排序
### 排序过程`divide-and-conquer`
* 确定数组的分割位置`partition`
  * 每次都取数组的最后一个元素作为比较标准`pivot`，凡是小于等于该元素的都放在它的左边，大于它的放在右边
  * 上一步中的元素`pivot`位置即为要找的分割位置
* 递归进行算法

### 时间复杂度
* 最优情况(划分是平衡的)：每一次取到的元素都刚好平分整个数组 O(nlgn)
* 最差情况(划分是不平衡的)：每一次取到的元素就是数组中最小/最大的，这种情况其实就是冒泡排序了(每一次都排好一个元素的顺序) O(n^2)
  * 几乎排序整齐的数组会引发最差情况
  * 用randomized quicksort来解决(两种方法：一是打乱输入数组，二是随机选取pivot值)
* 平均情况(随机输入)：O(nlgn)
* sort in place

## comparison sorts
* 上述所有排序算法都是comparison sorts，即获得一个序列中的顺序信息的唯一操作是两两比较
* 所有comparison sorts是Ω(nlgn): 
  * 一次comparison sort必须做O(n)次comparisons
* 用决策树来抽象comparison sorts
  * 排序n个元素的任何决策树有高度Ω(nlgn)
  
## sorting in linear time
### counting sort
* 没有元素间的比较，但依赖于被排序的数值范围(需要存储该数值范围，方便对该范围中的数字进行计数)
* not sorting in place
* 不适用于k值太大的情况

### radix sort
* 基于数据位数的一种排序算法
  * 从低位（个位）向高位排/LSD–Least Significant Digit first
  * 从高位向低位（个位）排/MSD–Most Significant Digit first
* not sorting in place
* 不适用于浮点数

### bucket sort
* 假设n个输入是在`[0,1)`之间
* 创建n个链表/桶，每个长度是1/n
* 如果输入是均匀分布，则桶大小为`O(1)`，总时间是`O(n)`

## order statistics
* 在有n个元素的集合中，第i个order statistic是第i个最小值(最小值就是第1个order statistic)
  * `O(3n/2)`：找到最大值和最小值(每两个元素之间需要三次比较，两元素之间比较、compare the largest to maximum, smallest to minimum)
* 找order statistics：选择问题
  * 随机选择：`O(n)`
    * 使用快排序中的partition() 
    * 当pivot均匀选取时，快选择算法的比较期望数最大不超过4n


