## Dynamic Sets
- 元素具有key和satellite data
- 支持查询(如以key值搜索、求最值、求Successor或Predecessor)，支持更改操作(如插入和删除)

### Binary Search Trees/BSTs
- dynamic sets中一种重要的数据结构
- 除了satellite data，元素具有key、left、right、p
- 属性：`key[x]`位于`[key[left(x)], key[right(x)]]`之间
  - inorder tree walk：升序打印元素
- BSTs上的操作：
  - 搜索、插入的运行时间都是O(h)，h是树的高度；最坏的情况是O(n)，即树是左孩子或右孩子的线性串
  - 利用BSTs进行排序
    - 平均情况：快排序形式，与快排序相同的partitions，但顺序不一样。与BSTs排序相比，快排序有更好的常数、可以sort in place、不需要建立数据结构
