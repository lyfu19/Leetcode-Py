from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next:Optional[ListNode] = None

    @staticmethod
    def createListNode(nums, pos):
        """
        根据数组和 pos 创建 ListNode。
        
        参数:
        - nums: 输入数组，表示链表节点的值。
        - pos: 指定尾节点指向的位置索引，-1 表示没有环。
        
        返回:
        - 链表的头节点。
        """
        if not nums:
            return None
        
        # 创建链表节点
        head = ListNode(nums[0])
        current = head
        nodes = [head]  # 用于存储所有节点以便处理环
        
        for num in nums[1:]:
            new_node = ListNode(num)
            current.next = new_node
            current = new_node
            nodes.append(new_node)
        
        # 处理 pos 指定的尾节点连接
        if pos != -1:
            current.next = nodes[pos]  # 将尾节点连接到 pos 指定的节点
        
        return head
    
    @staticmethod
    def createLinkedLists(intersectVal, listA, listB, skipA, skipB):
        """
        创建两个链表并在指定位置相交。

        参数:
        - intersectVal: 相交节点的值。如果为 0，则两个链表不相交。
        - listA: 链表 A 的值列表。
        - listB: 链表 B 的值列表。
        - skipA: 链表 A 中从头开始到相交节点的索引。
        - skipB: 链表 B 中从头开始到相交节点的索引。

        返回:
        - headA: 链表 A 的头节点。
        - headB: 链表 B 的头节点。
        """
        # 创建链表 A
        headA = ListNode(listA[0])
        currentA = headA
        nodesA = [headA]
        for val in listA[1:]:
            new_node = ListNode(val)
            currentA.next = new_node
            currentA = new_node
            nodesA.append(new_node)
        
        # 创建链表 B
        headB = ListNode(listB[0])
        currentB = headB
        nodesB = [headB]
        for val in listB[1:]:
            new_node = ListNode(val)
            currentB.next = new_node
            currentB = new_node
            nodesB.append(new_node)

        # 如果有相交点
        if intersectVal != 0:
            # 获取交点
            intersectNode = nodesA[skipA]  # 找到链表 A 中的交点节点
            currentB = headB
            for _ in range(skipB):
                currentB = currentB.next
            currentB.next = intersectNode  # 让链表 B 的尾部连接到交点

        return headA, headB
    
    @staticmethod
    def printList(node: "ListNode") -> None:
        """
        打印链表中所有节点的值
        :param node: 链表的头节点
        """
        current = node
        while current:
            print(current.val, end=" -> ")
            current = current.next
        print("None")  # 表示链表结束