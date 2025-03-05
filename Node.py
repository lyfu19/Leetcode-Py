from typing import Optional, List


class Node:
    def __init__(self, val: Optional[int] = None, children: Optional[list['Node']] = None):
        self.val = val
        self.children = children or []
    
    @staticmethod
    def create_tree(data: List[Optional[int]]) -> Optional['Node']:
        if not data:
            return None

        root = Node(data[0])  # 创建根节点
        queue = [root]  # 队列存放父节点
        idx = 2  # 从第一个孩子的层次开始

        while queue and idx < len(data):
            parent = queue.pop(0)  # 获取当前父节点

            # 处理当前父节点的子节点，直到遇到 None
            while idx < len(data) and data[idx] is not None:
                child = Node(data[idx])  # 创建子节点
                parent.children.append(child)  # 将子节点添加到父节点
                queue.append(child)  # 将子节点加入队列，作为下一层的父节点
                idx += 1

            idx += 1  # 跳过 None，表示当前父节点的子节点结束

        return root