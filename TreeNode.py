from collections import deque
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left:Optional["TreeNode"]=None, right:Optional["TreeNode"]=None):
        self.val = val
        self.left = left
        self.right = right
    
    @staticmethod
    def from_list(values: list[Optional[int]]) -> Optional["TreeNode"]:
        if not values:
            return None

        # 创建根节点
        root = TreeNode(values[0])
        queue = [root]  # 使用队列来辅助层序构建树
        index = 1  # 从第二个元素开始（根节点已处理）

        while index < len(values):
            node = queue.pop(0)  # 取出队列中的第一个节点

            # 为左子节点赋值
            if values[index] is not None:
                node.left = TreeNode(values[index])
                queue.append(node.left)
            index += 1

            # 为右子节点赋值
            if index < len(values) and values[index] is not None:
                node.right = TreeNode(values[index])
                queue.append(node.right)
            index += 1

        return root
    
    @staticmethod
    def to_list(root: Optional["TreeNode"]) -> list[Optional[int]]:
        result = []
        queue = [root]

        while queue:
            node = queue.pop(0)
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)

        # 去除末尾的 None，保持简洁
        while result and result[-1] is None:
            result.pop()

        return result