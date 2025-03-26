from ast import If, Set
from collections import deque, Counter
from functools import cmp_to_key
import heapq
from itertools import combinations
import re
import string
from typing import Optional, List
from unittest import result
from ListNode import ListNode
from TreeNode import TreeNode
import math
import numpy as np
from Node import Node
from datetime import datetime

class MyStack:

    def __init__(self):
        self.queueMain = deque()
        self.queueAuxiliary = deque()

    def push(self, x: int) -> None:
        self.queueMain.append(x)

    def pop(self) -> int:
        while len(self.queueMain) > 1:
            self.queueAuxiliary.append(self.queueMain.popleft())
        last = self.queueMain.popleft()
        temp = self.queueMain
        self.queueMain = self.queueAuxiliary
        self.queueAuxiliary = temp
        return last

    def top(self) -> int:
        last = self.pop()
        self.queueMain.append(last)
        return last

    def empty(self) -> bool:
        return not self.queueMain

class MyQueue:

    def __init__(self):
        self.listMain = []
        self.listAuxiliary = []

    def push(self, x: int) -> None:
        while self.listMain:
            self.listAuxiliary.append(self.listMain.pop())
        self.listMain.append(x)
        while self.listAuxiliary:
            self.listMain.append(self.listAuxiliary.pop())

    def pop(self) -> int:
        return self.listMain.pop()

    def peek(self) -> int:
        last = self.listMain.pop()
        self.listMain.append(last)
        return last

    def empty(self) -> bool:
        return not self.listMain

class NumArray:

    def __init__(self, nums: list[int]):
        self.nums = nums
        for i in range(len(nums)-1):
            self.nums[i+1] += self.nums[i]

    def sumRange(self, left: int, right: int) -> int:
        if left == 0:
            return self.nums[right]
        else:
            return self.nums[right] - self.nums[left-1]

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = []
        for num in nums:
            self.add(num)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
    
class MyHashSet:

    def __init__(self):
        self.size = 1000
        self.buckets: List[List] = [[] for _ in range(self.size)]

    def _hash(self, key: int) -> int:
        return key % self.size

    def add(self, key: int) -> None:
        i = self._hash(key)
        if key not in self.buckets[i]:
            self.buckets[i].append(key)

    def remove(self, key: int) -> None:
        i = self._hash(key)
        if key in self.buckets[i]:
            self.buckets[i].remove(key)

    def contains(self, key: int) -> bool:
        i = self._hash(key)
        return key in self.buckets[i]
    
class MyHashMap:

    def __init__(self):
        self.size = 1000
        self.buckets: List[List[List]] = [[] for _ in range(self.size)]

    def _hash(self, key: int) -> int:
        return key % self.size

    def put(self, key: int, value: int) -> None:
        hashKey = self._hash(key)
        for sub in self.buckets[hashKey]:
            if sub[0] == key:
                sub[-1] = value
                return
        self.buckets[hashKey].append([key, value])

    def get(self, key: int) -> int:
        hashKey = self._hash(key)
        for sub in self.buckets[hashKey]:
            if sub[0] == key:
                return sub[-1]
        return -1

    def remove(self, key: int) -> None:
        hashKey = self._hash(key)
        for sub in self.buckets[hashKey]:
            if sub[0] == key:
                self.buckets[hashKey].remove(sub)
                return

class Solution:
    # def inorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
    #     if not root:
    #         return []
        
    #     return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
        
    
    def inorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
        ans = []
        current = root
        stack: list[TreeNode] = []

        while current or stack:
            while current:
                stack.append(current)
                current = current.left

            current = stack.pop()
            ans.append(current.val)
            current = current.right

        return ans
    
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack = [(p, q)]

        while stack:
            item1, item2 = stack.pop()

            if not item1 and not item2:
                continue

            if not item1 or not item2 or item1.val != item2.val:
                return False
            
            stack.append((item1.left, item2.left))
            stack.append((item1.right, item2.right))

        return True
    
    def isMirror(self, left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        if not left and not right:
            return True
        if not left or not right or left.val != right.val:
            return False
        
        return self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return False
        
        # return self.isMirror(root.left, root.right)
        stack = [(root.left, root.right)]
        while stack:
            left, right = stack.pop()
            if not left and not right:
                continue
            if not left or not right or left.val != right.val:
                return False
            
            stack.append((left.left, right.right))
            stack.append((left.right, right.left))
        
        return True
            
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        
        # depth = max(self.maxDepth(root.left), self.maxDepth(root.right))
        # return depth + 1

        depth = 1
        stack = [root]
        while stack:
            temp = []
            for item in stack:
                if item.left:
                    temp.append(item.left)
                if item.right:
                    temp.append(item.right)
            if temp:
                depth += 1
            stack = temp
        return depth
    
    def sortedArrayToBST(self, nums: list[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        
        mid = len(nums) // 2
        root = TreeNode(nums[mid])

        left = nums[:mid]
        right = nums[mid + 1:]
        root.left = self.sortedArrayToBST(left)
        root.right = self.sortedArrayToBST(right)
        return root
    
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        
        # gap = abs(self.maxDepth(root.left) - self.maxDepth(root.right))
        # if gap > 1:
        #     return False
        
        # return self.isBalanced(root.left) and self.isBalanced(root.right)

        def height(root: Optional[TreeNode]) -> int:
            if not root:
                return 0
            leftH = height(root.left)
            rightH = height(root.right)
            if leftH == -1 or rightH == -1 or abs(leftH - rightH) > 1:
                return -1
            return max(leftH, rightH) + 1
        
        return height(root) != -1
    

    def minDepth(self, root: Optional[TreeNode]) -> int:
        # BFS
        if not root:
            return 0
        

        # result = 1
        # stack = deque([(root, result)])
        # while stack:
        #     node, depth = stack.popleft()
        #     if not node:
        #         continue

        #     if not node.left and not node.right:
        #         result =  depth
        #         break
            
            
        #     stack.append((node.left, depth+1))
        #     stack.append((node.right, depth+1))
        
        # return result

        if not root.left and not root.right:
            return 1
        
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right))+1
    

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        # DFS
        if not root.left and not root.right:
            return root.val == targetSum
        
        remain = targetSum - root.val
        hasLeftPathSum = self.hasPathSum(root.left, remain)
        hasRightPathSum = self.hasPathSum(root.right, remain)

        return hasLeftPathSum or hasRightPathSum

    def generate(self, numRows: int) -> list[list[int]]:
        if numRows < 1:
            return []
        
        ans:list[list[int]] = []
        for row in range(numRows):
            sub:list[int] = []
            for subRow in range(row+1):
                if subRow == 0 or subRow == row:
                    sub.append(1)
                else:
                    preList = ans[row-1]
                    val = preList[subRow] + preList[subRow-1]
                    sub.append(val)
            
            ans.append(sub)

        return ans 

    def getRow(self, rowIndex: int) -> list[int]:
        if rowIndex < 0:
            return []
        
        subList:list[int] = []
        for i in range(rowIndex+1):
            temp:list[int] = []
            for j in range(i+1):
                if j == 0 or j == i:
                    temp.append(1)
                else:
                    val = subList[j] + subList[j-1]
                    temp.append(val)
            subList = temp
        return subList
            
    def maxProfit(self, prices: list[int]) -> int:
        if not prices:
            return 0
        
        profit = 0
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                if prices[i] - buy > profit:
                    profit = prices[i] - buy
        return profit

    def isPalindrome(self, s: str) -> bool:
        if s.isspace():
            return True
        # filtered = ''.join([char.lower() for char in s if char.isalnum()])
        # for i in range(len(filtered) // 2):
        #     if filtered[i]!= filtered[len(filtered)-1-i]:
        #         return False
        # return True

        begin = 0
        end = len(s)-1
        while begin < end:
            if not s[begin].isalnum():
                begin += 1
                continue
            if not s[end].isalnum():
                end -= 1
                continue
            if s[begin].lower() != s[end].lower():
                return False
            begin += 1
            end -= 1
        return True
    

    def singleNumber(self, nums: list[int]) -> int:
        ans = 0
        for n in nums:
            ans ^= n
        return ans


    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        p1 = head
        p2 = head.next if head.next else None

        while p1 and p2:
            if p2.next == p1:
                return True
            p1 = p1.next
            p2 = p2.next.next if p2.next else None
        return False

    def postorderTraversalpreorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
        if not root:
            return []
        # if not root.left and not root.right:
        #     return [root.val]
        # return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

        stack: list[TreeNode] = []
        cur = root
        result = []
        while stack or cur:
            while cur:
                stack.append(cur)
                result.append(cur.val)
                cur = cur.left

            cur = stack.pop()
            cur = cur.right
        
        return result

    def postorderTraversal(self, root: Optional[TreeNode]) -> list[int]:
        # if not root:
        #     return []
        
        # if not root.left and not root.right:
        #     return [root.val]

        # return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

        def find(root: Optional[TreeNode], ans: list[int]):
            if not root:
                return
            find(root.left, ans)
            find(root.right, ans)
            ans.append(root.val)

        ans = []
        find(root, ans)
        return ans

    
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None

        a = headA
        b = headB
        while a or b:
            if a == b:
                return a
            a = a.next if a else headB
            b = b.next if b else headA
        return None

    def convertToTitle(self, columnNumber: int) -> str:
        ans = []
        mapping = {i : chr(i+65) for i in range(26)}

        while columnNumber > 0:
            columnNumber -= 1
            key = columnNumber % 26
            ans.append(mapping[key])
            columnNumber //= 26
            
        return ''.join(ans[::-1])
    
    def majorityElement(self, nums: list[int]) -> int:
        # mapping = {}
        # half = len(nums) / 2
        # for num in nums:            
        #     mapping[num] = mapping.get(num, 0) + 1
            
        #     if mapping[num] > half:
        #         return num
        # return -1

        count = 0
        majority = 0
        for n in nums:
            if count == 0:
                majority = n
            count += (1 if majority == n else -1)
        return majority

    def titleToNumber(self, columnTitle: str) -> int:
        ans = 0
        for char in columnTitle:
            ans = ans * 26 + (ord(char) - ord('A') + 1)
        return ans
    
    def reverseBits(self, n: int) -> int:
        # method 1
        # n = (n & 0xffff0000) >> 16 | (n & 0x0000ffff) << 16
        # print(format(n, '032b'))
        # n = (n & 0xff00ff00) >> 8 | (n & 0x00ff00ff) << 8 
        # print(format(n, '032b'))
        # n = (n & 0xf0f0f0f0) >> 4 | (n & 0x0f0f0f0f) << 4
        # print(format(n, '032b'))
        # n = (n & 0xcccccccc) >> 2 | (n & 0x33333333) << 2
        # print(format(n, '032b'))
        # n = (n & 0xaaaaaaaa) >> 1 | (n & 0x55555555) << 1
        # print(format(n, '032b'))
        # return n
    
        # method 2
        result = 0
        for _ in range(32):
            result = (result << 1) | (n & 1)
            n = n >> 1
            print(format(result, '032b'))
        return result
    
    def hammingWeight(self, n: int) -> int:
        times = math.floor(math.log2(n)) + 1
        ans = 0
        for _ in range(times):
            if n & 1 == 1:
                ans += 1
            n >>= 1
        return ans

    def isHappy(self, n: int) -> bool:
        if n <= 0:
            return False
        
        seen = set()

        while n != 1:
            sum = 0
            while n > 0:
                sum += ((n%10) ** 2)
                n //= 10
            n = sum
            if n in seen:
                break
            seen.add(n)

        return n == 1

        
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        temp = ListNode(-1)
        temp.next = head
        cur = temp
        next = temp.next
        
        while next:
            if next.val == val:
                cur.next = next.next
                next = next.next
                continue
            cur = next
            next = next.next
            
        return temp.next

    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        mapping = {}
        for i in range(len(s)):
            s_c = s[i]
            t_c = t[i]
            if s_c in mapping:
                if mapping[s_c] != t_c:
                    return False
            else:
                if t_c in mapping.values():
                    return False
                mapping[s_c] = t_c
        
        return True

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

    def containsDuplicate(self, nums: list[int]) -> bool:
        numSet = set()
        for n in nums:
            if n in numSet:
                return True
            numSet.add(n)
        return False

    def containsNearbyDuplicate(self, nums: list[int], k: int) -> bool:
        # mapping = {}
        # for i in range(len(nums)):
        #     value = nums[i]
        #     if value in mapping and i - mapping[value] <= k:
        #         return True
        #     mapping[value] = i
                
        # return False

        seen = set()
        for index, value in enumerate(nums):
            if index > k:
                seen.remove(nums[index - k -1])
            if value in seen:
                return True
            seen.add(value)
        return False


    def leftTreeHeight(self, root: Optional[TreeNode]) -> int:
        h = 0
        while root:
            h += 1
            root = root.left
        return h
    
    def rightTreeHeight(self, root: Optional[TreeNode]) -> int:
        h = 0
        while root:
            h += 1
            root = root.right
        return h

    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        # if not root.left and not root.right:
        #     return 1
        # return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        lH = self.leftTreeHeight(root)
        rH = self.rightTreeHeight(root)

        if lH == rH:
            return (1 << lH) - 1
        
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        if not root.left and not root.right:
            return root
        temp = root.left
        root.left = root.right
        root.right = temp
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
        
    def summaryRanges(self, nums: list[int]) -> list[str]:
        if not nums:
            return []
        ans:list[str] = []
        pre = nums[0]
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] != 1:
                if pre == nums[i-1]:
                    ans.append(str(pre))
                else:
                    ans.append(f"{pre}->{nums[i-1]}")
                pre = nums[i]

        if pre == nums[-1]:
            ans.append(str(pre))
        else:
            ans.append(f"{pre}->{nums[-1]}")
        return ans

    def isPowerOfTwo(self, n: int) -> bool:
        if n < 0:
            return False
        
        count = 0
        while n:
            count += (n & 1)
            n >>= 1
        return count == 1
    

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        nums = []
        while head:
            nums.append(head.val)
            head = head.next
        
        for i in range(len(nums)//2):
            if nums[i] != nums[len(nums) - i - 1]:
                return False
        return True

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        
        chars = set(s)
        for char in chars:
            if s.count(char) != t.count(char):
                return False
        return True

    def binaryTreePaths(self, root: Optional[TreeNode]) -> list[str]:
        # def dfs(root: Optional[TreeNode], path: str, result: list[str]):
        #     if not root:
        #         return
        #     path += str(root.val)
        #     if not root.left and not root.right:
        #         result.append(path)
        #     else:
        #         path += "->"
        #         dfs(root.left, path, result)
        #         dfs(root.right, path, result)
        # result = []
        # dfs(root, "", result)
        # return result

        if not root:
            return []
        
        que: deque[tuple[TreeNode, str]] = deque()
        que.append((root, str(root.val)))
        ans = []

        while que:
            node, path = que.popleft()
            if not node.left and not node.right:
                ans.append(path)
            if node.left:
                que.append((node.left, path+"->"+str(node.left.val)))
            if node.right:
                que.append((node.right, path+"->"+str(node.right.val)))
                
        return ans
    
    def addDigits(self, num: int) -> int:
        while num >= 10:
            num = (num % 10) + (num // 10)
        return num

    def isUgly(self, n: int) -> bool:
        if n <= 0: return False

        factors = [2,3,5]
        for f in factors:
            while n % f == 0:
                n //= f
        return n == 1
    
    def missingNumber(self, nums: list[int]) -> int:
        # nums.sort()
        # pre = -1
        # for num in nums:
        #     if num - pre > 1:
        #         break
        #     pre = num
        # return pre + 1
        length = len(nums)
        ans = int((1+length)*length / 2)
        for n in nums:
            ans -= n
        return ans
    
    

    def firstBadVersion(self, n: int) -> int:
        def isBadVersion(version: int) -> bool:
            return True if version >= 1 else False

        start = 1
        end  = n

        while end > start:
            middle = (start + end) // 2
            if isBadVersion(middle):
                end = middle
            else:
                start = middle+1
        return end
    
    def bubbleSort(self, nums: list[int]):
        length = len(nums)
        if length <= 1: return

        for i in range(length):
            flag = False
            for j in range(length-i-1):
                if nums[j] > nums[j+1]:
                    nums[j],nums[j+1] = nums[j+1],nums[j]
                    flag = True
            if not flag: break

    def insertionSort(self, nums: list[int]):
        length = len(nums)
        if length <= 1: return

        for i in range(1, length):
            value = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > value:
                nums[j+1] = nums[j]
                j -= 1
            nums[j+1] = value

    def selectionSort(self, nums: list[int]):
        length = len(nums)
        if length <= 1: return

        for i in range(length - 1):
            minIndex = i
            for j in range(i+1, length):
                if nums[j] < nums[minIndex]:
                    minIndex = j    # 修改最小值索引
            if minIndex != i:
                nums[i], nums[minIndex] = nums[minIndex], nums[i]
    
    def mergeSort(self, nums: list[int]):
        length = len(nums)
        if length <= 1: return
        
        def mergeC(nums: list[int], start: int, middle: int, end: int):
            left = nums[start: middle+1]
            right = nums[middle+1:end+1]
            i = j = 0
            k = start

            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i += 1
                else:
                    nums[k] = right[j]
                    j += 1  
                k += 1

            while i < len(left):
                nums[k] = left[i]
                i += 1
                k += 1
            while j < len(right):
                nums[k] = right[j]
                j += 1
                k += 1

        def mergeSortC(nums: list[int], start: int, end: int):
            if start >= end:
                return
            middle = (start + end) // 2
            mergeSortC(nums, start, middle)
            mergeSortC(nums, middle+1, end)
            mergeC(nums, start, middle, end)

        mergeSortC(nums, 0, length - 1)

    def quickSort(self, nums: list[int]):
        length = len(nums)
        if length <= 1: return

        def partition(nums: list[int], left: int, right: int) -> int:
            pivot = nums[right]
            i = left
            for j in range(left, right):
                if nums[j] < pivot:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
            nums[i], nums[right] = nums[right], nums[i]
            return i

        def quickSortC(nums: list[int], left: int, right: int):
            if left < right:
                middle = partition(nums, left, right)
                quickSortC(nums, left, middle-1)
                quickSortC(nums, middle+1, right)

        quickSortC(nums, 0, length-1)

    
    def moveZeroes(self, nums: list[int]) -> None:
        # gap = 0
        # for i, num in enumerate(nums):
        #     if num == 0:
        #         gap += 1
        #     elif gap > 0:
        #         nums[i - gap] = num
        #         nums[i] = 0 

        index = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                continue
            nums[index], nums[i] = nums[i], nums[index]
            index += 1
        
    def wordPattern(self, pattern: str, s: str) -> bool:
        words = s.split()
        if len(pattern) != len(words):
            return False
        mapping = {}
        mappedWords = set()

        for char, word in zip(pattern, words):
            if char not in mapping:
                if word in mappedWords:
                    return False
                mapping[char] = word
                mappedWords.add(word)
            elif mapping[char] != word:
                return False
        return True

    def canWinNim(self, n: int) -> bool:
        return n % 4 != 0

    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        
        while n % 3 == 0:
            n //= 3
        return n == 1
    
    def countBits(self, n: int) -> list[int]:
        # ans = []
        # for i in range(n+1):
        #     counts = 0
        #     while i > 0:
        #         i &= (i - 1)
        #         counts += 1
        #     ans.append(counts)
        # return ans

        ans = [0]*(n+1)
        for i in range(1, n+1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans

    def isPowerOfFour(self, n: int) -> bool:
        # if n < 0:
        #     return False
        
        # while n > 1 and n % 4 == 0:
        #     n >>= 2
        #     print(n)
        # return n == 1
        return n > 0 and (n & (n - 1)) == 0 and (n & 0b01010101010101010101010101010101) != 0
    
    def reverseString(self, s: list[str]) -> None:
        half = len(s) // 2
        for i in range(half):
            s[i], s[-i-1] = s[-i-1], s[i]

    def reverseVowels(self, s: str) -> str:
        # vowels = ('a', 'e', 'i', 'o','u', 'A', 'E', 'I', 'O','U')
        # indexs = []
        # chars = []
        # for i in range(len(s)):
        #     chars.append(s[i])
        #     if s[i] in vowels:
        #         indexs.append(i)
        # half = len(indexs) // 2
        # for i in range(half):
        #     chars[indexs[i]], chars[indexs[-i-1]] = chars[indexs[-i-1]], chars[indexs[i]]
        # return ''.join(chars)

        vowels = "aeiouAEIOU"
        start = 0
        end = len(s) - 1
        chars = list(s)
        while start < end:
            while start < end and chars[start] not in vowels:
                start += 1
            while start < end and chars[end] not in vowels:
                end -= 1

            chars[start], chars[end] = chars[end], chars[start]

            start += 1
            end -= 1
        return ''.join(chars)

    def intersection(self, nums1: list[int], nums2: list[int]) -> list[int]:
        np1 = np.array(nums1)
        np2 = np.array(nums2)
        intersection = np.intersect1d(np1, np2)
        return intersection.tolist()

    def intersect(self, nums1: list[int], nums2: list[int]) -> list[int]:
        # ans = []
        # for n in nums1:
        #     if n in nums2:
        #         ans.append(n)
        #         nums2.remove(n)
        # return ans
        c1 = Counter(nums1)
        ans = []
        for n in nums2:
            if c1.get(n, 0) > 0:
                ans.append(n)
                c1[n] -= 1
        return ans

    def isPerfectSquare(self, num: int) -> bool:
        if num <= 0:
            return False

        left = 0
        right = num
        while left <= right:
            middle = (left + right) // 2
            if middle * middle < num:
                left = middle + 1
            elif middle * middle > num:
                right = middle - 1
            else:
                return True
        return False
    
    

    def guessNumber(self, n: int) -> int:
        def guess(num: int) -> int:
            result = 6
            if num > result:
                return -1
            elif num < result:
                return 1
            else:
                return 0

        left = 1
        right = n        
        while left <= right:
            middle = (left + right) // 2
            if guess(middle) == 1:
                left = middle + 1
            elif guess(middle) == -1:
                right = middle - 1
            else:
                return middle
        return -1

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        c = Counter(magazine)
        for char in ransomNote:
            if c.get(char, 0) <= 0:
                return False
            c[char] -= 1
        return True
    
    def firstUniqChar(self, s: str) -> int:
        c = Counter(s)
        for index, char in enumerate(s):
            if c.get(char, 0) == 1:
                return index
        return -1
    
    def findTheDifference(self, s: str, t: str) -> str:
        # counter = Counter(t)
        # for char in s:
        #     if counter.get(char, 0) > 0:
        #         counter[char] -= 1
        
        # max_key = max(counter, key=counter.get)
        # return str(max_key)

        # counter = Counter(s)
        # for char in t:
        #     counter[char] -= 1
        #     if counter[char] < 0:
        #         return str(char)
        # return ""

        asc = 0
        for c in t:
            asc += ord(c)
        for c in s:
            asc -= ord(c)
        return chr(asc)

    def isSubsequence(self, s: str, t: str) -> bool:
        l_s = l_t = 0
        while l_s < len(s) and l_t < len(t):
            if s[l_s] == t[l_t]:
                l_s += 1
            l_t += 1
        return l_s == len(s)
    
    def readBinaryWatch(self, turnedOn: int) -> list[str]:
        def count(n: int) -> int:
            c = 0
            while n > 0:
                n &= (n-1)
                c += 1
            return c
        
        ans = []
        for h in range(12):
            for m in range(60):
                if (bin(h)+bin(m)).count('1') == turnedOn:
                    ans.append(f"{h}:{m:02d}")
        return ans
    
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        # non-recursive
        # stack: list[TreeNode] = []
        # sum = 0
        # while root or stack:
        #     while root:
        #         stack.append(root)
        #         root = root.left
            
        #     root = stack.pop()
        #     if root.left and not root.left.left and not root.left.right:
        #         sum += root.left.val
        #     root = root.right
        # return sum
    
        # cursive
        if not root:
            return 0
        sum = 0
        if root.left and not root.left.left and not root.left.right:
            sum = root.left.val
        return sum + self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def toHex(self, num: int) -> str:
        if num == 0:
            return "0"
        
        ans = ""
        hex = "0123456789abcdef"
        while num < 0:
            num += 2**32
        
        while num != 0:
            index = num & 15
            ans = hex[index] + ans
            num >>= 4
        return ans
    
    def longestPalindrome(self, s: str) -> int:
        counter = Counter(s)
        length = 0
        extra = False
        for v in list(counter.values()):
            if v % 2 == 0:
                length += v
            else:
                length += (v - 1)
                extra = True
        if extra:
            length += 1
        return length

    def fizzBuzz(self, n: int) -> list[str]:
        ans = []
        for i in range(1, n + 1):
            if i % 5 == 0 and i % 3 == 0:
                ans.append("FizzBuzz")
            elif  i % 5 == 0:
                ans.append("Buzz")
            elif  i % 3 == 0:
                ans.append("Fizz")
            else:
                ans.append(f"{i}")
        return ans
    
    def thirdMax(self, nums: list[int]) -> int:
        max1 = max2 = max3 = float('-inf')
        for n in nums:
            if n > max3 and n < max2:
                max3 = n
            elif n > max2 and n < max1:
                max3 = max2
                max2 = n
            elif n > max1:
                max3 = max2
                max2 = max1
                max1 = n
        if max3 != float('-inf'):
            return max3
        else:
            return max1

    def addStrings(self, num1: str, num2: str) -> str:
        # array = []
        # l1 = len(num1) - 1
        # l2 = len(num2) - 1
        # overTen = False
        # while l1 >= 0 or l2 >= 0:
        #     sum = 0
        #     if l1 >= 0:
        #         sum += int(num1[l1])
        #     if l2 >= 0:
        #         sum += int(num2[l2])
        #     if overTen:
        #         sum += 1
        #     overTen = (sum > 9)
        #     array.append(str(sum % 10))
        #     l1 -= 1
        #     l2 -= 1
        # if overTen:
        #     array.append("1")
        # return ''.join(reversed(array))
        ans = ""
        l1 = len(num1) - 1
        l2 = len(num2) - 1
        carry = 0
        while l1 >= 0 or l2 >= 0 or carry > 0:
            sum = 0
            if l1 >= 0:
                sum += int(num1[l1])
            if l2 >= 0:
                sum += int(num2[l2])
            sum += carry
            ans += str(sum % 10)
            carry = sum // 10
            l1 -= 1
            l2 -= 1
        return ans[::-1]
    
    def countSegments(self, s: str) -> int:
        segs = s.split()
        return len(segs)

    def arrangeCoins(self, n: int) -> int:
        left = 0
        right = n
        while left <= right:
            mid = (left + right) // 2
            sum = mid * (1 + mid) // 2
            if sum > n:
                right = mid - 1
            elif sum < n:
                left = mid + 1
            else:
                return mid
        return right

    def findDisappearedNumbers(self, nums: list[int]) -> list[int]:
        for n in nums:
            i = abs(n) - 1
            if nums[i] < 0:
                continue
            nums[i] = -nums[i]
        ans = []
        for i, n in enumerate(nums):
            if n > 0:
                ans.append(i + 1)
        return ans
    
    def findContentChildren(self, g: list[int], s: list[int]) -> int:
        g.sort()
        s.sort()
        gIndex = 0
        sIndex = 0
        ans = 0
        while gIndex < len(g) and sIndex < len(s):
            if s[sIndex] >= g[gIndex]:
                ans += 1
                gIndex +=1
            sIndex += 1
        return ans

    def repeatedSubstringPattern(self, s: str) -> bool:
        # 数学方式解决，高效
        # new = s + s
        # return s in new[1:-1]

        # 暴力解决
        length = len(s)
        middle = length // 2
        for i in range(1, middle + 1):
            if length % i == 0:
                subStr = s[:i]
                if subStr * (length // i) == s:
                    return True
        return False

    def hammingDistance(self, x: int, y: int) -> int:
        # ans = 0
        # while x > 0 and y > 0:
        #     lastX = x & 1
        #     lastY = y & 1
        #     if lastX != lastY:
        #         ans += 1
        #     x >>= 1
        #     y >>= 1
        
        # while x > 0:
        #     x &= (x - 1)
        #     ans += 1
        # while y > 0:
        #     y &= (y - 1)
        #     ans += 1
        # return ans
        z = x ^ y
        dis = 0
        while z > 0:
            z &= (z - 1)
            dis += 1
        return dis

    def islandPerimeter(self, grid: list[list[int]]) -> int:
        islands = 0
        connects = 0
        for i, l in enumerate(grid):
            for j, v in enumerate(l):
                if v == 1:
                    islands += 1
                    if j > 0 and l[j-1] == 1:
                        connects += 1
                    if i > 0 and grid[i-1][j] == 1:
                        connects += 1
        return islands * 4 - connects * 2
    
    def findComplement(self, num: int) -> int:
        # binaryStr = ""
        # while num > 0:
        #     last = (num & 1) ^ 1
        #     binaryStr = str(last) + binaryStr
        #     num >>= 1
        # ans = int(binaryStr, 2)
        # return ans
        length = num.bit_length()
        mask = (1 << length) - 1
        return mask ^ num
    
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        # i = len(s) - 1
        # subStr = ""
        # ans = ""
        # while i >= 0:
        #     c = s[i]
        #     i -= 1
        #     if c == "-":
        #         continue

        #     subStr = c.upper() + subStr
        #     if len(subStr) >= k:
        #         if ans:
        #             ans = subStr + "-" + ans
        #         else:
        #             ans = subStr
        #         subStr = ""
        # if subStr:
        #     if ans:
        #         ans = subStr + "-" + ans
        #     else:
        #         ans = subStr
        # return ans
        s = s.replace("-", "").upper()
        ans = ""
        count = 0
        for c in s[::-1]:
            if count == k:
                ans += "-"
                count = 0
            ans += c
            count += 1
        return ans[::-1]
    
    def findMaxConsecutiveOnes(self, nums: list[int]) -> int:
        max = 0
        temp = 0
        for n in nums:
            if n == 0:
                temp = 0
            else:
                temp += 1
                if temp > max:
                    max = temp
        return max
    
    def constructRectangle(self, area: int) -> list[int]:
        s = int(area ** 0.5)
        for i in range(s, 0, -1):
            if area % i == 0:
                l = area // i
                w = i
                break
        return [l, w]

    def findPoisonedDuration(self, timeSeries: list[int], duration: int) -> int:
        # total = 0
        # next = -1
        # for i, v in enumerate(timeSeries):
        #     if v > next:
        #         total += duration
        #     else:
        #         total += (v - timeSeries[i - 1])
        #     next = v + duration - 1
        # return total
        total = duration * len(timeSeries)
        for i in range(1, len(timeSeries)):
            gap = duration - (timeSeries[i] - timeSeries[i - 1])
            if gap > 0:
                total -= gap
        return total

    def nextGreaterElement(self, nums1: list[int], nums2: list[int]) -> list[int]:
        # 暴力解决
        # ans = [-1] * len(nums1)
        # for i1, v1 in enumerate(nums1):
        #     find = False
        #     for i2, v2 in enumerate(nums2):
        #         if v2 == v1:
        #             find = True
        #         elif find and v2 > v1:
        #             ans[i1] = v2
        #             break
        # return ans

        mapper = {}
        stack = []
        for n in nums2:
            while stack and stack[-1] < n:
                mapper[stack.pop()] = n
            stack.append(n)
        while stack:
            mapper[stack.pop()] = -1
        
        ans = []
        for n in nums1:
            ans.append(mapper[n])
        return ans
    
    def findWords(self, words: list[str]) -> list[str]:
        # mapper = {}
        # for c in "qwertyuiop":
        #     mapper[c] = 1
        # for c in "asdfghjkl":
        #     mapper[c] = 2
        # for c in "zxcvbnm":
        #     mapper[c] = 3

        # ans = []
        # for w in words:
        #     same = True
        #     for i in range(1, len(w)):
        #         if mapper[w[i].lower()] != mapper[w[i - 1].lower()]:
        #             same = False
        #             break
        #     if same:
        #         ans.append(w)
        # return ans
        set1 = set("qwertyuiop")
        set2 = set("asdfghjkl")
        set3 = set("zxcvbnm")
        ans = []
        for w in words:
            temp = w.lower()
            if set(temp) <= set1 or set(temp) <= set2 or set(temp) <= set3:
                ans.append(w)
        return ans
        
    def findMode(self, root: Optional[TreeNode]) -> list[int]:
        maxCount = 0
        currentCount = 0
        currentValue = None
        ans = []

        def inorder(node: Optional[TreeNode]):
            if not node:
                return
            nonlocal maxCount, currentCount, currentValue, ans
            
            inorder(node.left)
            
            if node.val == currentValue:
                currentCount += 1
            else:
                currentCount = 1
                currentValue = node.val
            
            if currentCount > maxCount:
                maxCount = currentCount
                ans = [node.val]
            elif currentCount == maxCount:
                ans.append(node.val)

            inorder(node.right)

        inorder(root)
        return ans

    def convertToBase7(self, num: int) -> str:
        ans = ""
        n = abs(num)
        while n:
            ans = str(n % 7) + ans
            n //= 7
        if num < 0:
            ans = '-' + ans
        return ans or "0"

    def findRelativeRanks(self, score: list[int]) -> list[str]:
        s = sorted(score, reverse=True)
        mapper = {}
        for i, v in enumerate(s):
            if i == 0:
                mapper[v] = "Gold Medal"
            elif i == 1:
                mapper[v] = "Silver Medal"
            elif i == 2:
                mapper[v] = "Bronze Medal"
            else:
                mapper[v] = str(i + 1)
        ans = []
        for s in score:
            rank = mapper[s]
            ans.append(rank)
        return ans

    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 1:
            return False
        sum = 1
        limit = int(num ** 0.5) + 1
        for i in range(2, limit):
            if num % i == 0:
                sum += i
                if i != (num // i):
                    sum += (num // i)
        return sum == num

    def fib(self, n: int) -> int:
        # if n == 1 or n == 0:
        #     return n
        # return self.fib(n - 1) + self.fib(n - 2)

        map = {1:1, 0:0}
        def fibc(n: int) -> int:
            nonlocal map
            if n in map:
                return map[n]
            ans = fibc(n - 1) + fibc(n - 2)
            map[n] = ans
            return ans
        return fibc(n)
        
    def detectCapitalUse(self, word: str) -> bool:
        return word.islower() or word.isupper() or word.istitle()
    
    def findLUSlength(self, a: str, b: str) -> int:
        if a == b:
            return -1
        return max(len(a), len(b))
    
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        min_diff = float('inf')
        pre = None

        def in_order(root: Optional[TreeNode]):
            if not root:
                return
            nonlocal min_diff, pre

            in_order(root.left)
            if pre is not None:
                min_diff = min(min_diff, root.val - pre)
            pre = root.val
            in_order(root.right)
        in_order(root)
        return min_diff
    
    def reverseStr(self, s: str, k: int) -> str:
        ans = ""
        t = 0
        for i in range(0, len(s), k):
            if (i // k) % 2 == 0:
                sub = s[i: i+k][::-1] 
            else:
                sub = s[i: i+k]
            ans += sub
        return ans

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        maximal = 0
        def depth(root: Optional[TreeNode]) -> int:
            nonlocal maximal
            if root is None:
                return 0
            lp = depth(root.left)
            rp = depth(root.right)
            if lp + rp > maximal:
                maximal = lp + rp
            dep = 1 + max(lp, rp)
            return dep
        depth(root)
        return maximal
    
    def checkRecord(self, s: str) -> bool:
        # if "LLL" in s:
        #     return False
        # absence = 0
        # for c in s:
        #     if c == "A":
        #         absence += 1
        #         if absence >= 2:
        #             return False
        # return True

        return "LLL" not in s and s.count("A") < 2
    
    def reverseWords(self, s: str) -> str:
        words = s.split()
        ans = " ".join(w[::-1] for w in words)
        return ans
    
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        if not root.children:
            return 1
        
        maxSubDepth = 0
        for n in root.children:
            maxSubDepth = max(self.maxDepth(n), maxSubDepth)
        return 1 + maxSubDepth

    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        sum = 0
        for n in nums[::2]:
            sum += n
        return sum
    
    
    def findTilt(self, root: Optional[TreeNode]) -> int:
        ans = 0
        def dfs(root: Optional[TreeNode]):
            nonlocal ans
            if not root:
                return 0
            leftSum = dfs(root.left)
            rightSum = dfs(root.right)
            ans += abs(leftSum - rightSum)
            return leftSum + rightSum + root.val
        dfs(root)
        return ans

    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        if m * n != r * c:
            return mat
        
        ans: List[List[int]] = []
        i = 0
        for sub in mat:
            for n in sub:
                if i % c == 0:
                    ans.append([n])
                else:
                    ans[-1].append(n)
                i += 1

        return ans
    
    def preorderisSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSametree(root1: Optional[TreeNode], root2: Optional[TreeNode]):
            if not root1 and not root2:
                return True
            if not root1 or not root2:
                return False
            if root1.val != root2.val:
                return False
            return isSametree(root1.left, root2.left) and isSametree(root1.right, root2.right)
        
        if not root:
            return False
        return isSametree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def distributeCandies(self, candyType: List[int]) -> int:
        type = len(set(candyType))
        n = len(candyType) // 2
        return min(type, n)
    
    def preorder(self, root: 'Node') -> List[int]:
        # ans = []
        # def dfs(root: 'Node'):
        #     if not root:
        #         return
        #     ans.append(root.val)
        #     for n in root.children:
        #         dfs(n)
        # dfs(root)
        # return ans

        if not root:
            return []

        stack = [root]
        result = []

        while stack:
            node = stack.pop()  # 从右端弹出节点
            result.append(node.val)  # 访问当前节点
            stack.extend(reversed(node.children))  # 按从右到左的顺序加入子节点

        return result
    
    def postorder(self, root: 'Node') -> List[int]:
        # ans = []
        # def dfs(root: 'Node'):
        #     if not root:
        #         return []
        #     for n in root.children:
        #         dfs(n)
        #     ans.append(root.val)
        # dfs(root)
        # return ans
        
        if not root:
            return
        
        stack = [root]
        ans = []
        while stack:
            node = stack.pop()
            ans.append(node.val)
            stack.extend(node.children)
        return ans[::-1]
    
    def findLHS(self, nums: List[int]) -> int:
        counter = Counter(nums)
        ans = 0
        for key in counter:
            if key+1 in counter:
                length = counter[key] + counter[key+1]
                ans = max(length, ans)
        return ans

    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        mina = m
        minb = n
        for arr in ops:
            mina = min(mina, arr[0])
            minb = min(minb, arr[-1])
        return mina * minb
    
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        lis1_map = {v: k for k,v in enumerate(list1)}
        ans = []
        minimum = float('inf')
        for i, v in enumerate(list2):
            if v in lis1_map:
                sum = i + lis1_map[v]
                if sum < minimum:
                    minimum = sum
                    ans = [v]
                elif sum == minimum:
                    ans.append(v)
        return ans
    
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        length = len(flowerbed)
        for i in range(length):
            if n <= 0:
                break
            pre = (i == 0 or flowerbed[i-1] == 0)
            next = (i == length - 1 or flowerbed[i+1] == 0)
            if flowerbed[i] == 0 and pre and next:
                flowerbed[i] = 1
                n -= 1
        return n <= 0
    
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1 and not root2:
            return None
        if not root1:
            return root2
        if not root2:
            return root1
        node = TreeNode()
        node.val = root1.val + root2.val
        node.left = self.mergeTrees(root1.left, root2.left)
        node.right = self.mergeTrees(root1.right, root2.right)
        return node
    
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        sum1 = nums[-1] * nums[-2] * nums[-3]   # Top 3 largest numbers
        sum2 = nums[0] * nums[1] * nums[-1]     # Two smallest + largest
        return max(sum1, sum2)
            
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        stack = deque([root])
        ans = []
        while stack:
            sum = 0
            length = times = len(stack)
            while times > 0:
                node = stack.popleft()
                sum += node.val
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
                times -= 1
            ans.append(sum / length)
        return ans
    
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        maxSum = sum(nums[:k])
        for i in range(k, len(nums)):
            cur = maxSum + nums[i] - nums[i-k]
            maxSum = max(maxSum, cur)
        return maxSum / k
    
    def findErrorNums(self, nums: List[int]) -> List[int]:
        dup = sum(nums) - sum(set(nums))
        mis = (1+len(nums)) * len(nums) // 2 - sum(set(nums))
        return [dup, mis]
    
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        nums = []
        def inorder(root: Optional[TreeNode]):
            if not root:
                return
            inorder(root.left)
            nums.append(root.val)
            inorder(root.right)
        
        inorder(root)
        if k < nums[0] or k > nums[-1] * 2:
            return False
        
        left = 0
        right = len(nums) - 1
        while left < right:
            if nums[left] + nums[right] == k:
                return True
            elif nums[left] + nums[right] < k:
                left += 1
            else:
                right -= 1
        return False

    def judgeCircle(self, moves: str) -> bool:
        v = 0
        h = 0
        for c in moves:
            if c == 'U':
                v += 1
            elif c == 'D':
                v -= 1
            elif c == 'L':
                h += 1
            elif c == 'R':
                h -= 1
        return v == 0 and h == 0

    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        ans = []
        rows = len(img)
        cols = len(img[0])
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1), ( 0, 0), ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)]
        for i in range(rows):
            subList = []
            for j in range(cols):
                sum = 0
                counts = 0
                for offI, offJ in directions:
                    tarI = offI + i
                    tarJ = offJ + j
                    if tarI >=0 and tarI < rows and tarJ >= 0 and tarJ < cols:
                        counts += 1
                        sum += img[tarI][tarJ]
                subList.append(sum // counts)
            ans.append(subList)

        return ans

    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        secondMinimal = float('inf')

        def dfs(node: Optional[TreeNode]):
            nonlocal secondMinimal
            if not node:
                return
            if root.val < node.val < secondMinimal:
                secondMinimal = node.val
            elif root.val == node.val:
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        return secondMinimal if secondMinimal < float('inf') else -1

    def findLengthOfLCIS(self, nums: List[int]) -> int:
        maxLen = 1
        curLen = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                curLen += 1
                maxLen = max(maxLen, curLen)
            else:
                curLen = 0
        return maxLen

    def validPalindrome(self, s: str) -> bool:
        def validPalindrome(s: str, left: int, right: int):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        left  = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return validPalindrome(s, left + 1, right) or validPalindrome(s, left, right - 1)
            left += 1
            right -= 1
        return True
    
    def calPoints(self, operations: List[str]) -> int:
        points = []
        for c in operations:
            if c == '+':
                points.append(points[-1] + points[-2])
            elif c == 'D':
                points.append(points[-1] * 2)
            elif c == 'C':
                points.pop()
            else:
                points.append(int(c))
        return sum(points)

    def hasAlternatingBits(self, n: int) -> bool:
        last = float('-inf')
        while n:
            temp = n & 1
            if last == temp:
                return False
            last = temp
            n >>= 1
        return True

    def countBinarySubstrings(self, s: str) -> int:
        counts = []
        curLen = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                curLen += 1
            else:
                counts.append(curLen)
                curLen = 1
        # add last substr length
        counts.append(curLen)
        result = 0
        for i in range(1, len(counts)):
            l = min(counts[i], counts[i - 1])
            result += l
        return result

    def findShortestSubArray(self, nums: List[int]) -> int:
        freq = {}
        first = {}
        last = {}
        degree = 0
        for index, value in enumerate(nums):
            times = freq.get(value, 0) + 1
            degree = max(degree, times)
            freq[value] = times
            if value not in first:
                first[value] = index
            last[value] = index
        
        minimalLength = float('inf')
        for key, value in freq.items():
            if value == degree:
                minimalLength = min(minimalLength, last[key] - first[key] + 1)
        return minimalLength
        
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return None

        if root.val < val:
            return self.searchBST(root.right, val)
        elif root.val > val:
            return self.searchBST(root.left, val)
        else:
            return root

    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            middle = (left + right) // 2
            if nums[middle] > target:
                right = middle - 1
            elif nums[middle] < target:
                left = middle + 1
            else:
                return middle
        return -1
    
    def toLowerCase(self, s: str) -> str:
        return s.lower()
    
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        # i = 0
        # while i < len(bits) - 1:
        #     if bits[i] == 0:
        #         i += 1
        #     elif bits[i] == 1:
        #         i += 2

        # return i <= len(bits) - 1

        # 如果最后的 0 前面有奇数个连续的 1，说明最后一个 0 是 2-bit character 的一部分。
        # 如果前面有偶数个连续的 1，说明最后一个 0 是独立的 1-bit character。
        ans = True
        for c in bits[-2::-1]:
            if c:
                ans = not ans
            else:
                break
        return ans
    
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)
        leftSum = 0
        for i, v in enumerate(nums):
            if leftSum + v == total:
                return i
            leftSum += v
            total -= v
        return -1
            
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        ans = []
        for i in range(left, right + 1):
            temp = i
            dividing = True
            while temp != 0:
                last = temp % 10
                if last == 0 or i % last != 0:
                    dividing = False
                    break
                temp //= 10
            if dividing:
                ans.append(i)
        return ans
    
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        original_color = image[sr][sc]
        if original_color == color:
            return image
        
        rows = len(image)
        cols = len(image[0])
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        q = deque([(sr,sc)])
        
        while q:
            r, c = q.popleft()
            image[r][c] = color

            for offset_r, offset_c in offsets:
                t_r = r + offset_r
                t_c = c + offset_c
                if t_r < 0 or t_r >= rows:
                    continue
                if t_c < 0 or t_c >= cols:
                    continue
                if image[t_r][t_c] == original_color:
                    q.append((t_r, t_c))
        return image

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        if letters[0] > target or letters[-1] <= target:
            return letters[0]
        
        left = 0
        right = len(letters) - 1
        while left <= right:
            middle = (left + right) // 2
            if letters[middle] <= target:
                left = middle + 1
            else:
                right = middle - 1
        return letters[left]

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        prev1, prev2 = 0, 0

        for i in range(2, len(cost) + 1):
            min1 = cost[i - 1] + prev1
            min2 = cost[i - 2] + prev2
            cur = min(min1, min2)
            prev1, prev2 = cur, prev1
        return prev1
    
    def dominantIndex(self, nums: List[int]) -> int:
        firstIndex, secondIndex = 0, 1
        if nums[1] > nums[0]:
            firstIndex, secondIndex = 1, 0
        for i in range(2, len(nums)):
            if nums[i] > nums[firstIndex]:
                secondIndex = firstIndex
                firstIndex = i
            elif nums[i] > nums[secondIndex]:
                secondIndex = i
        if nums[secondIndex] == 0 or nums[firstIndex] / nums[secondIndex] >= 2:
            return firstIndex
        return  -1
    
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        dic = Counter(char.lower() for char in licensePlate if char.isalpha())        
        words.sort(key=len)
        for word in words:
            complete = True
            for key, value in dic.items():
                if value > word.count(key):
                    complete = False
                    break
            if complete:
                return word
        return None
    
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def countOnes(num: int) -> int:
            ans = 0
            while num:
                if num & 1:
                    ans += 1
                num >>= 1
            return ans
        
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        ans = 0
        for i in range(left, right + 1):
            bits = bin(i).count('1')
            if bits in primes:
                ans += 1
        return ans

    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        rows = len(matrix)
        columns = len(matrix[0])
        for i in range(rows):
            for j in range(columns):
                diagonal_R = i + 1
                diagonal_C = j + 1
                if diagonal_R >= rows or diagonal_C >= columns:
                    continue
                if matrix[i][j] != matrix[diagonal_R][diagonal_C]:
                    return False
        return True

    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        num = 0
        for c in jewels:
            num += stones.count(c)
        return num

    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        pre = None
        ans = float('inf')

        def dfs(root: Optional[TreeNode]):
            nonlocal pre, ans
            if not root:
                return
            dfs(root.left)
            if pre is None:
                pre = root.val
            else:
                ans = min(ans, root.val - pre)
                pre = root.val
            dfs(root.right)
        
        dfs(root)
        return ans

    def rotateString(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        
        # for i in range(len(s)):
        #     shifted = s[i:] + s[:i]
        #     if shifted == goal:
        #         return True
        # return False
        return goal in (s + s)

    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        mapper = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        morses = set()
        for word in words:
            morse = ""
            for c in word:
                index = ord(c) - ord('a')
                morse += mapper[index]
            morses.add(morse)
        return len(morses)

    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        lines = 1
        pixels = 0
        for c in s:
            index = ord(c) - ord('a')
            pixels += widths[index]
            if pixels > 100:
                lines += 1
                pixels = widths[index]
        return [lines, pixels]

    def largestTriangleArea(self, points: List[List[int]]) -> float:
        def triangleArea(a: float, b: float, c: float) -> float:
            # 套用公式计算三角形面积,鞋带公式（Shoelace Formula）
            return 0.5 * abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        
        area = 0
        for a, b, c in combinations(points, 3):
            area = max(area, triangleArea(a, b, c))
        return area

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        filteredWords = re.findall(r'\b[a-zA-Z]+\b', paragraph.lower())
        wordsCounter = Counter(filteredWords).most_common()
        bannedSet = set(banned)

        for word, _ in wordsCounter:
            if word not in bannedSet:
                return word
        return None

    def shortestToChar(self, s: str, c: str) -> List[int]:
        # ans = [0] * len(s)
        # cIndexes = []
        # for i, v in enumerate(s):
        #     if v == c:
        #         cIndexes.append(i)
        
        # def distance(nums: List[int], index: int) -> int:
        #     pos = bisect.bisect_left(nums, index)
        #     if pos == 0:
        #         return nums[0] - index
        #     if pos == len(nums):
        #         return index - nums[-1]
            
        #     before = index - nums[pos - 1]
        #     after = nums[pos] - index
        #     return min(before, after)
        
        # for i, v in enumerate(s):
        #     if v == c:
        #         continue
        #     ans[i] = distance(cIndexes, i)
        # return ans

        n = len(s)
        ans = [0] * n

        prev = float('-inf')
        for i in range(n):
            if s[i] == c:
                prev = i
            ans[i] = abs(i - prev)
        
        prev = float('inf')
        for i in range(n - 1, -1, -1):
            if s[i] == c:
                prev = i
            ans[i] = min(ans[i], abs(i - prev))
        return ans

    def toGoatLatin(self, sentence: str) -> str:
        vowels = {'a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U'}
        originalWords = sentence.split()
        latinWords = []
        for index, word  in enumerate(originalWords):
            if word[0] not in vowels:
                word = word[1:] + word[0]
            word += ("ma" + 'a' * (index + 1))
            latinWords.append(word)
        return ' '.join(latinWords)

    def largeGroupPositions(self, s: str) -> List[List[int]]:
        ans = []
        n = len(s)
        start = 0
        for i in range(1, n + 1):
            if i == n or s[i] != s[i - 1]:
                if i - start >= 3:
                    ans.append([start, i - 1])
                start = i
        return ans
    
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        for sub in image:
            sub.reverse()
            for i in range(len(sub)):
                sub[i] ^= 1
        return image

    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        if rec1[0] >= rec2[2] or rec1[1] >= rec2[3]:
            return False
        if rec1[2] <= rec2[0] or rec1[3] <= rec2[1]:
            return False
        return True

    def backspaceCompare(self, s: str, t: str) -> bool:
        sArray = []
        tArray = []
        for c in s:
            if c == '#':
                if sArray:
                    sArray.pop()
            else:
                sArray.append(c)
        for c in t:
            if c == '#':
                if tArray:
                    tArray.pop()
            else:
                tArray.append(c)
        return sArray == tArray

    def buddyStrings(self, s: str, goal: str) -> bool:
        sn = len(s)
        gn = len(goal)

        if sn != gn:
            return False
        
        if s == goal:
            temp = set(s)
            return len(temp) < len(goal)
        
        diff = [(a, b) for a, b in zip(s, goal) if a != b]
        if len(diff) != 2:
            return False
        
        return diff[0] == diff[1][::-1]

    def lemonadeChange(self, bills: List[int]) -> bool:
        change5 = 0
        change10 = 0

        for bill in bills:
            if bill == 5:
                change5 += 1
                continue
            if bill == 10:
                if change5 <= 0:
                    return False
                change5 -= 1
                change10 += 1
            if bill == 20:
                if change10 > 0 and change5 > 0:
                    change10 -= 1
                    change5 -= 1
                elif change5 >= 3:
                    change5 -= 3
                else:
                    return False
        return True

    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        rows = len(matrix)
        columns = len(matrix[0])
        ans = []

        for column in range(columns):
            sub = []
            for row in range(rows):
                sub.append(matrix[row][column])
            ans.append(sub)
        return ans

    def binaryGap(self, n: int) -> int:
        # maxGap = 0
        # curGap = None
        # while n > 0:
        #     last = n & 1
        #     if curGap is None and last == 0:
        #         n >>= 1
        #         continue
            
        #     if last == 1:
        #         if curGap is not None:
        #             maxGap = max(maxGap, curGap)
        #         curGap = 1
        #     else:
        #         curGap += 1
        #     n >>= 1

        # return maxGap

        positions = [i for i, v in enumerate(bin(n)[2:]) if v == '1']
        print(positions)
        return max((positions[i] - positions[i - 1] for i in range(1, len(positions))), default=0)

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def dfs(root: Optional[TreeNode], nums: List[int]):
            if not root:
                return
            if not root.left and not root.right:
                nums.append(root.val)
            dfs(root.left, nums)
            dfs(root.right, nums)

        nums1 = []
        nums2 = []
        dfs(root1, nums1)
        dfs(root2, nums2)
        return nums1 == nums2

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
    
    def projectionArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        columns = len(grid[0])
        
        area = 0
        
        for i in range(rows):
            maxRow = 0
            maxColumn = 0
            for j in range(columns):
                maxRow = max(maxRow, grid[i][j])
                maxColumn = max(maxColumn, grid[j][i])
                if grid[i][j] > 0:
                    area += 1
            area += (maxRow + maxColumn)
        return area
            
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        counter = Counter(s1.split() + s2.split())

        return [word for word in counter if counter[word] == 1]

    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        sumA = sum(aliceSizes)
        sumB = sum(bobSizes)
        diff = (sumA - sumB) // 2

        bobSet = set(bobSizes)

        for x in aliceSizes:
            y = x - diff
            if y in bobSet:
                return [x, y]
        return []

    def surfaceArea(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        column = len(grid[0])
        overlaps = 0
        totalSurface = 0

        for i in range(rows):
            for j in range(column):
                if grid[i][j] == 0:
                    continue
                totalSurface += grid[i][j] * 6

                overlaps += (grid[i][j] - 1) * 2
                if i + 1 < rows and grid[i + 1][j] > 0:
                    overlaps += min(grid[i][j], grid[i + 1][j]) * 2
                if j + 1 < column and grid[i][j + 1] > 0: 
                    overlaps += min(grid[i][j], grid[i][j + 1]) * 2
        
        return totalSurface - overlaps
    
    def getDecimalValue(self, head: Optional[ListNode]) -> int:
        ans = 0
        while head:
            ans = ans * 2 + head.val
            head = head.next
        return ans
    
    def findNumbers(self, nums: List[int]) -> int:
        return sum(1 for num in nums if len(str(num)) % 2 == 0)
    
    def replaceElements(self, arr: List[int]) -> List[int]:
        rightMax = -1
        for i in range(len(arr) - 1, -1, -1):
            temp = rightMax
            rightMax = max(rightMax, arr[i])
            arr[i] = temp
        return arr
                    
    def sumZero(self, n: int) -> List[int]:
        # ans = []
        # if n % 2 != 0:
        #     ans.append(0)
        # repeat = n // 2
        # for i in range(1, repeat + 1):
        #     ans.append(i)
        #     ans.append(-i)
        # return ans
        serial = range(1 - n, n, 2)
        return list(serial)

# Characters ('a' to 'i') are represented by ('1' to '9') respectively.
# Characters ('j' to 'z') are represented by ('10#' to '26#') respectively.
    def freqAlphabets(self, s: str) -> str:
        num_to_char = {str(i): chr(i + 96) for i in range(1, 27)}
        ans = ""
        index = len(s) - 1
        while index >= 0:
            if s[index] == '#':
                key = s[index-2:index]
                ans += num_to_char[key]
                index -= 3
            else:
                key = s[index]
                ans += num_to_char[key]
                index -= 1
        return ans[::-1]

    def decompressRLElist(self, nums: List[int]) -> List[int]:
        ans = []
        for i in range(1, len(nums), 2):
            ans += [nums[i]] * nums[i-1]
        return ans
    
    def getNoZeroIntegers(self, n: int) -> List[int]:
        def containsZero(num):
            if num == 0:
                return True
            num = abs(num)
            while num:
                if num % 10 == 0:
                    return True
                num //= 10
            return False

        for i in range(1, n // 2 + 1):
            if not containsZero(i) and not containsZero(n-i):
                return [i, n-i]
        return [-1, -1]

    def maximum69Number(self, num: int):
        s_list = list(str(num))
        for index, char in enumerate(s_list):
            if char == '6':
                s_list[index] = '9'
                break
        
        return int(''.join(s_list))

    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        sorted_arr = sorted(set(arr))
        rank_map = { v: i + 1 for i, v in enumerate(sorted_arr) }
        return [rank_map[num] for num in arr]

    def removePalindromeSub(self, s: str) -> int:
        return 1 if s == s[::-1] else 2

    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        sodilers = [(sum(row), index) for index, row in enumerate(mat)]
        sodilers.sort()
        return [index for _, index in sodilers[:k]]
    
    def numberOfSteps(self, num: int) -> int:
        steps = 0
        while num > 0:
            if num % 2 == 0:
                num //= 2
            else:
                num -= 1
            steps += 1
        return steps

    def checkIfExist(self, arr: List[int]) -> bool:
        seen = set()
        for num in arr:
            if num * 2 in seen or (num % 2 == 0 and num // 2 in seen):
                return True
            seen.add(num)
        return False

    def countNegatives(self, grid: List[List[int]]) -> int:
        rows, columns = len(grid), len(grid[0])
        m = rows - 1
        n = 0

        ans = 0

        while m >= 0 and n < columns:
            v = grid[m][n]
            if v < 0:
                ans += (columns - n)
                m -= 1
            else:
                n += 1
        return ans
    
    def sortByBits(self, arr: List[int]) -> List[int]:
        return sorted(arr, key=lambda x: (bin(x).count('1'), x))
    
    def daysBetweenDates(self, date1: string, date2: string) -> int:
        d_format = "%Y-%m-%d"
        d1 = datetime.strptime(date1, d_format)
        d2 = datetime.strptime(date2, d_format)
        days = (d1 - d2).days
        return abs(days)

    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        sorted_num = sorted(nums)
        num_map = {}
        for i, v in enumerate(sorted_num):
            num_map.setdefault(v, i)
        ans = [num_map[n] for n in nums]
        return ans

    def sortString(self, s: str) -> str:
        counter = Counter(s)
        sorted_char = sorted(counter.keys())
        res = []

        while len(res) < len(s):
            for char in sorted_char:
                if counter[char] > 0:
                    res.append(char)
                    counter[char] -= 1
            for char in reversed(sorted_char):
                if counter[char] > 0:
                    res.append(char)
                    counter[char] -= 1
        
        return ''.join(res)

    def generateTheString(self, n: int) -> str:
        if n % 2 == 0:
            return 'a' * (n - 1) + 'b'
        
        return 'a' * n

    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        if not original:
            return None
        
        if original is target:
            return cloned
        
        left = self.getTargetCopy(original.left, cloned.left, target)
        if left:
            return left
        
        return self.getTargetCopy(original.right, cloned.right, target)

    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        minimums = [min(subList) for subList in matrix]
        maximums = [max(col) for col in zip(*matrix)]        
        return list(set(minimums) & set(maximums))

    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        set_arr2 = set(arr2)
        ans = 0

        for num in arr1:
            valid = True
            for i in range(d + 1):
                if num + i in set_arr2 or num - i in set_arr2:
                    valid = False
                    break
            if valid:
                ans += 1
        return ans

    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        ans = []
        for i in range(len(index)):
            m = index[i]
            n = nums[i]
            ans.insert(m, n)
        return ans

    def findLucky(self, arr: List[int]) -> int:
        counted = Counter(arr)
        lucky_numbers = [key for key, value in counted.items() if key == value]
        return max(lucky_numbers) if lucky_numbers else -1
    
    def countLargestGroup(self, n: int) -> int:
        def digitsSum(n: int) -> int:
            sum = 0
            while n > 0:
                sum += n % 10
                n //= 10
            return sum
        
        dic = {}
        for i in range(1, n + 1):
            s = digitsSum(i)
            dic[s] = dic.get(s, 0) + 1
        maxinum = max(dic.values())
        keys = [k for k, v in dic.items() if v == maxinum]
        return len(keys)

    def minSubsequence(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(nums, reverse=True)
        sum_nums = sum(nums)
        
        ans = []
        total = 0
        for n in sorted_nums:
            ans.append(n)
            total += n
            if total * 2 > sum_nums:
                break
        return ans
        
    def stringMatching(self, words: List[str]) -> List[str]:
        sorted_words = sorted(words, key=len)
        ans = set()
        for i, v in enumerate(sorted_words):
            for j in range(len(sorted_words) - 1, i, -1):
                if v in sorted_words[j]:
                    ans.add(v)
                    break
        return list(ans) 

    def minStartValue(self, nums: List[int]) -> int:
        minimum = float('inf')
        sum = 0
        for n in nums:
            sum += n
            minimum = min(minimum, sum)
        if minimum > 1:
            return 1
        return 1 - minimum

nums = [1, 2]
result = Solution().minStartValue(nums)
print(result)