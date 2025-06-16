#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排序算法章节总结 - 性能对比与学习工具
======================================

本程序提供：
1. 多种排序算法的性能对比
2. 时间空间复杂度的直观展示  
3. 不同数据规模和数据特征的测试
4. LeetCode题目的标准调用模板
5. 算法选择的决策树和方法论

作者: AI Assistant
日期: 2024
"""

import time
import random
import sys
from typing import List, Callable, Tuple, Dict
import copy

# 注意：如果需要绘图功能，请安装matplotlib: pip install matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("提示: 如需绘图功能，请安装matplotlib: pip install matplotlib")

# 导入所有排序算法实现
sys.path.append('.')

class SortingAlgorithms:
    """排序算法集合类"""
    
    @staticmethod
    def bubble_sort(nums: List[int]) -> List[int]:
        """冒泡排序 - O(n²) 时间, O(1) 空间"""
        nums = nums.copy()
        for i in range(len(nums) - 1):
            flag = False
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    flag = True
            if not flag:
                break
        return nums
    
    @staticmethod
    def selection_sort(nums: List[int]) -> List[int]:
        """选择排序 - O(n²) 时间, O(1) 空间"""
        nums = nums.copy()
        for i in range(len(nums) - 1):
            min_i = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[min_i]:
                    min_i = j
            if i != min_i:
                nums[i], nums[min_i] = nums[min_i], nums[i]
        return nums
    
    @staticmethod 
    def insertion_sort(nums: List[int]) -> List[int]:
        """插入排序 - O(n²) 时间, O(1) 空间, 对小数组和部分有序数组友好"""
        nums = nums.copy()
        for i in range(1, len(nums)):
            temp = nums[i]
            j = i
            while j > 0 and nums[j - 1] > temp:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = temp
        return nums
    
    @staticmethod
    def merge_sort(nums: List[int]) -> List[int]:
        """归并排序 - O(n log n) 时间, O(n) 空间, 稳定排序"""
        if len(nums) <= 1:
            return nums
        
        def merge(left: List[int], right: List[int]) -> List[int]:
            result = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            result.extend(left[i:])
            result.extend(right[j:])
            return result
        
        mid = len(nums) // 2
        left = SortingAlgorithms.merge_sort(nums[:mid])
        right = SortingAlgorithms.merge_sort(nums[mid:])
        return merge(left, right)
    
    @staticmethod
    def quick_sort(nums: List[int]) -> List[int]:
        """快速排序 - 平均O(n log n), 最坏O(n²) 时间, O(log n) 空间"""
        nums = nums.copy()
        
        def partition(low: int, high: int) -> int:
            pivot = nums[low]
            i, j = low, high
            while i < j:
                while i < j and nums[j] >= pivot:
                    j -= 1
                while i < j and nums[i] <= pivot:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[j], nums[low] = nums[low], nums[j]
            return j
        
        def quick_sort_helper(low: int, high: int):
            if low < high:
                pivot_i = partition(low, high)
                quick_sort_helper(low, pivot_i - 1)
                quick_sort_helper(pivot_i + 1, high)
        
        if nums:
            quick_sort_helper(0, len(nums) - 1)
        return nums
    
    @staticmethod
    def counting_sort(nums: List[int]) -> List[int]:
        """计数排序 - O(n+k) 时间, O(k) 空间, 适用于小范围整数"""
        if not nums:
            return []
        
        nums_min, nums_max = min(nums), max(nums)
        size = nums_max - nums_min + 1
        counts = [0] * size
        
        for num in nums:
            counts[num - nums_min] += 1
        
        for i in range(1, size):
            counts[i] += counts[i - 1]
        
        result = [0] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            num = nums[i]
            result[counts[num - nums_min] - 1] = num
            counts[num - nums_min] -= 1
        
        return result


class PerformanceTester:
    """性能测试类"""
    
    def __init__(self):
        self.algorithms = {
            'Bubble Sort': SortingAlgorithms.bubble_sort,
            'Selection Sort': SortingAlgorithms.selection_sort, 
            'Insertion Sort': SortingAlgorithms.insertion_sort,
            'Merge Sort': SortingAlgorithms.merge_sort,
            'Quick Sort': SortingAlgorithms.quick_sort,
            'Counting Sort': SortingAlgorithms.counting_sort
        }
        
        self.complexity_info = {
            'Bubble Sort': {'time': 'O(n²)', 'space': 'O(1)', 'stable': True},
            'Selection Sort': {'time': 'O(n²)', 'space': 'O(1)', 'stable': False},
            'Insertion Sort': {'time': 'O(n²)', 'space': 'O(1)', 'stable': True},
            'Merge Sort': {'time': 'O(n log n)', 'space': 'O(n)', 'stable': True},
            'Quick Sort': {'time': 'O(n log n)', 'space': 'O(log n)', 'stable': False},
            'Counting Sort': {'time': 'O(n+k)', 'space': 'O(k)', 'stable': True}
        }
    
    def generate_test_data(self, size: int, data_type: str = 'random') -> List[int]:
        """生成测试数据"""
        if data_type == 'random':
            return [random.randint(1, 1000) for _ in range(size)]
        elif data_type == 'sorted':
            return list(range(1, size + 1))
        elif data_type == 'reverse':
            return list(range(size, 0, -1))
        elif data_type == 'partially_sorted':
            data = list(range(1, size + 1))
            # 随机交换10%的元素
            for _ in range(size // 10):
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                data[i], data[j] = data[j], data[i]
            return data
        elif data_type == 'duplicates':
            return [random.randint(1, size // 4) for _ in range(size)]
    
    def measure_time(self, algorithm: Callable, data: List[int]) -> float:
        """测量算法执行时间"""
        start_time = time.perf_counter()
        try:
            algorithm(data)
            end_time = time.perf_counter()
            return end_time - start_time
        except:
            return float('inf')  # 算法执行失败
    
    def run_performance_test(self, sizes: List[int] = None, data_types: List[str] = None):
        """运行性能测试"""
        if sizes is None:
            sizes = [100, 500, 1000, 2000]
        if data_types is None:
            data_types = ['random', 'sorted', 'reverse', 'partially_sorted']
        
        print("🚀 排序算法性能测试报告")
        print("=" * 80)
        
        for data_type in data_types:
            print(f"\n📊 数据类型: {data_type.upper()}")
            print("-" * 60)
            print(f"{'算法名称':<15} {'时间复杂度':<12} {'空间复杂度':<12} {'稳定性':<8} ", end="")
            for size in sizes:
                print(f"{size}个元素", end="    ")
            print()
            print("-" * 60)
            
            for name, algorithm in self.algorithms.items():
                complexity = self.complexity_info[name]
                print(f"{name:<15} {complexity['time']:<12} {complexity['space']:<12} {'✓' if complexity['stable'] else '✗':<8} ", end="")
                
                for size in sizes:
                    test_data = self.generate_test_data(size, data_type)
                    execution_time = self.measure_time(algorithm, test_data)
                    
                    if execution_time == float('inf'):
                        print("FAIL      ", end="")
                    elif execution_time < 0.001:
                        print(f"{execution_time*1000:.2f}ms    ", end="")
                    else:
                        print(f"{execution_time:.3f}s     ", end="")
                print()
        
        print("\n" + "=" * 80)


class LeetCodeTemplate:
    """LeetCode问题模板和方法论"""
    
    @staticmethod
    def print_decision_tree():
        """打印算法选择决策树"""
        print("\n🌳 排序算法选择决策树")
        print("=" * 50)
        print("""
数据规模 <= 50?
├─ Yes → 插入排序 (简单，对小数组高效)
└─ No → 数据范围小且为非负整数?
    ├─ Yes → 计数排序 (线性时间)
    └─ No → 需要稳定排序?
        ├─ Yes → 归并排序 (保证O(n log n)，稳定)
        └─ No → 快速排序 (平均最快，原地排序)

特殊情况:
• 部分有序数据 → 插入排序
• 内存严格限制 → 堆排序
• 数据流排序 → 外部排序算法
• 字符串排序 → 基数排序
        """)
    
    @staticmethod
    def show_leetcode_patterns():
        """展示LeetCode中的排序应用模式"""
        print("\n📝 LeetCode排序应用模式")
        print("=" * 50)
        
        patterns = [
            {
                'pattern': '数组排序基础',
                'examples': ['912. 排序数组', '215. 数组中的第K个最大元素'],
                'template': '''
def sortArray(self, nums: List[int]) -> List[int]:
    # 根据数据特征选择合适算法
    if len(nums) <= 50:
        return self.insertion_sort(nums)
    elif all(0 <= x <= 1000 for x in nums):
        return self.counting_sort(nums)  
    else:
        return self.quick_sort(nums)
                '''
            },
            {
                'pattern': '双指针 + 排序',
                'examples': ['1. 两数之和', '15. 三数之和', '16. 最接近的三数之和'],
                'template': '''
def twoSum(self, nums: List[int], target: int) -> List[int]:
    # 先排序，然后双指针
    indexed_nums = [(num, i) for i, num in enumerate(nums)]
    indexed_nums.sort()  # 按值排序
    
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = indexed_nums[left][0] + indexed_nums[right][0]
        if current_sum == target:
            return [indexed_nums[left][1], indexed_nums[right][1]]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
                '''
            },
            {
                'pattern': '自定义排序',
                'examples': ['56. 合并区间', '252. 会议室', '435. 无重叠区间'],
                'template': '''
def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    # 按起始时间排序
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
                '''
            }
        ]
        
        for i, pattern in enumerate(patterns, 1):
            print(f"\n{i}. {pattern['pattern']}")
            print(f"   典型题目: {', '.join(pattern['examples'])}")
            print(f"   代码模板:")
            print(pattern['template'])
    
    @staticmethod
    def show_optimization_tips():
        """展示优化技巧"""
        print("\n💡 排序算法优化技巧")
        print("=" * 50)
        
        tips = [
            "1. 小数组优化: 当子数组长度 < 10时,切换到插入排序",
            "2. 三路快排: 处理大量重复元素时，使用三路快排",
            "3. 内省排序: 快排递归深度过深时切换到堆排序",
            "4. 混合排序: Timsort(Python内置)结合归并和插入排序",
            "5. 缓存友好: 考虑数据访问模式的局部性",
            "6. 并行化: 归并和快排可以并行处理子问题",
            "7. 原地排序: 优先选择空间复杂度为O(1)的算法"
        ]
        
        for tip in tips:
            print(f"  {tip}")


def main():
    """主函数 - 演示所有功能"""
    print("🎯 排序算法章节总结工具")
    print("=" * 50)
    
    # 1. 复杂度分析展示
    tester = PerformanceTester()
    print("\n📚 算法复杂度对比表")
    print("-" * 50)
    print(f"{'算法名称':<15} {'时间复杂度':<12} {'空间复杂度':<12} {'稳定性'}")
    print("-" * 50)
    for name, info in tester.complexity_info.items():
        stable = "✓" if info['stable'] else "✗"
        print(f"{name:<15} {info['time']:<12} {info['space']:<12} {stable}")
    
    # 2. 性能测试
    print("\n" + "="*50)
    choice = input("是否运行性能测试? (y/n): ").lower().strip()
    if choice == 'y':
        tester.run_performance_test(sizes=[100, 500, 1000], 
                                  data_types=['random', 'sorted'])
    
    # 3. LeetCode模板和方法论
    template = LeetCodeTemplate()
    template.print_decision_tree()
    template.show_leetcode_patterns()
    template.show_optimization_tips()
    
    # 4. 实战演示
    print("\n🔬 算法实战演示")
    print("=" * 50)
    test_data = [64, 34, 25, 12, 22, 11, 90]
    print(f"原始数据: {test_data}")
    
    algorithms_to_demo = ['Insertion Sort', 'Merge Sort', 'Quick Sort']
    for name in algorithms_to_demo:
        algorithm = tester.algorithms[name]
        sorted_data = algorithm(test_data)
        print(f"{name:<15}: {sorted_data}")
    
    print(f"\nPython内置排序: {sorted(test_data)}")
    
    print("\n✨ 学习建议:")
    print("  1. 理解每种算法的核心思想和适用场景")
    print("  2. 动手实现并调试每个算法") 
    print("  3. 在LeetCode上练习相关题目")
    print("  4. 关注算法的稳定性和空间复杂度")
    print("  5. 学会根据数据特征选择最优算法")


if __name__ == "__main__":
    main() 