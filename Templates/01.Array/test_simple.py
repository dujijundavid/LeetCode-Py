#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("🎯 排序算法测试程序")
print("=" * 30)

# 测试基本功能
def bubble_sort(nums):
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

# 测试数据
test_data = [64, 34, 25, 12, 22, 11, 90]
print(f"原始数据: {test_data}")

# 测试排序
sorted_data = bubble_sort(test_data)
print(f"冒泡排序结果: {sorted_data}")

# 验证结果
expected = sorted(test_data)
print(f"Python内置排序: {expected}")

if sorted_data == expected:
    print("✅ 排序算法测试通过!")
else:
    print("❌ 排序算法测试失败!")

print("\n程序运行完成!") 