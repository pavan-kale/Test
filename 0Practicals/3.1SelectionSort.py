def selection_sort(arr):
  for i in range(len(arr)):
    min_idx = i
    for j in range(i+1, len(arr)):
      if (arr[j]) < arr[min_idx]:
        min_idx = j
    
    arr[i], arr[min_idx] = arr[min_idx], arr[i]

  return arr


my_list=[]
num = int(input("Enter Number of elements: "))
for i in range (0,num):
    ele = int(input())
    my_list.append(ele)

#my_list = [64, 25, 12, 22, 11,101,22]
sorted_list = selection_sort(my_list)
print(sorted_list)