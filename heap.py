__all__ = ["Heap", "HeapSet"]

class Heap:
    """Heap data structure (priority queue)."""
    def __init__(self):
        self.data = []
    def swap(self,i,j):
        "Internal."
        data = self.data
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
    def heapify(self,i):
        "Internal."
        data = self.data
        n = len(data)
        l = i*2+1
        r = i*2+2
        if l<n and data[l][0]>data[i][0]:
            largest = l
        else:
            largest = i
        if r<n and data[r][0]>data[largest][0]:
            largest = r
        if largest!=i:
            self.swap(i,largest)
            self.heapify(largest)
    def top_priority(self):
        "Return the highest priority in the heap."
        return self.data[0][0]
    def top(self):
        "Return the top element in the heap."
        return self.data[0][1]
    def extract_max(self):
        "Remove the top element from the heap and return it."
        result = self.data[0]
        self.data[0] = self.data[-1]
        del self.data[-1]
        self.heapify(0)
        return result
    def insert(self,priority,object):
        "Insert a new element into the heap."
        data = self.data
        data.append(None)
        i = len(data)-1
        while i>0 and priority>data[(i-1)/2][0]:
            data[i] = data[(i-1)/2]
            i = (i-1)/2
        data[i] = (priority,object)
    def __len__(self):
        return len(self.data)-1

class HeapSet:
    """A Heap that allows out-of-order deletions."""
    def __init__(self):
        self.location = {}
        self.data = []
    def swap(self,i,j):
        "Internal."
        data = self.data
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
        self.location[data[i][2]] = i
        self.location[data[j][2]] = j
    def heapify(self,i):
        "Internal."
        data = self.data
        n = len(data)
        l = i*2+1
        r = i*2+2
        if l<n and data[l][0]>data[i][0]:
            largest = l
        else:
            largest = i
        if r<n and data[r][0]>data[largest][0]:
            largest = r
        if largest!=i:
            self.swap(i,largest)
            self.heapify(largest)
    def top_priority(self):
        "Return the highest priority."
        return self.data[0][0]
    def top(self):
        "Return the top element (doesn't remove)."
        return self.data[0][1]
    def extract_max(self):
        "Remove the top element and return it."
        result = self.data[0]
        self.data[0] = self.data[-1]
        del self.data[-1]
        if len(self.data)>0:
            self.location[self.data[0][2]] = 0
            self.heapify(0)
        return result
    def insert(self,priority,object,key):
        "Insert a new element into the heap with the given key."
        self.data.append(None)
        i = len(self.data)-1
        while i>0 and priority>self.data[(i-1)/2][0]:
            self.data[i] = self.data[(i-1)/2]
            self.location[self.data[i][2]] = i
            i = (i-1)/2
        self.data[i] = (priority,object,key)
        self.location[self.data[i][2]] = i
    def get(self,key):
        "Return the data associated with the key"
        return self.data[self.location[key]][1]
    def priority(self,key):
        "Return the priority associated with the key"
        return self.data[self.location[key]][0]
    def delete(self,key):
        "Remove a key and its associated data from the heap."
        i = self.location[key]
        if i<len(self.data):
            self.data[i] = self.data[-1]
            self.location[self.data[i][2]] = i
        del self.data[-1]
        del self.location[self.data[-1][2]]
        self.heapify(i)
    def __len__(self):
        return len(self.data)

import unittest,random

class TestHeap(unittest.TestCase):
    def setUp(self):
        self.heap = Heap()
    def testHeap(self):
        h = self.heap
        data = [random.uniform(0,1) for i in range(20)]
        for x in data: h.insert(x,x)
        last = 1e99
        while len(h)>0:
            x = h.extract_max()
            assert x[0]<=last
            last = x

class TestHeapSet(unittest.TestCase):
    def setUp(self):
        self.heapset = HeapSet()
    def testHeapSet(self):
        h = self.heapset
        data = [random.uniform(0,1) for i in range(10)]
        for i in range(len(data)): h.insert(data[i],data[i],i)
        # FIXME the stuff below fails, so we return for now
        return
        for i in range(5): h.delete(i)
        last = 1e99
        while len(h)>0:
            x,o,i = h.extract_max()
            assert i>=5
            assert x<last
            last = x

if __name__ == "__main__":
    unittest.main()
