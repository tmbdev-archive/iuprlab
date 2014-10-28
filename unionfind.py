__all__ = ["UnionFind"]

class UnionFind:
    def __init__(self):
        self.p = {}
        self.rank = {}
        self.max = -1
        self.nsets = 0
    def make_set(self,i):
        assert self.p.get(i)==None
        self.nsets += 1
        self.max += 1
        i = self.max
        self.p[i] = i
        self.rank[i] = 0
        return i
    def make_union(self,i,j):
        if i==j: return i
        i = self.find_set(i)
        j = self.find_set(j)
        if i!=j:
            self.nsets -=1
            self.link(i,j)
    def find_set(self,i):
        if i!=self.p[i]: self.p[i] = self.find_set(self.p[i])
        return self.p[i]
    def link(self,i,j):
        if self.rank[i]>self.rank[j]:
            self.p[j] = i
            return i
        else:
            self.p[i] = j
            if self.rank[i]==self.rank[j]: self.rank[j] += 1
            return j
    def same_set(self,i,j):
        return self.find_set(i)==self.find_set(j)
    def extract_sets(self):
        sets = {}
        for i in range(len(self.p)):
            key = self.find_set(i)
            sets[key] = sets.get(key,[])+[i]
        return [sets[i] for i in sets.keys()]


import unittest,random

def sort(x):
    x.sort()
    return x

class TestUnionFind(unittest.TestCase):
    def setUp(self):
        self.u = UnionFind()
    def testUnionFind(self):
        u = self.u
        for i in range(12): u.make_set(i)
        for i in range(12):
            for j in range(12):
                if i%3==j%3: u.make_union(i,j)
        sets = u.extract_sets()
        sets = sort([sort(x) for x in sets])
        print u.extract_sets()
        assert [0,3,6,9] in sets
        assert [1,4,7,10] in sets
        assert [2,5,8,11] in sets

if __name__ == "__main__":
    unittest.main()

