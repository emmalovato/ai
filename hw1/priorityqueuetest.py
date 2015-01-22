class PriorityQueue:
    def __init__(self):
        self.q = list()

    def put(self, item):
        data, priority = item
        self._insort_right((priority, data))

    def get(self):
        # print type(self.q[0][1])
        return self.q.pop(0)[1] # returns the first path in the queue

    # sort according to f cost such that the first path has the lowest f cost 
    def _insort_right(self, x):
        lo = 0
        hi = len(self.q)
        while lo < hi:
            mid = (lo+hi)/2
            if x[0] < self.q[mid][0]:
                hi = mid
            else:
                lo = mid+1
        self.q.insert(lo, x)
