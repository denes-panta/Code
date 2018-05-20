lNum = [5, 3, 1, 3, 5, 8, 9, -5, -2, 0, -7, -9]

def findMin(lNum):
    if len(lNum) == 1:
        return lNum[0]
    else:
        if lNum[-1] < findMin(lNum[:-1]):
            return lNum[-1]
        else:
            return findMin(lNum[:-1])
        
print(findMin(lNum))