lNum = [5, 3, 1, 3, 5, 8, 9, -5, -2, 0, -7, -9]

def findMax(lNum):
    if len(lNum) == 1:
        return lNum[0]
    else:
        if lNum[-1] > findMax(lNum[:-1]):
            return lNum[-1]
        else:
            return findMax(lNum[:-1])
        
print(findMax(lNum))