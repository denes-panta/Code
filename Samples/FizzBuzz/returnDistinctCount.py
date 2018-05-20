lNum = [1, 3, 5, 3, 7, 3, 1, 1, 5]

def getDistinctCount(lNum):
    lResult = [lNum[0]]
    lCount = [1]    
    for num in lNum[1:]:
        bFound = False
        i = 0
        
        while not bFound and i < len(lResult):
            if num == lResult[i]:
                bFound = True
                lCount[i] += 1
            i += 1
        
        if not bFound:
            lResult.append(num)
            lCount.append(1)

    return list(zip(lResult, lCount))[:]

print(getDistinctCount(lNum))

