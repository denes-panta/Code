lNum = [1, 3, 5, 3, 7, 3, 1, 1, 5]

def getDistinct(lNum):
    lResult = [lNum[0]]
    
    for num in lNum[1:]:
        bFound = False
        i = 0
        while not bFound and i < len(lResult):
            if num == lResult[i]:
                bFound = True
            i += 1
        
        if not bFound:
            lResult.append(num)

    return lResult

print(getDistinct(lNum))

