iNum = 7
iDen = 5

def calcRemainder(iNum, iDen):
    if iDen == 0:
        return "Dividing with 0"
    if abs(iDen) > abs(iNum):
        return iNum
    else:
        return calcRemainder(abs(iNum)-abs(iDen), iDen)
        
print(calcRemainder(iNum, iDen))