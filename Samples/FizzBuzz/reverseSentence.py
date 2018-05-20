sString = "Fizz Buzz Fizz Buzz Fizz Buzz"

def reverseSentence(sStr):
    lStr = sStr.split(" ")
    
    def revSent(lStr):
        if lStr == []:
            return []
        else:
            return list(lStr[-1]) + [" "] + revSent(lStr[0:-1])

    return "".join(revSent(lStr))
        
print(reverseSentence(sString)) 