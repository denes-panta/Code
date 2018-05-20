sString = "Fizz Buzz"

def reverseString(sStr):
    if sStr == "":
        return ""
    else:
        return sStr[len(sStr)-1] + reverseString(sStr[:len(sStr)-1])
    
print(reverseString(sString)) 