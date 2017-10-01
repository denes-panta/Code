# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:52:09 2017

@author: Denes
"""
#Clear all global variables

import IPython
IPython.get_ipython().magic('reset -sf') 

#Class definition

class game:
    gameCount = 0

    def __init__(self, dev, name, p, q, gid, at, pr, sp, cn):
        self.dev = dev
        self.name = name
        self.p = p
        self.id = gid
        self.accept = at
        self.player = pr
        self.spin = sp
        self.coin = cn
        game.gameCount += 1

#Reading from text file

source = open('Casino.txt','r')
content = source.readlines()[1:]

casino = []

for i in range(len(content)):
    casino.append(game('dn', 'gn', 'p', 'q', 0, 0, 0, 0, 0))

for i in range(len(content)):
    casino[i].dev = content[i].split('\t')[0]
    casino[i].name = content[i].split('\t')[1]
    casino[i].p = content[i].split('\t')[2]

#Sort the object list
casino.sort(key = lambda g: (g.dev + g.name), reverse = False)
    
source.close()

#GUI building

import tkinter

app = tkinter.Tk()
app.title("FreeSpin Bonus Cost Estimator")
app.geometry("530x480+50+100")

for i in range(len(content)):
    casino[i].id = 0
    casino[i].accept = 0
    casino[i].player = 0
    casino[i].spin = 0

#Game Selection Form with Checkboxes
def openCheckBoxForm():

    #Close CheckForm & Save Button
    def closeCheckBoxForm():
        global radio
        radio = radioVar.get()

        global byGame
        byGame = totalVar.get()

        global zProbability
        zProbability = probEntry.get()
        
        for i in range(len(content)):
            casino[i].id = 0
            casino[i].accept = 0
            casino[i].player = 0
            casino[i].spin = 0
            casino[i].coin = 0
            
        c = 0
        j1 = j2 = j3 = j4 = 0
        
        for i in range(len(content)):
            if checkList[i].get():
                casino[i].id = 1
                c += 1

                if uacceptEntry['state'] == 'normal':
                    try:
                        int(uacceptEntry.get())
                    except:
                        casino[i].accept = 0
                    else:
                        if (int(uacceptEntry.get()) > 0 and int(uacceptEntry.get()) <= 100 and len(uacceptEntry.get()) != 0):
                            casino[i].accept = uacceptEntry.get()
                            j1 += 1
                else:
                    try:
                        int(acceptList[i].get())
                    except:
                        casino[i].accept = 0
                    else:
                        if (int(acceptList[i].get()) > 0 and int(acceptList[i].get()) <= 100 and len(acceptList[i].get()) != 0):
                            casino[i].accept = acceptList[i].get()
                            j1 += 1

                if uplayerEntry['state'] == 'normal':
                    try:
                        int(uplayerEntry.get())
                    except:
                        casino[i].player = 0
                    else:
                        if (int(uplayerEntry.get()) > 0 and len(uplayerEntry.get()) != 0):
                            casino[i].player = uplayerEntry.get()
                            j2 += 1
                else:
                    try:
                        int(playerList[i].get())
                    except:
                        casino[i].player = 0
                    else:
                        if (int(playerList[i].get()) > 0 and len(playerList[i].get()) != 0):
                            casino[i].player = playerList[i].get()
                            j2 += 1

                if uspinEntry['state'] == 'normal':
                    try:
                        int(uspinEntry.get())
                    except:
                        casino[i].spin = 0
                    else:
                        if (int(uspinEntry.get()) > 0 and len(uspinEntry.get()) != 0):
                            casino[i].spin = uspinEntry.get()
                            j3 += 1
                else:
                    try:
                        int(spinList[i].get())
                    except:
                        casino[i].spin = 0
                    else:
                        if (int(spinList[i].get()) > 0 and int(spinList[i].get()) <= 100 and len(spinList[i].get()) != 0):
                            casino[i].spin = spinList[i].get()
                            j3 += 1
                            
                if ucoinEntry['state'] == 'normal':
                    try:
                        float(ucoinEntry.get())
                    except:
                        casino[i].coin = 0
                    else:
                        if (float(ucoinEntry.get()) > 0 and len(ucoinEntry.get()) != 0):
                            casino[i].coin = ucoinEntry.get()
                            j4 += 1
                else:
                    try:
                        float(coinList[i].get())
                    except:
                        casino[i].coin = 0
                    else:
                        if (float(coinList[i].get()) > 0 and float(coinList[i].get()) <= 100 and len(coinList[i].get()) != 0):
                            casino[i].coin = coinList[i].get()
                            j4 += 1
                try:
                    int(probEntry.get())
                except:
                    zProbability = 0
                else:
                    if (int(probEntry.get()) > 0 and int(probEntry.get()) <= 100 and len(probEntry.get()) != 0):
                        zProbability = probEntry.get()
            else:
                casino[i].id = 0
                        
        if c == j1 == j2 == j3 == j4 and j1 != 0 and j2 != 0 and j3 != 0 and j4 != 0:
            calcButton.config(bg = 'green')
        else:
            calcButton.config(bg = 'red')
            
        checkForm.withdraw()
        app.deiconify()

    #Dimensions
    checkForm = tkinter.Toplevel()
    checkForm.title("Game Selection & Options")
    checkForm.geometry("860x400+50+100")

    app.withdraw()

    #Buttons
    checkFormCloseButton = tkinter.Button(master = checkForm, text = "Save & Close", command = closeCheckBoxForm)
    checkFormCloseButton.place(x = 725, y = 65, width = 80)
    
    #Scrollbar
    scbar = tkinter.Scrollbar(master = checkForm)
    scbar.pack(side = 'right', fill = 'y')
    
    #Labels
    uniformVar = tkinter.StringVar()
    uniformLabel = tkinter.Label(master = checkForm, textvariable = uniformVar)
    uniformVar.set("Uniform:")
    uniformLabel.place(x = 10, y = 6)

    uniformVar = tkinter.StringVar()
    uniformLabel = tkinter.Label(master = checkForm, textvariable = uniformVar)
    uniformVar.set("Number of Players")
    uniformLabel.place(x = 10, y = 50)

    uniformVar = tkinter.StringVar()
    uniformLabel = tkinter.Label(master = checkForm, textvariable = uniformVar)
    uniformVar.set("Number of Spins")
    uniformLabel.place(x = 10, y = 70)
    
    uniformVar = tkinter.StringVar()
    uniformLabel = tkinter.Label(master = checkForm, textvariable = uniformVar)
    uniformVar.set("Acceptance Rate")
    uniformLabel.place(x = 200, y = 70)
    
    uniformVar = tkinter.StringVar()
    uniformLabel = tkinter.Label(master = checkForm, textvariable = uniformVar)
    uniformVar.set("Spin Value")
    uniformLabel.place(x = 200, y = 50)

    probVar = tkinter.StringVar()
    probLabel = tkinter.Label(master = checkForm, textvariable = probVar)
    probVar.set('Probablity')
    probLabel.place(x = 200, y = 30)

    probVar = tkinter.StringVar()
    probLabel = tkinter.Label(master = checkForm, textvariable = probVar)
    probVar.set("(Must be filled out everytime)")
    probLabel.place(x = 350, y = 30)
    
    unifpercVar = tkinter.StringVar()
    unifpercLabel = tkinter.Label(master = checkForm, textvariable = unifpercVar)
    unifpercVar.set('%')
    unifpercLabel.place(x = 332, y = 30)

    unifpercVar = tkinter.StringVar()
    unifpercLabel = tkinter.Label(master = checkForm, textvariable = unifpercVar)
    unifpercVar.set('%')
    unifpercLabel.place(x = 332, y = 70)
    
    devVar = tkinter.StringVar()
    devLabel = tkinter.Label(master = checkForm, textvariable = devVar)
    devVar.set("Developer Name")
    devLabel.place(x = 20, y = 100)
    
    nameVar = tkinter.StringVar()
    nameLabel = tkinter.Label(master = checkForm, textvariable = nameVar)
    nameVar.set("Game Name")
    nameLabel.place(x = 150, y = 100)
    
    pVar = tkinter.StringVar()
    pLabel = tkinter.Label(master = checkForm, textvariable = pVar)
    pVar.set("Chance to Win")
    pLabel.place(x = 255, y = 100)
    
    acceptVar = tkinter.StringVar()
    acceptLabel = tkinter.Label(master = checkForm, textvariable = acceptVar)
    acceptVar.set("Acceptance Rate")
    acceptLabel.place(x = 370, y = 100)

    playerVar = tkinter.StringVar()
    playerLabel = tkinter.Label(master = checkForm, textvariable = playerVar)
    playerVar.set("Number of Players")
    playerLabel.place(x = 490, y = 100)

    spinVar = tkinter.StringVar()
    spinLabel = tkinter.Label(master = checkForm, textvariable = spinVar)
    spinVar.set("Number of Spins")
    spinLabel.place(x = 620, y = 100)

    coinVar = tkinter.StringVar()
    coinLabel = tkinter.Label(master = checkForm, textvariable = coinVar)
    coinVar.set("Spin Value")
    coinLabel.place(x = 740, y = 100)
   
    devNameVar = []
    devNameLabels = []
    
    for i in range(len(content)):
        devNameVar.insert(i, tkinter.StringVar())
        devNameLabels.insert(i, tkinter.Label(master = checkForm, textvariable = devNameVar[i]))
        devNameVar[i].set(casino[i].dev)
        devNameLabels[i].place(x = 20, y = 130 + i*20)

    pVar = []
    pLabels = []
    
    for i in range(len(content)):
        pVar.insert(i, tkinter.StringVar())
        pLabels.insert(i, tkinter.Label(master = checkForm, textvariable = pVar[i]))
        pVar[i].set(str(round(float(casino[i].p)*100, 2)) + ' %')
        pLabels[i].place(x = 280, y = 130 + i*20)

    percentVar = tkinter.StringVar()
    percentLabel = []

    for i in range(len(content)):
        percentLabel.insert(i, tkinter.Label(master = checkForm, textvariable = percentVar))
        percentVar.set('%')
        percentLabel[i].place(x = 425, y = 130 + i*20)

    #Entries
    acceptList = []
    playerList = []
    spinList = []
    coinList = []

    uniformacceptList = []
    uniformplayerList = []
    uniformspinList = []
    uniformcoinList = []

    probEntry = tkinter.Entry(master = checkForm, width = 3)
    probEntry.place(x = 310, y = 31)
    
    uacceptEntry = tkinter.Entry(master = checkForm, width = 3)
    uacceptEntry.place(x = 310, y = 71)
    
    uplayerEntry = tkinter.Entry(master = checkForm, width = 5)
    uplayerEntry.place(x = 125, y = 51)
    
    uspinEntry = tkinter.Entry(master = checkForm, width = 5)
    uspinEntry.place(x = 125, y = 71)
    
    ucoinEntry = tkinter.Entry(master = checkForm, width = 5)
    ucoinEntry.place(x = 310, y = 51)
    
    for i in range(len(content)):            
        acceptList.insert(i, tkinter.Entry(master = checkForm, width = 5))
        acceptList[i].place(x = 390, y = 130 + i*20)
        
        playerList.insert(i, tkinter.Entry(master = checkForm, width = 6))
        playerList[i].place(x = 520, y = 130 + i*20)
        
        spinList.insert(i, tkinter.Entry(master = checkForm, width = 4))
        spinList[i].place(x = 650, y = 130 + i*20)
        
        coinList.insert(i, tkinter.Entry(master = checkForm, width = 4))
        coinList[i].place(x = 755, y = 130 + i*20)
        
    for i in range(len(content)):
       uniformacceptList.insert(i, int(casino[i].accept))
       uniformplayerList.insert(i, int(casino[i].player))
       uniformspinList.insert(i, int(casino[i].spin))
       uniformcoinList.insert(i, float(casino[i].coin))

    try:
        if probEntry.get() != 0:
            probEntry.insert(0, zProbability)
    except:
        pass
    
    try:
        if radio == 2:
            if max(set(uniformacceptList)) != 0:
                uacceptEntry.insert(0, max(set(uniformacceptList)))

            if max(set(uniformplayerList)) != 0:
                uplayerEntry.insert(0, max(set(uniformplayerList)))

            if max(set(uniformspinList)) != 0:    
                uspinEntry.insert(0, max(set(uniformspinList)))

            if max(set(uniformcoinList)) != 0:    
                ucoinEntry.insert(0, max(set(uniformcoinList)))
       
        elif radio == 1:
            for i in range(len(content)):            
                if casino[i].accept != 0:
                    acceptList[i].insert(i, casino[i].accept)

                if casino[i].player != 0:
                    playerList[i].insert(i, casino[i].player)
            
                if casino[i].spin != 0:
                    spinList[i].insert(i, casino[i].spin)
                    
                if casino[i].coin != 0:
                    coinList[i].insert(i, casino[i].coin)
    except:
        pass

    for i in range(len(content)):
        try:
            if casino[i].id == 1 and radio == 1:
                if len(acceptList[i].get()) == 0:
                    acceptList[i].config(bg = 'red')
                    
                if len(playerList[i].get()) == 0:
                    playerList[i].config(bg = 'red')
                    
                if len(spinList[i].get()) == 0:
                    spinList[i].config(bg = 'red')
                    
                if len(coinList[i].get()) == 0:
                    coinList[i].config(bg = 'red')                    
        except:
            pass
        
    try:
        if radio == 2 and len(uacceptEntry.get()) == 0:
            uacceptEntry.config(bg = 'red')

        if radio == 2 and len(uplayerEntry.get()) == 0:
            uplayerEntry.config(bg = 'red')

        if radio == 2 and len(uspinEntry.get()) == 0:
            uspinEntry.config(bg = 'red')
                
        if radio == 2 and len(ucoinEntry.get()) == 0:
            ucoinEntry.config(bg = 'red')
            
        if len(probEntry.get()) == 0:
            probEntry.config(bg = 'red')
    except:
        pass
    
    #CheckBoxes
    checkList = []
    checkButton = []

    for i in range(len(content)):
        checkList.insert(i, tkinter.IntVar())
        checkButton.insert(i, tkinter.Checkbutton(master = checkForm, text = casino[i].name, variable = checkList[i]))
        checkButton[i].place(x = 130, y = 129 + i*20)    
        if casino[i].id == 1:
            checkButton[i].select()

    totalVar = tkinter.IntVar()
    totalCheck = tkinter.Checkbutton(master = checkForm, text = "By Casino Game", variable = totalVar)
    totalCheck.place(x = 9, y = 28)

    try:
        if byGame == 1:
            totalCheck.select()
    except:
        pass
    
    #RadioButtons
    def individualRadio():
        uacceptEntry.config(state = "disabled")
        uplayerEntry.config(state = "disabled")
        uspinEntry.config(state = "disabled")
        ucoinEntry.config(state = "disabled")
        totalCheck.config(state = "disabled")
        
        for i in range(len(content)):
            acceptList[i].config(state = "normal")
            playerList[i].config(state = "normal")
            spinList[i].config(state = "normal")
            coinList[i].config(state = "normal")
        
    def uniformRadio():
        uacceptEntry.config(state = "normal")
        uplayerEntry.config(state = "normal")
        uspinEntry.config(state = "normal")
        ucoinEntry.config(state = "normal")
        totalCheck.config(state = "normal")
        
        for i in range(len(content)):
            acceptList[i].config(state = "disabled")
            playerList[i].config(state = "disabled")
            spinList[i].config(state = "disabled")
            coinList[i].config(state = "disabled")

    radioVar = tkinter.IntVar()
    
    unifindR1 = tkinter.Radiobutton(master = checkForm, text = 'Individual', variable = radioVar, value = 1, command = individualRadio)
    unifindR1.place(x = 725, y = 10)

    unifindR2 = tkinter.Radiobutton(master = checkForm, text = 'Uniform', variable = radioVar, value = 2, command = uniformRadio)
    unifindR2.place(x = 725, y = 30)

    try:
        if radio == 1:
            unifindR1.select()
            unifindR1.invoke()
        elif radio == 2:
            unifindR2.select()
            unifindR2.invoke()
    except:
            unifindR2.select()
            unifindR2.invoke()
    
    checkForm.mainloop()

#Calculations
import numpy as np
import scipy.stats as stats
#from tkinter import messagebox as msgbox

def calculateCost():
    result.delete(1.0, "end")
    exportButton["state"] = "normal"
    
    if calcButton["bg"] == 'red':
        tkinter.messagebox.showwarning(title = "Calculation Error", message = "Please select at least one game from the Options")
    else:
        costMinTotal = 0
        costAvgTotal = 0
        costMaxTotal = 0
        
        j = 0
      
        while casino[j].id != 1:
            j += 1
        
        gameCostMin = int(casino[j].player) * int(casino[j].spin)
        gameCostMax = 0
        
        casino.sort(key = lambda g: (g.dev+g.name), reverse = True)
        
        for i in range(len(content)):
            accept = int(casino[i].accept)
            player = int(casino[i].player)
            spin = int(casino[i].spin)
            coin = float(casino[i].coin)
            p = float(casino[i].p)
            prob = float(zProbability)
            
            t = round(stats.t.ppf(1-(1-prob/100)/2, 10000))
            
            binom_sim = stats.binom.rvs(n = player * spin, p = p, size = 100000)
            binom_mean = np.mean(binom_sim) * coin * accept
            binom_std = np.std(binom_sim) * coin * accept
            
            if round(binom_mean - binom_std * t, 2) < 0:
                costMin = 0
            else:
                costMin = round(binom_mean - binom_std * t, 2)
            
            costAvg = round(binom_mean, 2)
            costMax = round(binom_mean + binom_std * t, 2)
            costMinTotal += round(binom_mean - binom_std * t, 2)
            costAvgTotal += round(binom_mean, 2)
            costMaxTotal += round(binom_mean + binom_std * t, 2)
            
            if byGame == 0 and casino[i].id == 1 and costMax > gameCostMax:
                gameCostMax = costMax
            
            if byGame == 0 and casino[i].id == 1 and costMin < gameCostMin:
                gameCostMin = costMin
            
            if byGame == 1 and casino[i].id == 1:               
                result.insert(1.0, casino[i].dev + ' - ' + casino[i].name + "\n")
                result.insert(2.0, "Cost Min:" + "\t" + str(costMin) + "\n")
                result.insert(3.0, "Cost Mean:" + "\t" + str(costAvg) + "\n")
                result.insert(4.0, "Cost Max:" + "\t" + str(costMax) + "\n")
                result.insert(5.0, "\n")
                
        if byGame == 1:    
            result.insert(1.0, "Total" + ' - ' + "Bonus Cost" + "\n")
            result.insert(2.0, "Total Min:" + "\t" + str(costMinTotal) + "\n")
            result.insert(3.0, "Total Mean:" + "\t" + str(costAvgTotal) + "\n")
            result.insert(4.0, "Total Max:" + "\t" + str(costMaxTotal) + "\n")
            result.insert(5.0, "\n")
            exportButton["state"] = "normal"
            
        elif byGame == 0:
            result.insert(1.0, "Total" + ' - ' + "Bonus Cost" + "\n")
            result.insert(2.0, "Total Min:" + "\t" + str(gameCostMin) + "\n")
            result.insert(3.0, "Total Mean:" + "\t" + str((gameCostMin + gameCostMax)/2) + "\n")
            result.insert(4.0, "Total Max:" + "\t" + str(gameCostMax) + "\n")
            result.insert(5.0, "\n")
            
#Exit
def closeApplication(): 
    app.destroy()
 
#Help
def helpDisplay(): 
    helpForm = tkinter.Toplevel()
    helpForm.title("Help")
    helpForm.geometry("850x480+50+100")
    
    def closehelpForm():
        helpForm.destroy()
        
    #Help text
    helpText = tkinter.Text(master = helpForm, width = 90, height = 17, spacing3 = 10)
    helpText.place(x = 100, y = 10)
    helpText.insert(1.0, "Number of Players: The number of players the Bonus is issued.\n")
    helpText.insert(2.0, "Number of Spins: The number of spins issued to 1 player.\n")
    helpText.insert(3.0, "Spin Value: The amount of money issued for 1 spin.\n")
    helpText.insert(4.0, "Acceptance Rate: (Accepted Bonus / Issued Bonus) * 100\n")
    helpText.insert(5.0, "Probability: The probablity that the Cost falls between Max Cost and Min Cost.\n")
    helpText.insert(6.0, "             The higher it is, the bigger the difference between Max and Min Cost.\n")
    helpText.insert(7.0, "By Game: If ticked, the issued bonus is times each game.\n")
    helpText.insert(8.0, "Individual/Uniform: Whether the bonus needs to be issued for each game individually.\n")
    helpText.insert(9.0, "\n")
    helpText.insert(10.0, "How to use:\n")
    helpText.insert(11.0, "1. Click on Options, and set the Variables on the new window.\n")
    helpText.insert(12.0, "2. If everything is set, the Calculate button turns green after saving the Variables.\n")
    helpText.insert(13.0, "3. Click Calculate, and the results will apper.\n")
    helpText.insert(14.0, "4. Export the information to a text file, which can be imported into excel. (Tabulated)\n")
    helpText.insert(15.0, "\n")
    helpText.insert(16.0, "\n")
    helpText.insert(17.0, "Created by Denes Panta - Gibraltar, June 2017. (636 Lines Total)")

    #Button
    closeHelpButton = tkinter.Button(master = helpForm, text = "Close", command = closehelpForm)
    closeHelpButton.place(x = 10, y = 10, width = 80)

#Export
def exportResults():
    file = open("bonus_cost.txt", "w") 
 
    file.write(result.get(1.0, "end"))
 
    file.close()

#Text
result = tkinter.Text(master = app, width = 40)
result.place(x = 100, y = 10)

#Buttons
checkButton = tkinter.Button(master = app, text = "Options", command = openCheckBoxForm)
checkButton.place(x = 10, y = 10, width = 80)

calcButton = tkinter.Button(master = app, text = "Calculate", command = calculateCost, bg = 'red')
calcButton.place(x = 10, y = 40, width = 80)

helpButton = tkinter.Button(master = app, text = "Help", command = helpDisplay)
helpButton.place(x = 10, y = 70, width = 80)

exportButton = tkinter.Button(master = app, text = "Export", command = exportResults, state = "disabled")
exportButton.place(x = 10, y = 100, width = 80)

closeButton = tkinter.Button(master = app, text = "Exit", command = closeApplication)
closeButton.place(x = 10, y = 130, width = 80)

app.mainloop()