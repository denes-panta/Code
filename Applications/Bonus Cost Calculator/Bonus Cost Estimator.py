import numpy as np
import pandas as pd
import tkinter
from scipy import stats
from tkinter import messagebox

class estimator(object):
    def __init__(self, location):
        
        #Reading from text file 
        source = open(location,'r')
        content = source.readlines()[1:]
        
        #Create the DataFrames for data storage
        self.input_data = pd.DataFrame(index = range(len(content)), 
                                       columns = ['dev_name', 
                                                  'game_name', 
                                                  'p_value', 
                                                  'id'])
        
        self.options_data = pd.DataFrame(index = range(len(content)), 
                                         columns = ['accept_rate', 
                                                    'player', 
                                                    'spin', 
                                                    'coin'])
        
        self.options_indv = pd.DataFrame(index = range(1),
                                         columns = ['accept_rate', 
                                                    'player', 
                                                    'spin', 
                                                    'coin'])
        
        #Populate the dataframes with input data and set the options data to ''
        for i, line in enumerate(content):
            self.input_data.loc[i, 'dev_name'] = str(line.split('\t')[0])
            self.input_data.loc[i, 'game_name'] = str(line.split('\t')[1])
            self.input_data.loc[i, 'p_value'] = float(line.split('\t')[2].rstrip())
            self.input_data.loc[i, 'id'] = 0
            
            #Set Acceptance rate to 100% 
            self.options_data.loc[i, 'accept_rate'] = '100' 
            self.options_data.loc[i, 'player'] = ''
            self.options_data.loc[i, 'spin'] = ''
            self.options_data.loc[i, 'coin'] = ''
        
        #Populate the uniform dataframe
        for i in range(self.options_indv.shape[1]):    
            self.options_indv.iloc[0, i] = ''
        #Set Acceptance rate to 100%
        self.options_indv.iloc[0, 0] = 100 
        
        #Sort the list of data
        self.input_data.sort_values(['dev_name', 'game_name'], 
                                    ascending = [True, True])
        
        #Number of games in the Text file
        self.n = len(content)
    
        #Confidence probability
        self.options_confidence = 0.95
        
        #Radio button default value
        self.options_radio = 1

        #Radio byGame variable
        self.options_total = 0
        
        #Calculation matrix
        self.calc_matrix = 0
        
        #Close the source files
        source.close()
        
        #Display the GUI
        self.application = tkinter.Tk()
        self.application.title("FreeSpin Bonus Cost Estimator")
        self.application.geometry("530x480+50+100")

        #Text
        self.result = tkinter.Text(master = self.application, width = 40)
        self.result.place(x = 100, y = 10)
        
        #Buttons
        #Options button
        app_bn_options = tkinter.Button(master = self.application, 
                                        text = "Options", 
                                        command = self.openOptionsForm)
        app_bn_options.place(x = 10, y = 10, width = 80)
        
        #Calculation button
        self.app_bn_calculate = tkinter.Button(master = self.application, 
                                               text = "Calculate", 
                                               command = self.calculateCost, 
                                               bg = 'red')
        self.app_bn_calculate.place(x = 10, y = 40, width = 80)
        
        #Help button
        app_bn_help = tkinter.Button(master = self.application, 
                                     text = "Help", 
                                     command = self.displayHelp)
        app_bn_help.place(x = 10, y = 70, width = 80)
        
        #Export button
        path = "F:/Code/Bonus Cost/bonus_cost.txt"
        self.app_bn_export = tkinter.Button(master = self.application, 
                                       text = "Export", 
                                       command = lambda: self.exportResults(path), 
                                       state = "disabled")
        self.app_bn_export.place(x = 10, y = 100, width = 80)
        
        #Close button
        app_bn_close = tkinter.Button(master = self.application, 
                                      text = "Exit",
                                      command = self.closeApplication)
        app_bn_close.place(x = 10, y = 130, width = 80)

        self.application.mainloop()


    def openOptionsForm(self):
        #Form
        self.application.withdraw()

        self.optionsForm = tkinter.Toplevel()
        self.optionsForm.title("Game Selection & Options")
        self.optionsForm.geometry("860x400+50+100")
        
        #Buttons
        #Save Button
        opt_bn_close = tkinter.Button(master = self.optionsForm, 
                                      text = "Save & Close", 
                                      command = self.closeoptionsForm)
        opt_bn_close.place(x = 725, y = 35, width = 80)
        
        #Cancel button
        opt_bn_cancel = tkinter.Button(master = self.optionsForm, 
                                      text = "Cancel", 
                                      command = self.cancelOptionsForm)
        opt_bn_cancel.place(x = 725, y = 65, width = 80)

        #Scrollbar
        opt_scbar = tkinter.Scrollbar(master = self.optionsForm)
        opt_scbar.pack(side = 'right', fill = 'y')
        
        #Labels
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Uniform:")
        opt_ll_info.place(x = 10, y = 6)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Number of Players")
        opt_ll_info.place(x = 10, y = 50)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Number of Spins")
        opt_ll_info.place(x = 10, y = 70)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Acceptance Rate")
        opt_ll_info.place(x = 200, y = 70)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)    
        opt_ll_var.set("Spin Value")
        opt_ll_info.place(x = 200, y = 50)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set('Probablity')
        opt_ll_info.place(x = 200, y = 30)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("(Must be filled in everytime)")
        opt_ll_info.place(x = 360, y = 30)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set('%')
        opt_ll_info.place(x = 342, y = 30)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set('%')
        opt_ll_info.place(x = 332, y = 70)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Developer Name")
        opt_ll_info.place(x = 20, y = 100)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Game Name")
        opt_ll_info.place(x = 150, y = 100)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Chance to Win")
        opt_ll_info.place(x = 255, y = 100)
        
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Acceptance Rate")
        opt_ll_info.place(x = 370, y = 100)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Number of Players")
        opt_ll_info.place(x = 490, y = 100)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Number of Spins")
        opt_ll_info.place(x = 620, y = 100)
    
        opt_ll_var = tkinter.StringVar()
        opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                    textvariable = opt_ll_var)
        opt_ll_var.set("Spin Value")
        opt_ll_info.place(x = 740, y = 100)
       
        for i in range(self.n):
            opt_ll_var = tkinter.StringVar()
            opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                        textvariable = opt_ll_var)
            opt_ll_var.set(self.input_data.loc[i, 'dev_name'])
            opt_ll_info.place(x = 20, y = 130 + i*20)

            opt_ll_var = tkinter.StringVar()
            opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                        textvariable = opt_ll_var)
            opt_ll_var.set(str(round(float(
                    self.input_data.loc[i, 'p_value'])*100, 2)) + ' %')
            opt_ll_info.place(x = 280, y = 130 + i*20)

            opt_ll_var = tkinter.StringVar()
            opt_ll_info = tkinter.Label(master = self.optionsForm, 
                                        textvariable = opt_ll_var)
            opt_ll_var.set('%')
            opt_ll_info.place(x = 425, y = 130 + i*20)
        
        #Entries - Confidence
        self.optionsForm.opt_ey_confidence = tkinter.Entry(
                master = self.optionsForm, width = 5)
        self.optionsForm.opt_ey_confidence.place(
                x = 310, y = 31)
        self.optionsForm.opt_ey_confidence.insert(
                0, str(np.round(self.options_confidence * 100, 2)))
        
        #Entries - Uniform
        self.optionsForm.opt_ey_accept = tkinter.Entry(
                master = self.optionsForm, width = 3)
        self.optionsForm.opt_ey_accept.place(
                x = 310, y = 71)
        self.optionsForm.opt_ey_accept.insert(
                0, str(self.options_indv.iloc[0, 0]))
        
        self.optionsForm.opt_ey_player = tkinter.Entry(
                master = self.optionsForm, width = 5)
        self.optionsForm.opt_ey_player.place(
                x = 125, y = 51)
        self.optionsForm.opt_ey_player.insert(
                0, str(self.options_indv.iloc[0, 1]))
        
        self.optionsForm.opt_ey_spin = tkinter.Entry(
                master = self.optionsForm, width = 5)
        self.optionsForm.opt_ey_spin.place(
                x = 125, y = 71)
        self.optionsForm.opt_ey_spin.insert(
                0, str(self.options_indv.iloc[0, 2]))
        
        self.optionsForm.opt_ey_coin = tkinter.Entry(
                master = self.optionsForm, width = 5)
        self.optionsForm.opt_ey_coin.place(
                x = 310, y = 51)
        self.optionsForm.opt_ey_coin.insert(
                0, str(self.options_indv.iloc[0, 3]))

        #Entries - Individual
        self.optionsForm.optionsEntries = []

        for i in range(self.n):
            self.optionsForm.optionsEntries.append([])
        
        #Entries for each game
        for i in range(self.n):
            #Acceptance rate            
            self.optionsForm.optionsEntries[0].append(
                    tkinter.Entry(master = self.optionsForm, width = 5))
            self.optionsForm.optionsEntries[0][i].place(
                    x = 390, y = 130 + i*20)
            self.optionsForm.optionsEntries[0][i].insert(
                    0, self.options_data.loc[i, 'accept_rate']) 
            
            #Nunber of players
            self.optionsForm.optionsEntries[1].append(
                    tkinter.Entry(master = self.optionsForm, width = 6))
            self.optionsForm.optionsEntries[1][i].place(
                    x = 520, y = 130 + i*20)
            self.optionsForm.optionsEntries[1][i].insert(
                    0, self.options_data.loc[i, 'player']) 
            
            #Number of Spins
            self.optionsForm.optionsEntries[2].append(
                    tkinter.Entry(master = self.optionsForm, width = 4))
            self.optionsForm.optionsEntries[2][i].place(
                    x = 650, y = 130 + i*20)
            self.optionsForm.optionsEntries[2][i].insert(
                    0, self.options_data.loc[i, 'spin']) 
            
            #Money            
            self.optionsForm.optionsEntries[3].append(
                    tkinter.Entry(master = self.optionsForm, width = 4))
            self.optionsForm.optionsEntries[3][i].place(
                    x = 755, y = 130 + i*20)
            self.optionsForm.optionsEntries[3][i].insert(
                    0, self.options_data.loc[i, 'coin'])

        #CheckBoxes
        
        #Lists for the CheckBox parts (Variable & Button)
        self.optionsForm.optionsCheckBoxes = [[], [], []] 
        
        #[0] = Integer Variable
        #[1] = CheckBox Buttons
        #[2] = Checkbox Variable & button for 'Total' checkbox
        for i in range(self.n):
            self.optionsForm.optionsCheckBoxes[0].append(
                    tkinter.IntVar(value = self.input_data.loc[i, 'id']))
            self.optionsForm.optionsCheckBoxes[1].append(
                    tkinter.Checkbutton(master = self.optionsForm, 
                                        text = self.input_data.loc[i, 'game_name'], 
                                        variable = \
                                        self.optionsForm.optionsCheckBoxes[0][i]))
            self.optionsForm.optionsCheckBoxes[1][i].place(
                    x = 130, y = 129 + i*20)    
            
        #The Uniform values count as total, or the values are for each game individually
        self.optionsForm.optionsCheckBoxes[2].append(
                tkinter.IntVar(value = self.options_total))
        self.optionsForm.optionsCheckBoxes[2].append(
                tkinter.Checkbutton(master = self.optionsForm, 
                                    text = "Total", 
                                    variable = \
                                    self.optionsForm.optionsCheckBoxes[2][0]))
        self.optionsForm.optionsCheckBoxes[2][1].place(
                x = 9, y = 27)
    
        #RadioButtons
        
        #Disable the options related to uniform calculations and enable the individual ones
        def uniformRadio(): 
            for i in range(self.n):
                self.optionsForm.optionsEntries[0][i].config(state = "disabled")
                self.optionsForm.optionsEntries[1][i].config(state = "disabled")
                self.optionsForm.optionsEntries[2][i].config(state = "disabled")
                self.optionsForm.optionsEntries[3][i].config(state = "disabled")            
            
                self.optionsForm.opt_ey_accept.config(state = "normal")
                self.optionsForm.opt_ey_player.config(state = "normal")
                self.optionsForm.opt_ey_spin.config(state = "normal")
                self.optionsForm.opt_ey_coin.config(state = "normal")
                self.optionsForm.optionsCheckBoxes[2][1].config(state = "normal")
        
        #Disable the options related to individual calculations and enable the uniform ones        
        def individualRadio(): 
            for i in range(self.n):
                self.optionsForm.optionsEntries[0][i].config(state = "normal")
                self.optionsForm.optionsEntries[1][i].config(state = "normal")
                self.optionsForm.optionsEntries[2][i].config(state = "normal")
                self.optionsForm.optionsEntries[3][i].config(state = "normal")
            
                self.optionsForm.opt_ey_accept.config(state = "disabled")
                self.optionsForm.opt_ey_player.config(state = "disabled")
                self.optionsForm.opt_ey_spin.config(state = "disabled")
                self.optionsForm.opt_ey_coin.config(state = "disabled")
                self.optionsForm.optionsCheckBoxes[2][1].config(state = "disabled")
                
        #Put up the Radio Buttons
        self.optionsForm.RadioButtons = []
        self.optionsForm.RadioButtons.append(tkinter.IntVar(value = \
                                                            self.options_radio))
        
        #Individual
        self.optionsForm.RadioButtons.append(
                tkinter.Radiobutton(master = self.optionsForm, 
                                    text = 'Individual', 
                                    variable = self.optionsForm.RadioButtons[0], 
                                    value = 1, 
                                    command = individualRadio))
        self.optionsForm.RadioButtons[1].place(x = 360, y = 50)
        
        #Uniform
        self.optionsForm.RadioButtons.append(
                tkinter.Radiobutton(master = self.optionsForm, 
                                    text = 'Uniform', 
                                    variable = self.optionsForm.RadioButtons[0], 
                                    value = 2, 
                                    command = uniformRadio))
        
        self.optionsForm.RadioButtons[2].place(x = 360, y = 70)

        self.optionsForm.RadioButtons[self.options_radio].invoke()
        
        self.optionsForm.mainloop()
        
        
    #Save & Close the optionsForm
    def closeoptionsForm(self):
        error = False
        
        #Get Option Values
        self.options_radio = self.optionsForm.RadioButtons[0].get()
        self.options_total = self.optionsForm.optionsCheckBoxes[2][0].get()
        
        #Error handling & Saving user defined values
        
        #Confidence interval
        try:
            if (float(self.optionsForm.opt_ey_confidence.get()) > 0) and \
            (float(self.optionsForm.opt_ey_confidence.get()) <= 100):
                self.options_confidence = np.round(float(
                        self.optionsForm.opt_ey_confidence.get())/100, 2)
                self.optionsForm.opt_ey_confidence.config(bg = 'white')
            else:
                self.optionsForm.opt_ey_confidence.config(bg = 'red')
                error = True  
        except:
            self.optionsForm.opt_ey_confidence.config(bg = 'red')
            error = True
            
        games = 0 #Count the number of games selected

        #Individual
        if self.optionsForm.RadioButtons[0].get() == 1:
            for i in range(self.n):
                #Number of games selected
                self.input_data.loc[i, 'id'] = \
                self.optionsForm.optionsCheckBoxes[0][i].get()
                games += self.optionsForm.optionsCheckBoxes[0][i].get()
                
                #Acceptance Rate
                try:
                    if (int(self.optionsForm.optionsEntries[0][i].get()) > 0) and \
                    (int(self.optionsForm.optionsEntries[0][i].get()) <= 100):
                        self.options_data.loc[i, 'accept_rate'] = \
                        str(self.optionsForm.optionsEntries[0][i].get())
                        self.optionsForm.optionsEntries[0][i].config(bg = 'white')
                    else:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[0][i].config(bg = 'red')
                except:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[0][i].config(bg = 'red')
                
                #Number of players te bonus is issued to
                try:
                    if int(self.optionsForm.optionsEntries[1][i].get()) > 0:
                       self.options_data.loc[i, 'player'] = \
                       str(self.optionsForm.optionsEntries[1][i].get())
                       self.optionsForm.optionsEntries[1][i].config(bg = 'white')
                    else:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[1][i].config(bg = 'red')
                except:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[1][i].config(bg = 'red')
                
                #Number of Spins issued
                try:
                    if int(self.optionsForm.optionsEntries[2][i].get()) > 0:
                       self.options_data.loc[i, 'spin'] = \
                       str(self.optionsForm.optionsEntries[2][i].get())
                       self.optionsForm.optionsEntries[2][i].config(bg = 'white')
                    else:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[2][i].config(bg = 'red')
                except:
                    if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                        error = True
                        self.optionsForm.optionsEntries[2][i].config(bg = 'red')
                
                #Value of 1 spin
                try:
                    if float(self.optionsForm.optionsEntries[3][i].get()) > 0:
                       self.options_data.loc[i, 'coin'] = \
                       str(self.optionsForm.optionsEntries[3][i].get())
                       self.optionsForm.optionsEntries[3][i].config(bg = 'white')
                    else:
                        if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                            error = True
                            self.optionsForm.optionsEntries[3][i].config(bg = 'red')
                except:
                    if self.optionsForm.optionsCheckBoxes[0][i].get() == 1:
                        error = True
                        self.optionsForm.optionsEntries[3][i].config(bg = 'red')
                    
            if games == 0: error = True
       
        #Uniform
        elif self.optionsForm.RadioButtons[0].get() == 2:
            
            #Number of games selection
            for i in range(self.n):
                self.input_data.loc[i, 'id'] = \
                self.optionsForm.optionsCheckBoxes[0][i].get()
                games += self.optionsForm.optionsCheckBoxes[0][i].get()
            
            if games == 0: error = True
            
            #Uniform Acceptance Rate
            try:
                if (int(self.optionsForm.opt_ey_accept.get()) > 0) and \
                (int(self.optionsForm.opt_ey_accept.get()) <= 100):
                    self.options_indv.iloc[0, 0] = \
                    str(self.optionsForm.opt_ey_accept.get())
                    self.optionsForm.opt_ey_accept.config(bg = 'white')
                else:
                    error = True
                    self.optionsForm.opt_ey_accept.config(bg = 'red')
            except:
                error = True
                self.optionsForm.opt_ey_accept.config(bg = 'red')
            
            #Uniform number of players
            try:                
                if int(self.optionsForm.opt_ey_player.get()) > 0:
                   self.options_indv.iloc[0, 1] = \
                   str(self.optionsForm.opt_ey_player.get())
                   self.optionsForm.opt_ey_player.config(bg = 'white')
                else:
                    error = True
                    self.optionsForm.opt_ey_player.config(bg = 'red')
            except:
                error = True
                self.optionsForm.opt_ey_player.config(bg = 'red')
            
            #Uniform Number of spins
            try:
                if int(self.optionsForm.opt_ey_spin.get()) > 0:
                   self.options_indv.iloc[0, 2] = \
                   str(self.optionsForm.opt_ey_spin.get())
                   self.optionsForm.opt_ey_spin.config(bg = 'white')
                else:
                    error = True
                    self.optionsForm.opt_ey_spin.config(bg = 'red')
            except:
                error = True
                self.optionsForm.opt_ey_spin.config(bg = 'red')
            
            #Value of 1 spin
            try:        
                if float(self.optionsForm.opt_ey_coin.get()) > 0:
                   self.options_indv.iloc[0, 3] = \
                   str(self.optionsForm.opt_ey_coin.get())
                   self.optionsForm.opt_ey_coin.config(bg = 'white')
                else:
                    error = True
                    self.optionsForm.opt_ey_coin.config(bg = 'red')
            except:
                error = True
                self.optionsForm.opt_ey_coin.config(bg = 'red')            

        #Give feedback on errors
        if error == False:
            self.app_bn_calculate.config(bg = 'green')
            self.optionsForm.destroy()
            self.application.deiconify()
        else:
            self.error()


    #Close options without checking for error or saving anything
    def cancelOptionsForm(self):
            self.optionsForm.destroy()
            self.application.deiconify()


    #Display Error Message
    def error(self):
        messagebox.showwarning("Error Message", 
                               "Please double-check your values.")


    #Calculate the costs
    def calculateCost(self):
        #Warning to set the varuibles
        if self.app_bn_calculate["bg"] == 'red': 
            messagebox.showwarning("Calculation Error", 
                                   "Please select at least one game from the Options")
        else:
            self.result.delete(1.0, "end") #Clear the Results
            
            #Define the variable for the Totals
            costMinTotal = 0
            costAvgTotal = 0
            costMaxTotal = 0
            
            #Dataframe for the calculation data          
            df_calc = pd.DataFrame(index = range(self.n),  
                                          columns = ['id', 
                                                     'accept_rate', 
                                                     'player', 
                                                     'spin', 
                                                     'coin', 
                                                     'p_value'])
            
            #Wheater the game is selected or not
            df_calc.iloc[:, 0] = self.input_data.loc[:, 'id'] 
            
            if self.options_radio == 1: #Individual/Uniform by Game
                df_calc.iloc[:, 1] = self.options_data.loc[:, 'accept_rate']
                df_calc.iloc[:, 2] = self.options_data.loc[:, 'player']
                df_calc.iloc[:, 3] = self.options_data.loc[:, 'spin']
                df_calc.iloc[:, 4] = self.options_data.loc[:, 'coin']
                df_calc.iloc[:, 5] = self.input_data.loc[:, 'p_value']
            
            elif self.options_radio == 2: #Uniform Total
                df_calc.iloc[:, 1] = self.options_indv.loc[0, 'accept_rate']
                df_calc.iloc[:, 2] = self.options_indv.loc[0, 'player']
                df_calc.iloc[:, 3] = self.options_indv.loc[0, 'spin']
                df_calc.iloc[:, 4] = self.options_indv.loc[0, 'coin']
                df_calc.iloc[:, 5] = self.input_data.loc[:, 'p_value']
            
            #Clear the empty placeholder strings
            df_calc[df_calc[:] == ''] = 0
            df_calc = df_calc.astype(float) 
            
            df_calc.loc[:, 'accept_rate'] = df_calc.loc[:, 'accept_rate']/100
            
            #Set to 0 all non selected games
            df_calc.loc[df_calc.loc[:, 'id'] == 0, :] = 0
            
            #Calculate the acceptad spins by players
            df_calc.loc[:, 'acc_pl_sp'] = df_calc.loc[:, 'accept_rate'] * \
                                          df_calc.loc[:, 'player'] * \
                                          df_calc.loc[:, 'spin']
            
            #Calculate the t values for each game
            df_calc.loc[:, 't'] = np.round(stats.t.ppf(
                                            1-(1-self.options_confidence)/2, 
                                            df_calc.loc[:, 'acc_pl_sp']), 2)
            
            #Calculate std, mean, min, max of each game
            for game in range(self.n):
                df_calc.loc[game, 'std'] = stats.binom.std(
                        n = int(df_calc.loc[game, 'acc_pl_sp']), 
                        p = df_calc.loc[game, 'p_value'])
                
                df_calc.loc[game, 'mean'] = stats.binom.mean(
                        n = int(df_calc.loc[game, 'acc_pl_sp']), 
                        p = df_calc.loc[game, 'p_value'])
                
                df_calc.loc[game, 'min'] = df_calc.loc[game, 'mean'] - \
                                           df_calc.loc[game, 't'] * \
                                           df_calc.loc[game, 'std']
                
                df_calc.loc[game, 'max'] = df_calc.loc[game, 'mean'] + \
                                           df_calc.loc[game, 't'] * \
                                           df_calc.loc[game, 'std']
            
                df_calc.loc[game, 'min'] = df_calc.loc[game, 'min'] * \
                                           df_calc.loc[game, 'coin']
                                           
                df_calc.loc[game, 'mean'] = df_calc.loc[game, 'mean'] * \
                                            df_calc.loc[game, 'coin']
                                            
                df_calc.loc[game, 'max'] = df_calc.loc[game, 'max'] * \
                                           df_calc.loc[game, 'coin']
            
            #Set all 0 values to NaN
            df_calc[df_calc.loc[:, :] == 0.0] = np.NaN
            #Set all negative values to 0
            df_calc[df_calc.loc[:, :] < 0.0] = 0
            
            #Save the calculation matrix, in case it is needed           
            self.calc_matrix = df_calc 
            
            #Print out the calculation results
            if self.options_total == 0:  #Individual/Uniform by Game

                costMinTotal = int(np.nansum(df_calc.loc[:, 'min']))
                costMaxTotal = int(np.nansum(df_calc.loc[:, 'max']))
                costAvgTotal = int(np.nansum(df_calc.loc[:, 'mean']))
                
                #Print out the Totals
                self.result.insert(1.0, "Total Estimated Cost:" + "\n")
                self.result.insert(2.0, "Cost Max:" + "\t" + \
                                   str(costMaxTotal) + "\n")
                self.result.insert(3.0, "Cost Mean:" + "\t" + \
                                   str(costAvgTotal) + "\n")
                self.result.insert(4.0, "Cost Min:" + "\t" + \
                                   str(costMinTotal) + "\n")
                self.result.insert(5.0, "\n")       
                
                row = 1 #Variable for printing out the result
                
                #Print out the subtotals for each game
                for game in range(self.n):
                    if df_calc.loc[game, 'id'] == 1:
                        self.result.insert(6.0 * row, 
                                           self.input_data.loc[game, 'dev_name'] + \
                                           " - " + \
                                           self.input_data.loc[game, 'game_name'] + \
                                           "\n")
                        self.result.insert(7.0 * row, "Cost Max:" + \
                                           "\t" + \
                                           str(int(df_calc.loc[game, 'max'])) + \
                                           "\n")
                        self.result.insert(8.0 * row, "Cost Mean:" + \
                                           "\t" + \
                                           str(int(df_calc.loc[game, 'mean'])) + \
                                           "\n")
                        self.result.insert(9.0 * row, \
                                           "Cost Min:" + \
                                           "\t" + \
                                           str(int(df_calc.loc[game, 'min'])) + \
                                           "\n")
                        self.result.insert(10.0 * row, "\n")       
                        row += 1
                        
            elif self.options_total == 1: #Uniform Total
                #Calculate the min, max, mean
                costMinTotal = int(np.nanmin(df_calc.loc[:, 'min']))
                costMaxTotal = int(np.nanmax(df_calc.loc[:, 'max']))
                costAvgTotal = int((costMaxTotal + costMinTotal)/2)
                #Print out the Totals
                self.result.insert(1.0, "Total Estimated Cost:" + "\n")
                self.result.insert(2.0, "Cost Min:" + "\t" + \
                                   str(costMinTotal) + "\n")
                self.result.insert(3.0, "Cost Mean:" + "\t" + \
                                   str(costAvgTotal) + "\n")
                self.result.insert(4.0, "Cost Max:" + "\t" + \
                                   str(costMaxTotal) + "\n")
                self.result.insert(5.0, "\n")                
            
            #Enable export button
            self.app_bn_export["state"] = "normal"            


    #Exit
    def closeApplication(self): 
        self.application.destroy()
     
        
    #Help
    def displayHelp(self): 
        self.helpForm = tkinter.Toplevel()
        self.helpForm.title("Help")
        self.helpForm.geometry("950x480+50+100")
                    
        #Help text
        help_tx_howto = tkinter.Text(master = self.helpForm, 
                                    width = 100, 
                                    height = 17, 
                                    spacing3 = 10)
        help_tx_howto.place(x = 100, y = 10)
        help_tx_howto.insert(1.0, "Number of Players: \
                             The number of players the Bonus is issued to.\n")
        help_tx_howto.insert(2.0, "Number of Spins: \
                             The number of spins issued to 1 player.\n")
        help_tx_howto.insert(3.0, "Spin Value: \
                             The amount of money 1 spin is worth.\n")
        help_tx_howto.insert(4.0, "Acceptance Rate: \
                             (Accepted Bonus / Issued Bonus) * 100\n")
        help_tx_howto.insert(5.0, "Probability: The probablity that the \
                             Cost falls between Max Cost and Min Cost.\n")
        help_tx_howto.insert(6.0, "             The higher the probability, \
                             the larger divide between Max and Min Cost.\n")
        help_tx_howto.insert(7.0, "Total: If ticked, the values specified in the\
                             Uniform section count as the total for all the games.\n")
        help_tx_howto.insert(8.0, "Individual/Uniform: Select individual if you \
                             want to assign different values to each selected game.\n")
        help_tx_howto.insert(9.0, "\n")
        help_tx_howto.insert(10.0, "How to use:\n")
        help_tx_howto.insert(11.0, "1. Click on Options, and set the Variables \
                             in the new window.\n")
        help_tx_howto.insert(12.0, "2. Once done, close it.\n")
        help_tx_howto.insert(13.0, "3. If everything is set, no warnings will \
                             appear and the Calculate button turns green.\n")
        help_tx_howto.insert(14.0, "4. Click Calculate, and the results will appear.\n")
        help_tx_howto.insert(15.0, "5. Export the information to a text file, \
                             which can be imported into excel. (Tabulated)\n")
    
        #Close Button
        help_bn_close = tkinter.Button(master = self.helpForm, 
                                         text = "Close", 
                                         command = self.closehelpForm)
        help_bn_close.place(x = 10, y = 10, width = 80)


    #Close helpForm
    def closehelpForm(self):
        self.helpForm.destroy()

    
    #Export
    def exportResults(self, location = ''):
        file = open(location, "w") #Open File to write 
        file.write(self.result.get(1.0, "end")) #Write the result to the file
        file.close() #Close the file

#Script
if __name__ == "__main__":
    BCE = estimator(location = "F:/Code/Bonus Cost/Casino.txt")
