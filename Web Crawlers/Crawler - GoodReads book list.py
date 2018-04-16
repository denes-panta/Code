import requests as r
import re
import pandas as pd
from bs4 import BeautifulSoup as soup

class MultiCrawler(object):
    
    def __init__(self, site = '', dest = None):
        #Check for destination file address
        if dest == None:
            print("No destination folder")
        else:
            #Get destination file
            self.sDest = dest
            #Get site address
            self.sUrl = site
            #Extract the main address
            self.sMain = re.findall("^.*.com", self.sUrl)[0]
            #Get the links of the Book Lists
            self.lLists = self.getLinks(self.sUrl, "listTitle")
            #Get the links to each book
            self.lItems = list(set(self.getItems(self.lLists)))
            #Extract the data from each webpage
            self.dfBooks = self.getData(self.lItems)
            #Export data to CSV file
            self.exportData(self.dfBooks, self.sDest)
            
    def getItems(self, lst):
        lItems = list()
  
        for link in lst:
            lItems += self.getLinks(link, "bookTitle")

        return (lItems)
        
    def getLinks(self, url, cls):
        lLinks = list()
        bPage = True
        iPage = 0
        
        while bPage == True:
            iPage += 1
            sUrl = url + "?page=" + str(iPage)
            sPage = self.extractLinks(sUrl, cls)
            
            if sPage != []:
                lLinks += sPage
            else:
                bPage = False

        return (lLinks)
    
    def extractLinks(self, url, cls):
        lLinks = list()
        bPageReached = False
        iTries = 0
        
        #Try to reach each link 10 times
        while iTries < 10 and bPageReached == False:
            try:
                h_page = r.get(url)
                s_page = soup(h_page.text, "html.parser")
                bPageReached = True                
            except:
                iTries += 1

        if bPageReached == True:                
            for a in s_page.find_all('a', class_ = cls):
                lLinks.append(self.sMain + a["href"])

        return (lLinks)

    def getData(self, lst):
        dfItems = pd.DataFrame()
        print("### Extracting Data ###")
        
        #Print out the progress
        for i, link in enumerate(lst):
            print("Extracting " + 
                  str(i + 1) + 
                  " of " + 
                  str(len(lst)) + 
                  ": " +
                  link
                  )
            
            iTries = 0
            bPageReached = False

            #Try to reach each link 10 times
            while iTries < 10 and bPageReached == False:
                try:
                    h_page = r.get(link)
                    s_page = soup(h_page.text, "html.parser")
                    bPageReached = True
                except:
                    iTries += 1

            #If the link was reached, extract the data
            if bPageReached == True:                        
                print("Done")
                lAuthor = s_page.find_all('a', class_ = "authorName")
                try:
                    sAuthor = \
                    re.sub(' +',' ', lAuthor[0].text.replace('\n', '').strip())
                except:
                    sAuthor = "Not Available"
    
                lTitle = s_page.find_all('h1', class_ = "bookTitle")
                try:
                    sTitle = \
                    re.sub(' +',' ', lTitle[0].text.replace('\n', '').strip())
                except:
                    sTitle = "Not Available"
        
                lAvgRating = s_page.find_all('span', class_ = "value")
                try:
                    sAvgRating = \
                    lAvgRating[0].text.replace('\n', '').replace(',','').strip()
                    fAvgRating = float(sAvgRating)
                except:
                    fAvgRating = "Not Available"
    
                lRatings = s_page.find_all('span', class_ = "votes value-title")
                try:
                    sRatings = \
                    lRatings[0].text.replace('\n', '').replace(',','').strip()
                    iRatings = int(sRatings)
                except:
                    iRatings = "Not Available"
    
                lReviews = s_page.find_all('span', class_ = "count value-title")
                try:
                    sReviews = \
                    lReviews[0].text.replace('\n', '').replace(',','').strip()
                    iReviews = int(sReviews)
                except:
                    iReviews = "Not Available"
    
                lDetails = s_page.find_all('div', class_ = "row")
                try:
                    sDetails_1 = re.findall("(^.*),", lDetails[0].text)[0]
                except:
                    sDetails_1 = "Not Available"
    
                try:
                    sDetails_2 = re.findall("(\d.*) pages", lDetails[0].text)[0]
                    iDetails_2 = int(sDetails_2)
                except:
                    iDetails_2 = "Not Available"
    
                try:
                    sDetails_3 = re.findall("\d.*", lDetails[1].text)[0][-4:]
                    iDetails_3 = int(sDetails_3)
                except:
                    iDetails_3 = "Not Available"
                
                lNextRecord = [[sAuthor,
                                sTitle,
                                fAvgRating, 
                                iRatings, 
                                iReviews,
                                sDetails_1,
                                iDetails_2, 
                                iDetails_3]]
                
                #Append the df with the new record                
                dfItems = dfItems.append(lNextRecord)
            else:
                print("Error")

        return (dfItems)

    def exportData(self, df, dest):
        df.columns = ["Author", 
                      "Title",
                      "Avg. Rating", 
                      "Number of Ratings", 
                      "Number of Reviews", 
                      "Cover Type", 
                      "Pages", 
                      "Year Published"]

        df.to_csv(dest, index = False)
        print("Data exported")
        
if __name__ == "__main__":       
    Crawler = \
    MultiCrawler(site = "https://www.goodreads.com/list/tag/dutch",
                 dest = "F:\\Code\\Web Crawlers\\GoodReads - Dutch.csv")
    