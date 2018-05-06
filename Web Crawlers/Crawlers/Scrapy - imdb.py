import scrapy
import pandas as pd
import io

class MemriseSpider(scrapy.Spider):
    name = "imdb"

    sPos = "F:\\Code\\Web Crawlers\\reviews\\positive\\"
    sNeg = "F:\\Code\\Web Crawlers\\reviews\\negative\\"
    iPos = 0
    iNeg = 0
    
    allowed_domains = ["www.imdb.com"]
    start_urls = ["https://www.imdb.com/chart/top?ref_=nv_mv_250_6",
                  "https://www.imdb.com/chart/bottom?ref_=nv_mv_250_6",
                  "https://www.imdb.com/chart/moviemeter?ref_=nv_mv_mpm_8",
                  "https://www.imdb.com/india/top-rated-indian-movies/?ref_=nv_mv_250_in_7",
                  "https://www.imdb.com/chart/toptv/?ref_=nv_tvv_250_3",
                  "https://www.imdb.com/chart/tvmeter?ref_=nv_tvv_mptv_4",
                  "https://www.imdb.com/chart/boxoffice?ref_=nv_ch_cht_1"]

    #follow the links to the title pages    
    def parse(self, response):
        # follow links to the title pages
        for title in response.xpath("//td[@class = 'titleColumn']//a[@href]"):
            yield response.follow(title, self.parseTitle)

    #follow links to the review pages
    def parseTitle(self, response):
        review = response.xpath("//div[@id = 'quicklinksMainSection']//a[@href]")
        yield response.follow(review[2], self.parseTexts)
        
    #get reviews
    def parseTexts(self, response):
        #Get the actual reviews into text files
        for i, rating in enumerate(response.xpath("//span[@class = 'rating-other-user-rating']//span[1]/text()").extract()):
            if int(rating) == 10 or int(rating) == 9:
                file = io.open(self.sPos + str(self.iPos) + ".txt", "w", encoding = "utf-8")
                file.write(response.xpath("//div[@class = 'text show-more__control']/text()").extract()[i])
                file.close()
                self.iPos += 1
            elif int(rating) == 1 or int(rating) == 2 or int(rating) == 3:
                file = io.open(self.sNeg + str(self.iNeg) + ".txt", "w", encoding = "utf-8")
                file.write(response.xpath("//div[@class = 'text show-more__control']/text()").extract()[i])
                file.close()
                self.iNeg += 1
