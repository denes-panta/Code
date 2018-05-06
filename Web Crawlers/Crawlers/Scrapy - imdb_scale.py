import scrapy
import pandas as pd
import io

class MemriseSpider(scrapy.Spider):
    name = "imdb_scale"

    sPath = "F:\\Code\\Web Crawlers\\reviews\\Scale\\"
    mScale = [0 for i in range(1, 11)]
    
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
            file = io.open(self.sPath + 
                           rating + 
                           "//" + 
                           str(self.mScale[int(rating) - 1]) + 
                           ".txt", 
                           "w", encoding = "utf-8")
            file.write(response.xpath("//div[@class = 'text show-more__control']/text()").extract()[i])
            file.close()
            self.mScale[int(rating) - 1] += 1
