import scrapy
import pandas as pd

class MemriseSpider(scrapy.Spider):
    name = "memrise"
    allowed_domains = ["www.memrise.com"]
    start_urls = ["https://www.memrise.com/course/37726/advanced-german-vocabulary/1/?action=next"]
    dfVocab = pd.DataFrame()
    
    def parse(self, response):
        dest = "F:\\Code\\Web Crawlers\\memrise - DE Vocabulary.csv"

        ger_words = response.xpath("//div[@class='col_a col text']//div[@class='text']/text()").extract()
        eng_words = response.xpath("//div[@class='col_b col text']//div[@class='text']/text()").extract()
        iL = len(ger_words) 

        for i in range(0, iL):
            self.dfVocab = self.dfVocab.append([[ger_words[i], eng_words[i]]])

        next_page = response.xpath("//a[@class='level-nav level-nav-next']/@href").extract_first()
        
        if next_page is not None:
            yield response.follow(next_page, callback = self.parse)
        else:
            self.dfVocab.columns = ["German", "English"]        
            self.dfVocab.to_csv(dest, index = False)
