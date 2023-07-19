```python
import scrapy
from scrapy.crawler import CrawlerProcess
from web_scraper.spiders.reddit_spider import RedditSpider
from web_scraper.settings import Settings

def main():
    process = CrawlerProcess(Settings())
    process.crawl(RedditSpider)
    process.start()

if __name__ == "__main__":
    main()
```