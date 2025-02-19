import sys
import os
import time
import logging
from datetime import datetime
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cybernews.CyberNews import CyberNews
from models.NewsModel import CybernewsDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rss_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RSSChecker')

class RSSChecker:
    def __init__(self):
        self.pinecone_api = dotenv_values(".env").get("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.pinecone_api)
        self.index_name = "cybernews-index"
        self.namespace = "c2si"
        self.index = self.pc.Index(self.index_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.news = CyberNews()
        self.db = CybernewsDB()
        
    def get_existing_articles(self):
        """Fetch existing articles from the database"""
        try:
            existing_articles = self.db.get_news_collections()
            # Create a set of URLs for quick comparison
            return {article["newsURL"] for article in existing_articles if "newsURL" in article}
        except Exception as e:
            logger.error(f"Error fetching existing articles: {str(e)}")
            return set()

    def process_new_articles(self, articles, existing_urls):
        """Process and insert new articles into the database"""
        for article in articles:
            try:
                if article["newsURL"] not in existing_urls:
                    # Combine headline and full news for embedding
                    text = article["headlines"] + " " + article["fullNews"]
                    
                    # Convert text to vector
                    vector = self.model.encode(text).tolist()
                    
                    # Prepare document ID
                    document_id = str(article["id"])
                    
                    # Prepare metadata
                    metadata = {
                        "headlines": article["headlines"],
                        "author": article["author"],
                        "fullNews": article["fullNews"],
                        "newsURL": article["newsURL"],
                        "newsImgURL": article["newsImgURL"],
                        "newsDate": article["newsDate"]
                    }
                    
                    # Upsert the vector with metadata
                    self.index.upsert(
                        vectors=[(document_id, vector, metadata)],
                        namespace=self.namespace
                    )
                    
                    logger.info(f"Inserted new article: {article['headlines']}")
            except Exception as e:
                logger.error(f"Error processing article {article.get('headlines', 'Unknown')}: {str(e)}")

    def check_and_update(self):
        """Main method to check and update RSS feeds"""
        try:
            logger.info("Starting RSS feed check")
            
            # Get existing articles
            existing_urls = self.get_existing_articles()
            
            # Get new articles for each news type
            news_types = ["general", "cyberAttack", "vulnerability", 
                         "malware", "security", "dataBreach"]
            
            for news_type in news_types:
                try:
                    logger.info(f"Checking {news_type} news")
                    articles = self.news.get_news(news_type)
                    self.process_new_articles(articles, existing_urls)
                except Exception as e:
                    logger.error(f"Error processing {news_type} news: {str(e)}")
                    continue
                
            logger.info("RSS feed check completed")
            
        except Exception as e:
            logger.error(f"Error in check_and_update: {str(e)}")

def run_periodic_check(interval_minutes=30):
    """Run the RSS checker periodically"""
    checker = RSSChecker()
    
    while True:
        try:
            start_time = datetime.now()
            logger.info(f"Starting RSS check at {start_time}")
            
            checker.check_and_update()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"RSS check completed. Duration: {duration:.2f} seconds")
            
            # Wait for the next interval
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("RSS checker stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in periodic check: {str(e)}")
            # Wait a shorter time before retry in case of error
            time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    try:
        # Run the checker every 180 minutes
        run_periodic_check(180)
    except KeyboardInterrupt:
        logger.info("RSS checker stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")