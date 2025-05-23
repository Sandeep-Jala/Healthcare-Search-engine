from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import json
from selenium.common.exceptions import StaleElementReferenceException
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Shared resources and options
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.184 Safari/537.36")

# Global variables
start_url = "https://www.healthline.com"
visited_urls = set()
output_file = "collected_urls_FinalS.json"
lock = threading.Lock()

try:
    with open(output_file, "r") as file:
        saved_data = json.load(file)
        visited_urls = set(saved_data["visited_urls"])
        all_collected_urls = saved_data["all_urls"]
except (FileNotFoundError, json.JSONDecodeError):
    all_collected_urls = []

def save_progress():
    """Saves the current progress to a JSON file."""
    with open(output_file, "w") as file:
        json.dump({
            "visited_urls": list(visited_urls),
            "all_urls": all_collected_urls
        }, file)

def scrape_links(url, depth=0, max_depth=10):
    """Scrapes links from the given URL and returns new URLs."""
    if url in visited_urls or depth > max_depth:
        return []

    # Each thread creates its own webdriver instance
    driver = webdriver.Chrome(options=options)
    new_urls = set()
    
    try:
        driver.get(url)
        time.sleep(random.uniform(10, 20))
        visited_urls.add(url)

        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            try:
                href = link.get_attribute("href")
                if href and href.startswith("https://www.healthline"):
                    new_urls.add(href)
            except StaleElementReferenceException:
                continue
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
    finally:
        driver.quit()

    # Update shared resources with a lock
    with lock:
        all_collected_urls.extend(new_urls)
        save_progress()

    return list(new_urls)

def parallel_scrape(start_url):
    """Scrapes links in parallel starting from the root URL."""
    urls_to_scrape = [start_url]
    depth = 0
    max_depth = 10

    with ThreadPoolExecutor(max_workers=1) as executor:
        while depth <= max_depth:
            futures = {executor.submit(scrape_links, url, depth, max_depth): url for url in urls_to_scrape}
            urls_to_scrape.clear()

            for future in as_completed(futures):
                url = futures[future]
                try:
                    new_urls = future.result()
                    if new_urls:
                        urls_to_scrape.extend(new_urls)
                    logging.info(f"Completed scraping: {url} with {len(new_urls)} new URLs found.")
                except Exception as e:
                    logging.error(f"Error processing {url}: {e}")

            depth += 1

# Start parallel scraping
parallel_scrape(start_url)
print("Scraping completed. All URLs saved to", output_file)
