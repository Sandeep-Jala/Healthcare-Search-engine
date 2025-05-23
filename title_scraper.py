import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import json

# Load the URLs
data = pd.read_csv("healthline_urls.csv")
counter = 0
# Function to scrape a single URL with retry for 403 errors
def scrape_url(url,counter):
    retry_count = 0
    max_retries = 3  # Set maximum retry attempts
    wait_time = 240  # Wait time in seconds if 403 error is encountered
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.6167.184 Safari/537.36")

    result = {"ID":counter,"URL": url }

    while retry_count <= max_retries:
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            time.sleep(random.randint(3, 4))

            # Extract og:title
            try:
                og_title = driver.find_element(By.XPATH, "//meta[@property='og:title']")
                result["Title"] = og_title.get_attribute("content")
            except:
                result["Title"] = "No title found"

            # Extract the body text
            try:
                text = driver.find_element(By.TAG_NAME, "body").text
                # Check for "403 ERROR" in the body text
                if "403 ERROR" in text:
                    print(f"403 ERROR encountered: {url}, retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    retry_count += 1
                    continue  # Retry by re-entering the loop
                else:
                    result["text"] = text
            except:
                result["text"] = "No body text found"
            break  # Exit loop if no 403 error
        except Exception as e:
            print(f"Error with {url}: {e}")
            result["Title"] = "Error"
            result["text"] = "Error"
            break
        finally:
            driver.quit()

    if retry_count > max_retries:
        print(f"Max retries exceeded for {url}")
        result["Body"] = "403 ERROR - Max retries exceeded"
        
    print(f"Scraped ({url})")

    # Append the result to the JSON file
    with open("Scraped_data_13000.jsonl", "a") as json_file:
        json_file.write(json.dumps(result) + "\n")
    counter += 1
    return result

# Use ThreadPoolExecutor to run multiple scrapes in parallel
with ThreadPoolExecutor(max_workers=1) as executor:  # Adjust max_workers based on your system's capability
    future_to_url = {executor.submit(scrape_url, url, counter+13000): url for counter, url in enumerate(data["URL"][13000:15000])}   # Adjust URL count as needed
    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            future.result()
        except Exception as exc:
            print(f"{url} generated an exception: {exc}")

print("Data saved to Scraped_data_13000.jsonl")
