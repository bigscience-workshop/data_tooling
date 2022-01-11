#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datasets
datasets.set_caching_enabled(False)

from datasets import load_dataset

from pathlib import Path


# In[15]:


import multiprocessing

NUM_PROCS=multiprocessing.cpu_count()
print(f"num_procs: {NUM_PROCS}")

CC_INDEX_FOLDER=Path("/Users/thomas/code/bigscience/pseudo_crawl/") / "cc"


# ## To download index
# 
# You have run:
# ```bash
# aws s3 sync s3://commoncrawl-dev/big-science-workshop/data-sourcing-sheet/cc/ $CC_INDEX_FOLDER/
# ```

# In[2]:


def get_all_parquet_files(path):
    path = Path(path)
    all_crawls = [crawl for crawl in path.iterdir() if crawl.is_dir()]
    only_warcs = [subset for crawl in all_crawls for subset in crawl.iterdir() if subset.is_dir() and subset.name == "subset=warc"]
    return [str(file.absolute().resolve()) for subset in only_warcs for file in subset.iterdir() if file.is_file()]

ds = load_dataset("parquet", data_files=get_all_parquet_files(CC_INDEX_FOLDER), split="train[:10000]")


# In[3]:


print("\n".join(ds.column_names))


# In[4]:


print(len(ds))


# In[5]:


print(ds[0])


# In[6]:


print(set(zip(ds["content_languages"], ds["language"])))


# ## Getting pdf urls

# In[9]:


def get_pdf_urls(batch):
    content_mime_detected = batch["content_mime_detected"]
    urls = batch["url"]
    assert len(content_mime_detected) == len(urls)
    # Arrow doesn't support None, setting empty string for now
    batch["pdf_url"] = [url if mime == "application/pdf" else "" for mime, url in zip(content_mime_detected, urls)]
    return batch
    
    
ds = ds.map(get_pdf_urls, batched=True, num_proc=NUM_PROCS)

# Test that there are other paths
set(ds["pdf_url"])


# ## Get HTML and outgoing links
# 
# Problems:
#  - fetching data is too slow using http -> need to implement an asynchronous pipeline

# In[10]:


set(ds["warc_filename"])


# In[ ]:


import requests
from warcio.archiveiterator import ArchiveIterator

'''
Download all warc files and extract html
'''

HTML_TYPES=['text/html', 'application/xhtml+xml']
def get_html(batch):
    content_mime_detected = batch["content_mime_detected"] # select only text/html
    url_host_registered_domains = batch["url_host_registered_domain"]
    warc_filenames = batch["warc_filename"]
    warc_record_length = batch["warc_record_length"]
    warc_record_offset = batch["warc_record_offset"]
    assert len(content_mime_detected) == len(warc_filenames)
    assert len(content_mime_detected) == len(warc_record_length)
    assert len(content_mime_detected) == len(warc_record_offset)
    
    htmls = []
    for mime, filename, length, offset, domain in zip(content_mime_detected, warc_filenames, warc_record_length, warc_record_offset, url_host_registered_domains):
        if mime not in HTML_TYPES:
            htmls.append("")
            continue
            
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}"
        }

        with requests.get(f'https://commoncrawl.s3.amazonaws.com/{filename}', headers=headers, stream=True) as response:
    
            for record in ArchiveIterator(response.raw):
                if record.rec_type == 'response':
                    html = record.content_stream().read()
                    break
        
        htmls.append(html)
        
    batch["html"] = htmls
    return batch
    
ds = ds.map(get_html, batched=True, batch_size=100, num_proc=NUM_PROCS)


# In[ ]:


from bs4 import BeautifulSoup
import re

#Retrieves a list of all external links found on a page
def get_external_links(soup, exclude_url):
    external_links = []
    #Finds all links that start with "http" that do
    #not contain the current URL
    for link in soup.find_all('a', {'href' : re.compile('^(((http|https)://)|www){1,2}((?!'+exclude_url+').)*$')}):
        if link.attrs['href'] is not None:
            if link.attrs['href'] not in external_links:
                external_links.append(link.attrs['href'])
    return external_links


def preprocess_html(soup):
    text = soup.get_text()
    text = re.sub(r"\t{2,}","\t",text)
    text = re.sub(r"((\s+)\n(\s+))+","\n",text)
    return text
    
def get_text_and_outgoing_lings(batch):
    content_mime_detected = batch["content_mime_detected"] # select only text/html
    url_host_registered_domains = batch["url_host_registered_domain"]
    htmls = batch["html"]
    assert len(content_mime_detected) == len(warc_filenames)
    assert len(content_mime_detected) == len(warc_record_length)
    assert len(content_mime_detected) == len(warc_record_offset)

    texts=[]
    external_urls=[]    
    for mime, html, domain in zip(content_mime_detected, htmls, url_host_registered_domains):
        if mime not in HTML_TYPES:
            texts.append("")
            external_urls.append([])
            continue
           
        soup = BeautifulSoup(html, 'html.parser')
        text = preprocess_html(soup)
        texts.append(text)
        external_urls.append(get_external_links(soup, domain))
        
        
    batch["text"] = texts
    batch["external_urls"] = external_urls
    return batch
    
import time

t0=time.time()
ds = ds.map(get_text_and_outgoing_lings, batched=True, batch_size=100, num_proc=NUM_PROCS)
t1=time.time()
print(f"get_html: {t1-t0}")


# In[ ]:


ds[0]["warc_record"]


# In[ ]:


ds[4]["text"]


# In[ ]:


ds[0]["external_urls"]


# ## Cleaning up dataset

# In[ ]:


columns_to_keep = ["id", "title", "link", "languages", "pdf_url", "html", "text", "external_urls"]
columns_to_remove = [column for column in ds.column_names if column not in columns_to_keep]
print(columns_to_remove)
cleaned_ds = ds.remove_columns(columns_to_remove)


# In[ ]:


print(cleaned_ds[0])

