## Strategy to get depth 1

### Context

Once we've extract all the seed pages, we plan to make a pseudo crawl. The idea is simple:
 - we extract the outgoing urls from those pages.
 - we find the most recent record in CC matching that url (if it exists).
 - we do the entire processing for all the new records.pages
 - we update `outgoing_urls` to obtain `outgoing_ids`

### Problems

The issue if that last part

### Solution