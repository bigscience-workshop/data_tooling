# Data Tooling and Governance
Tools for managing datasets for governance and training large language models.

## Issues we aim to address
- How do we automatically curate data to create datasets that are performant and comply with BigScience ethical values?
- How do we remediate a dataset for personally identifiable information without degrading performance?
- How should we store and serve the dataset?
- How do we store and serve meta-data in datasets?
- How do we address contestation of data?
- How do we prove legal compliance in the use of datasets?
- How do we prevent dissemination of the data beyond approved uses?
- How do we keep trusted data secure?

## Format to distribute data samples

Current consensus is to use [`jsonl`](https://jsonlines.org/).

## Metadata guideline

Trying to keep things as simple as possible, the proposed metadata guideline is simply a flat format in a key/value format where values format could be constrained. The goal is not to be exhaustive on all possible metadata but to be pragmatic to align on the things that are already in the process of being recorded.

### Metadata subjects

For now 3 "objects of interests" are forseen in the project on which metadata could be applied:

- data sources
- data set
- data sample (or document)

### Key general format

Simply text, in small cap and avoiding any punctuation (including spaces to be replaced by underline character '_' if really necessary)  to ease automated parsing.

### Value general format

It will vary for each metadata key with an open standard to be used as reference whenever applicable. All text should be encoded in UTF-8 to cope with any language scripts.

Some proposed value formats:

| Type of value           | Value                                                                                                                                                                                                                                      |
|-----------    |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    |
| language      | ISO_639-3 => 3 letter codes with the most coverage (see for reference[wikipedia list of ISO_639-3 codes](https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Languages/List_of_ISO_639-3_language_codes_(2019))) ex: afr for Afrikaans     |
|               | Note: IETF bcp 47 was proposed as an alternative in the data sourcing group, a list of standard values would be necessary to be practical (like the wikipedia list for ISO_639-3)
| timestamp     | ISO_8601 normalized to UTC time zone ex: 2021-07-06T15:47:46+00:00                                                                                                                                                                         |
| URL           | Full length URL including the scheme (eg http/ftp...)  ex: https://en.wikipedia.org/wiki/URL                                                                                                                                               |
| text          | free text encoded in UTF-8

#### Annotation of text content

For data sample, it is foreseen that information might be extracted from the original text content such as named entities using position reference to the original content.

TO BE CONTINUED

### Data source metadata format

TO BE CONTINUED

### Dataset metadata format

TO BE CONTINUED

### Data sample metadata format

| Key                       | Value                                                                                                                                                                                                                                      |
|-----------------------    |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    |
| hash                | unique fixed length that could be used as identifier - proposal to use murmur hash 128 bits                                                                                                                                                                         |
| main_language             | ISO_639-3 => 3 letter codes with the most coverage (see for reference[wikipedia list of ISO_639-3 codes](https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Languages/List_of_ISO_639-3_language_codes_(2019))) ex: afr for Afrikaans     |
| other_languages           | list of ISO_639-3 codes of all languages possibly found in the sample without specific order                                                                                                                                               |
| collection_timestamp      | timestamp of original collection of the data (eg. for web crawl) if precisely known the default would be the timestamp of dataset creation                                                                                                 |
| publication_timestamp     | timestamp of the publication of the data (first time a web page has been online or last edit, radio/tv shows publication...)                                                                                                               |
| original_content          | free text

TO BE CONTINUED

ex:
```json
["89faeee174d2ddbc2b761207efbc8464", "fra", ["eng", "deu"], "2021-07-06T19:06:02Z", null, "je crois il est parti à Stuttgart ou bien à London"]
```
