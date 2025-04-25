# LinkedIn Job Scraper & Analyzer

A comprehensive Python toolkit for scraping, analyzing, and extracting insights from job postings on LinkedIn.

## Features

- **Job ID Extraction**: Find relevant job postings based on custom keywords and trigger words
- **Job Details Retrieval**: Extract comprehensive information from LinkedIn job postings
- **Skill Analysis**: Automatically identify required skills in job descriptions
- **Experience Requirements**: Extract years of experience requirements
- **Relevant Section Extraction**: Focus on the most important parts of job descriptions
- **Text Processing**: NLP-based analysis of job postings
- **Blacklisting**: Prevent duplicate job postings across multiple searches

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/linkedin-job-scraper.git
   cd linkedin-job-scraper
   ```

2. Create a virtual environment and install the required dependencies:
   ```
   python -m venv .env
   
   # On Windows
   .env\Scripts\activate
   
   # On Linux/Mac
   source .env/bin/activate
   
   pip install -r requirements.txt
   ```

## Usage

### Basic Workflow

```python
from functions import *

# 1. Define your search parameters
trigger_words = ['data', 'machine learning', 'ai', 'ml ', 'statist', 'artificial intelligence', 'python']
keyword = 'Data science'
skills = ['python', 'sql', 'machine learning', 'pandas', 'numpy', 'tensorflow', 'pytorch']

# 2. Get job IDs matching your criteria
job_ids = get_job_ids(trigger_words, keyword, search_count=350, headers=None, 
                      internship=False, blacklist=True)

# 3. Fetch detailed job information
data = fetch_job_details(job_ids, headers=None)

# 4. Extract relevant information from job descriptions
data_info = job_info_extractor(data, skills=skills)

# 5. Save the results to CSV with today's date in the filename
save_jobs(data_info, keyword, date=True)

# 6. Add the current job IDs to blacklist for future searches
blacklist_job_ids(keyword, rows=None, cleanup=True, date=True)
```

### Geographic Location

The default location is set to the Netherlands (geoid=102890719). To change the location, update the geoid parameter.

### Custom Headers

You can provide custom headers for the HTTP requests to avoid rate limiting:

```python
headers = {
    'User-Agent': 'Your User Agent String',
    'Accept': '*/*'
    # Add other headers as needed
}

job_ids = get_job_ids(trigger_words, keyword, headers=headers)
```

### Automatic Skill Extraction

Instead of providing a predefined list of skills, you can automatically extract common skills from job descriptions:

```python
data_info = job_info_extractor(data, skills='extract')
```

## Function Reference

### get_job_ids(trigger_words, keyword, geoid, search_count, headers, internship, blacklist)
Scrapes LinkedIn jobs search results to find job IDs matching your criteria.

### fetch_job_details(job_ids, headers)
Fetches detailed information for each job ID, including job title, company name, location, and full description.

### job_info_extractor(df, skills, drop_original_text)
Extracts relevant information from job descriptions, including skills and years of experience.

### save_jobs(dataframe, job_title, date)
Saves the job data to a CSV file with optional date in the filename.

### load_jobs(job_title, date)
Loads previously saved job data from a CSV file.

### blacklist_job_ids(job_title, rows, cleanup, date)
Manages a blacklist of job IDs to avoid duplicates in future searches.

### extract_common_skills(job_postings, min_doc_freq, ngram_range)
Uses NLP techniques to automatically identify common skills mentioned in job postings.

## Limitations

- LinkedIn may change their API or HTML structure, requiring updates to the scraping functions
- Excessive scraping may lead to IP blocks - use responsibly with delays between requests
- The skill extraction is based on pattern matching and may require customization for specific domains

## License

This project is licensed under the GNU License - see the LICENSE file for details.

## Disclaimer

This tool is for educational purposes only. Web scraping may violate the terms of service of websites. Use responsibly and at your own risk.
