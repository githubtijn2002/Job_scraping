A comprehensive Python toolkit for scraping, analyzing, and extracting insights from job postings on LinkedIn.

## Features

- **Job ID Extraction**: Find relevant job postings based on custom keywords and trigger words
- **Job Details Retrieval**: Extract comprehensive information from LinkedIn job postings
- **Skill Analysis**: Automatically identify required skills in job descriptions
- **Experience Requirements**: Extract years of experience requirements
- **Relevant Section Extraction**: Focus on the most important parts of job descriptions
- **Text Processing**: NLP-based analysis of job postings

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/linkedin-job-scraper.git
   cd linkedin-job-scraper
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure you have NLTK data downloaded:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Usage

### Basic Workflow

```python
from functions import *

# Define your search parameters
trigger_words = ['data', 'machine learning', 'ai', 'ml', 'statist', 'artificial intelligence', 'python']
keyword = 'data scientist'
skills = ['python', 'sql', 'machine learning', 'pandas', 'numpy', 'tensorflow', 'pytorch']

# Get job IDs matching your criteria
job_ids = get_job_ids(trigger_words, keyword, geoid=102890719, search_count=350)

# Fetch detailed job information
data = fetch_job_details(job_ids)

# Save the data to CSV
save_jobs(data, keyword, date=True)

# Extract job requirements and skills
df = job_info_extractor(data, skills=skills)
```

### Geographic Location

The default location is set to the Netherlands (`geoid=102890719`). To change the location, update the `geoid` parameter.

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

## Function Reference

### `get_job_ids(trigger_words, keyword, geoid, search_count, headers, internship)`
Scrapes LinkedIn jobs search results to find job IDs matching your criteria.

### `fetch_job_details(job_ids, headers)`
Fetches detailed information for each job ID, including job title, company name, location, and full description.

### `save_jobs(dataframe, job_title, date)`
Saves the job data to a CSV file with optional date in the filename.

### `load_jobs(job_title, date)`
Loads previously saved job data from a CSV file.

### `job_info_extractor(df, skills)`
Extracts relevant information from job descriptions, including skills and years of experience.

### `extract_common_skills(job_postings, min_doc_freq, ngram_range)`
Uses NLP techniques to automatically identify common skills mentioned in job postings.

## Limitations

- LinkedIn may change their API or HTML structure, requiring updates to the scraping functions
- Excessive scraping may lead to IP blocks - use responsibly with delays between requests
- The skill extraction is based on pattern matching and may require customization for specific domains

## License

This project is licensed under the GNU License - see the LICENSE file for details.

## Disclaimer

This tool is for educational purposes only. Web scraping may violate the terms of service of websites. Use responsibly and at your own risk.
