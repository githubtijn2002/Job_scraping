# FUNCTION FOR FETCHING JOB IDs FROM LINKEDIN
def get_job_ids(trigger_words, keyword, geoid=102890719, search_count=250, headers=None, internship=False, blacklist=False):
    """
    Get job IDs from LinkedIn based on trigger words and keyword.
    # explain the inputs and outputs
    :param trigger_words: List of words to search for in job titles or descriptions.
    :param keyword: Keyword to search for in job titles or descriptions.
    :param geoid: Geographical ID for the job search location. Default is 102890719 (Netherlands).
    :param search_count: Number of job postings to fetch. Default is 250.
    :param headers: Optional headers for the request. If None, default headers will be used. If False, no headers will be used.
    :param internship: Boolean indicating whether to include internship positions. Default is False.
    :param blacklist: Boolean indicating whether to use a blacklist of job IDs. Default is False.
    :return: List of job IDs that match the trigger words and keyword.
    """
    import requests
    from bs4 import BeautifulSoup
    from time import sleep
    import random
    job_ids = []
    keyword = keyword.replace(' ', '%2B')
    # round search_count to the nearest multiple of 25
    search_count = (search_count // 25) * 25
    if search_count > 1000:
        search_count = 1000
        print(f"Search count exceeds 1000, setting to 1000.")
    if search_count < 25:
        search_count = 25
        print(f"Search count is less than 25, setting to 25.")
        
    if headers is None:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'}
    if isinstance(headers, dict) == False and headers != False:
        exception = TypeError("Headers must be a dictionary or False.")
        raise exception
    
    for start in range(0, search_count, 25):
        URL = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={keyword}&location=Nederland&geoId={geoid}&start={start}"
        if headers == False:
            response = requests.get(URL)
        else:
            response = requests.get(URL, headers=headers)
        if response.status_code != 200:
            print(f"Error: Unable to fetch data from LinkedIn. Status code: {response.status_code}")
            # leave the loop if the request fails
            break
        else:
            print("Data fetched successfully!")
            data = response.text
        soup = BeautifulSoup(data, 'html.parser')
        job_listings = soup.find_all('a', class_='base-card__full-link absolute top-0 right-0 bottom-0 left-0 p-0 z-[2]')


        for job in job_listings:
            if any(word in job.text.strip().lower() for word in ['intern', 'afstudeeropdracht', 'stage']):
                if internship == False:
                    print(f"--- Job ID {job.text.strip()} is an intern position.")
            elif any(word in job.text.strip().lower() for word in trigger_words):
                if internship == True:
                    print(f"--- Job ID {job.text.strip()} is an intern position.")
                else:
                    job_ids.append(('').join(job.get('href').split('/')[5:]).split('-')[-1].split('?')[0])
                    print(f"!!! Job ID {job.text.strip()} contains trigger words.")
            else:
                print(f"Job ID {job.text.strip()} does not contain trigger words.")
        print('')
        sleep(random.randint(0, 2))

    print(f"Total job IDs found: {len(job_ids)}")
    print("--------------------------------------")
    if blacklist == True:
        try:
            with open('blacklist_ids.txt', 'r') as f:
                blacklist = set(f.read().splitlines())
        except FileNotFoundError:
            print("Blacklist was enabled, but no blacklist file was found.")
            return job_ids
        # remove the blacklist_ids from the job_ids list
        job_ids = list(set(job_ids) - blacklist)
        print(f"Total job IDs after removing blacklist: {len(job_ids)}")
        print("--------------------------------------")
    return job_ids





# FUNCTIONS FOR FETCHING JOB DETAILS FROM LINKEDIN
def fetch_job_details(job_ids, headers=None):
    """
    Fetch job details from LinkedIn based on job IDs.
    :param job_ids: List of job IDs to fetch details for.
    :param headers: Optional headers for the request. If None, default headers will be used. If False, no headers will be used.
    :return: DataFrame containing job details.
    """
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from time import sleep
    import random
    
    if headers == 'debug':
        debug = True
        headers = None
        print("Debug mode enabled, using no headers for first request.")
    else:
        debug = False
    if headers is None:
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'}
        if debug == False:
            print("Using default headers.")
    if isinstance(headers, dict) == False and headers != False:
        exception = TypeError("Headers must be a dictionary or False.")
        raise exception
    
    if isinstance(job_ids, list) == False:
        raise ValueError("Job IDs must be a list created by the get_job_ids function.")
    if len(job_ids) == 0:
        raise ValueError("No job IDs found. Please check the get_job_ids function.")
    data = {'job_title' : [], 'job_company': [], 'job_location': [], 'days_ago': [], 'company_description': [], 'job_description': []}
    error_count = 0
    perm_error = 0
    for idx,job_id in enumerate(job_ids):
        URL = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        if idx > 0 and debug == True:
            response = requests.get(URL, headers=headers)
        elif headers == False:
            response = requests.get(URL)
        else:
            response = requests.get(URL, headers=headers)
        if response.status_code != 200:
            print(f"Error: Unable to fetch data from LinkedIn. Status code: {response.status_code}")
            data['job_title'].append('')
            data['job_company'].append('')
            data['job_location'].append('')
            data['days_ago'].append('')
            data['company_description'].append('')
            data['job_description'].append('')
            error_count += 1
            perm_error += 1
            if error_count > 5:
                raise ValueError("Too many errors encountered. Please check your headers or try setting headers to 'debug'.")
            continue
        else:
            error_count = 0
            print("Data fetched successfully!")
            data2 = response.text
        soup2 = BeautifulSoup(data2, 'html.parser')
        soup2.find('div', class_= "show-more-less-html__markup show-more-less-html__markup--clamp-after-5 relative overflow-hidden")
        text = soup2.get_text(strip=True, separator="~~")
        data['job_title'].append(text.split('~~')[0])
        data['job_company'].append(text.split('~~')[1])
        data['job_location'].append(text.split('~~')[2])
        data['days_ago'].append(text.split('~~')[3].split(' ')[0])
        data['company_description'].append((' ').join(text.split('~~')[10:12]))
        job_desc = (' ').join(text.split('~~')[13:-12])
        #if job_desc[0:12] == 'Remove photo':
        #    job_desc = job_desc.split(' ')[49:]
        data['job_description'].append(job_desc)
        if (idx+1) % 10 == 0:
            print(f"Processed {idx+1} job postings.")
            print('')
        sleep(random.randint(0, 1))

    print(f"Total job postings processed: {len(data['job_title'])}")
    print(f"Total errors: {perm_error}")
    print("--------------------------------------")
    print("Cleaning up data...")
    # CLeaning linkedin prefixes
    for idx,desc in enumerate(data['job_description']):
        if desc[:10] == 'Sign in to':
            data['job_description'][idx] = desc[1212:]
        elif desc[:12] == 'Remove photo':
            if desc[291:][:6] == 'Use AI':
                data['job_description'][idx] = desc[1642:]
            else:
                data['job_description'][idx] = desc[291:]
        else:
            data['job_description'][idx] = desc
    print("Data cleaned up.")
    print("--------------------------------------")
    job_links = [f"https://www.linkedin.com/jobs/search/?currentJobId={job_id}" for job_id in job_ids]
    print("Creating DataFrame...")
    df = pd.DataFrame(data)
    df['job_link'] = job_links
    print(f'Entries before dropping duplicates: {len(df)}')
    print("--------------------------------------")
    df = df.drop_duplicates(subset=['job_title', 'job_company', 'job_description'], keep='first')
    print(f'Entries after dropping duplicates: {len(df)}')
    df.reset_index(drop=True, inplace=True)

    hyperlink = []
    for i in range(len(df)):
        hyperlink.append(f'=HYPERLINK("{df["job_link"][i]}")')
    df['job_link'] = hyperlink
    print("DataFrame created successfully!")
    return df





# FUNCTIONS FOR SAVING AND LOADING JOB POSTINGS from csv files
def save_jobs(dataframe, job_title, date = False):
    """
    Save job postings to a CSV file.
    :param dataframe: DataFrame containing job postings.
    :param job_title: Job title used for naming the CSV file.
    :param date: Date string in 'dd-mm-yyyy' format. If True, uses today's date. If False, uses no date.
    :return: None
    """
    import time
    if date == True:
        today = time.strftime("%d-%m-%Y")
        filename = f"{job_title.replace(' ', '_')}_jobs_{today}.csv"
    elif date == False:
        filename = f"{job_title.replace(' ', '_')}_jobs.csv"
    else:
        # see if date is in the right format
        try:
            time.strptime(date, "%d-%m-%Y")
            filename = f"{job_title.replace(' ', '_')}_jobs_{date}.csv"
        except ValueError:
            print("Date format is incorrect. Please use 'dd-mm-yyyy'.")
            return None
    dataframe.to_csv(filename, index=False)
    print(f"Job postings saved to {filename}")

def load_jobs(job_title, date = False):
    """
    Load job postings from a CSV file.
    :param job_title: Job title used for naming the CSV file.
    :param date: Date string in 'dd-mm-yyyy' format. If True, uses today's date. If False, uses no date.
    :return: DataFrame containing job postings.
    """
    import time
    import pandas as pd
    if date == True:
        today = time.strftime("%d-%m-%Y")
        filename = f"{job_title.replace(' ', '_')}_jobs_{today}.csv"
    elif date == False:
        filename = f"{job_title.replace(' ', '_')}_jobs.csv"
    else:
        # see if date is in the right format
        try:
            time.strptime(date, "%d-%m-%Y")
            filename = f"{job_title.replace(' ', '_')}_jobs_{date}.csv"
        except ValueError:
            print("Date format is incorrect. Please use 'dd-mm-yyyy'.")
            return None
    try:
        dataframe = pd.read_csv(filename)
        dataframe['job_link'] = dataframe['job_link'].apply(lambda x: x.split(',')[0])
        print(f"Job postings loaded from {filename}")
        return dataframe
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    

# FUNCTION FOR BLACKLISTING JOB IDs
def blacklist_job_ids(job_title, rows=None, cleanup=True, date=True):
    """
    This function takes a CSV file and a number of rows as input and checks if the job ids in the CSV file are in the blacklist.
    If the job id is not in the blacklist, it will be added to the blacklist.
    :param job_title: The job title to search for in the CSV file.
    :param rows: The number of rows to check in the CSV file. If None, all rows will be checked.
    :param cleanup: If True, the function will remove the outdated job ids from the blacklist.
    :param date: If True, the function will use the current date to load the CSV file. If False, it will use the date in the filename.
    :return: None
    """
    import pandas as pd
    # try to read in the blacklist as a list
    try:
        with open('blacklist_ids.txt', 'r') as f:
            blacklist = set(f.read().splitlines())
    except FileNotFoundError:
        blacklist = set()
    df = load_jobs(job_title, date=date)
    # 
    ids = df['job_link'].apply(lambda x: x.split('"')[1].split('=')[1]).tolist()
    if rows:
        ids = ids[:rows]
    set_ids = set(ids)
    # find the ids in blacklist that are not in the set_ids
    blacklist_ids = set_ids.union(blacklist)
    if cleanup == True:
        blacklist_ids = blacklist_ids - (blacklist - set_ids)
        print(f'Found {len(blacklist - set_ids)} outdated job ids in the blacklist.')
        if len(blacklist - set_ids) > 0:
            print(f'Cleaning up the blacklist...')
        else:
            print(f'No outdated job ids in the blacklist.')
    # write blacklist_ids to a file
    with open('blacklist_ids.txt', 'w') as f:
        for id in blacklist_ids:
            f.write(id + '\n')




# FUNCTIONS FOR EXTRACTING RELEVANT SECTIONS FROM JOB POSTINGS
def create_patterns():
    """
    Create regex patterns from keywords for efficient matching.
    :return: List of compiled regex patterns.
    """
    import re
    section_keywords = [
        # English
        "skills", "requirements", "responsibilities", "who you are", "who are you",
        "qualifications", "desired profile", "your background", "what you bring", 
        "about the candidate", "candidate profile", 
        "nice to have", "we are looking for", "we're looking for", 
        "looking for someone who", "experience and skills", "python", "years of experience", "SQL"

        # Dutch
        "wat ga je doen", "wat je gaat doen", "functie-eisen", "wie zoeken wij", 
        "wat breng je mee", "vaardigheden", "wie ben jij", "jouw profiel", 
        "gewenst profiel", "eisen", "jij bent", "jij hebt", "jouw kwalificaties", 
        "over jou", "wat wij zoeken", "wie we zoeken", "je profiel", 
        "wij zoeken iemand die", "wij zijn op zoek naar"
    ] 
    patterns = [re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE) for token in section_keywords]
    return patterns


def extract_relevant_sections(job_posting_text, context_window):
    """
    Extract relevant sections from job postings based on keywords.
    :param job_posting_text: Text of the job posting.
    :param context_window: Number of surrounding sentences to include for context.
    :return: Extracted relevant sections as a string.
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(job_posting_text)
    patterns = create_patterns()

    matches = set()  # Use a set to avoid duplicate sentences
    for i, sentence in enumerate(sentences):
        if any(p.search(sentence) for p in patterns):
            # Optionally, capture surrounding sentences for context
            start = max(0, i - context_window)
            end = min(len(sentences), i + context_window + 1)
            for j in range(start, end):
                matches.add(sentences[j])  # Add each sentence to the set
    relevant_text = ' '.join(sorted(matches, key=sentences.index))  # Preserve original order

    return relevant_text

def job_info_extractor(df, skills = 'extract', drop_original_text = True):
    """
    Extract relevant sections from job postings based on keywords.
    :param df: DataFrame containing job postings.
    :param skills: List of skills to extract. If 'extract', common skills will be extracted.
    :param drop_original_text: Boolean indicating whether to drop the original job description column.
    :return: DataFrame with relevant sections extracted.
    """
    import re
    import nltk
    nltk.download('punkt_tab')

    jobs_relevant_info = [extract_relevant_sections(job, context_window=2) for job in df['job_description']]
    df['relevant_text'] = jobs_relevant_info
    print("Relevant sections extracted successfully!")

    years_of_experience = []
    for job in df['job_description']:
        match = re.search(r"(\d+)\s*[-]?\s*(?:years?|jaar)", job, re.IGNORECASE)
        if match:
            years_of_experience.append(int(match.group(1)))
        else:
            years_of_experience.append(None)
    df['years_of_experience'] = years_of_experience
    print(f"Years of experience extracted successfully for {len(years_of_experience)} job postings!")

    if skills == 'extract':
        skills = extract_common_skills(df['job_description'].to_list(), ngram_range=(1, 2))

    skills_found = []
    for job in df['job_description']:
        found_skills = [skill for skill in skills if re.search(rf"\b{re.escape(skill)}\b", job, re.IGNORECASE)]
        skills_found.append(found_skills)
    df['skills'] = skills_found
    print(f"Skills extracted successfully for {len(skills_found)} job postings!")
    if drop_original_text == True:
        df.drop(columns=['job_description'], inplace=True)
    return df





# FUNCTIONS FOR EXTRACTING COMMON SKILLS FROM JOB POSTINGS
def create_blacklist(*args):
    """
    Create a blacklist of generic words and verbs to exclude from n-gram extraction.
    :param args: Additional words to include in the blacklist (lists, sets, or strings).
    :return: Set of words to be excluded from n-gram extraction.
    """
    generic_words = {
        "experience", "skills", "team", "environment", "communication",
        "responsible", "ability", "knowledge", "understanding", "excellent",
        "good", "working", "collaboration", "background", "position", "task",
        "project", "interpersonal", "detail", "driven", "motivated", "youll", "you", "your", "years", "experience",
        "we", "us", "our", "they", "them", "their", "candidate", "candidates", "max", "min", "solutions", "impact", "expertise", "teams",
        "role", "qualifications", "job", "strong"
    }
    dutch_generic_words = {
        "ervaring", "vaardigheden", "team", "omgeving", "communicatie",
        "verantwoordelijk", "vermogen", "kennis", "begrip", "uitstekend",
        "goed", "werkend", "samenwerking", "achtergrond", "functie", "taak",
        "project", "interpersoonlijk", "gedetailleerd", "gedreven", "gemotiveerd", "jij", "jouw",
        "wij", "ons", "onze", "zij", "hun", "hunnen", "kandidaat", "kandidaten", "je", "jaar", "ervaring", "jaren", "baan",
        "oplossingen",
    }
    verbs_en = {
        "develop", "manage", "support", "work", "design", "lead",
        "implement", "coordinate", "assist", "analyze", "help", "drive",
        "collaborate", "contribute", "communicate", "engage", "build", "need", "meet", "require",
        "have", "possess", "demonstrate", "show", "provide", "offer", "possess", "bring",
        "create", "deliver", "execute", "achieve", "conduct", "perform", "maintain", "include", "ensure",
        "develop", "apply", "utilize", "leverage", "adapt", "implement", "execute", "manage", "oversee", "looking",
        "make", "take", "drive", "lead", "influence", "shape", "foster", "cultivate",
    }
    verbs_nl = {
        "ontwikkelen", "beheren", "ondersteunen", "werken", "werkt", "ontwerpen", "leiden",
        "implementeren", "coördineren", "assisteren", "analyseren", "stuur", "drijven",
        "samenwerken", "bijdragen", "communiceren", "betrekken", "bouwen",
    }
    # retrieve additional words from args
    for arg in args:
        if isinstance(arg, set):
            generic_words.update(arg)
        elif isinstance(arg, list):
            generic_words.update(set(arg))
        elif isinstance(arg, str):
            generic_words.add(arg)

    blacklist = generic_words.union(dutch_generic_words, verbs_en, verbs_nl)
    return blacklist

# ---------- Text Processing and N-gram Extraction ----------
def generate_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.
    :param tokens: List of tokens (words).
    :param n: The size of the n-grams to generate.
    :return: List of n-grams as strings.
    """
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def preprocess_and_extract_ngrams(text, stopwords, ngram_range):
    """
    Preprocess text and extract n-grams.
    :param text: Input text (job posting).
    :param stopwords: Set of stopwords to exclude.
    :param ngram_range: Tuple specifying the range of n-grams to extract (min_n, max_n).
    :return: Set of n-grams.
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = [t for t in text.split() if t not in stopwords and len(t) > 1] # Remove stopwords and short tokens
    ngram_set = set()
    for n in range(ngram_range[0], ngram_range[1]+1):
        ngrams = generate_ngrams(tokens, n)
        for ngram in ngrams:
            ngram_set.add(ngram)
    return ngram_set

# ---------- Main Function ----------
def extract_common_skills(job_postings, min_doc_freq=None, ngram_range=(1,2), *args):
    """
    Extract common skills from job postings using n-grams.
    :param job_postings: List of job postings (text).
    :param min_doc_freq: Minimum document frequency for n-grams to be considered common.
    :param ngram_range: Tuple specifying the range of n-grams to extract (min_n, max_n).
    :param args: Additional words to include in the blacklist (lists, sets, or strings).
    :return: Dictionary of common n-grams and their frequencies.
    """
    import nltk
    from collections import Counter
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    min_doc_freq = max(3, len(job_postings) // 10 if min_doc_freq is None else min_doc_freq)
    dutch_stopwords = set(stopwords.words('dutch'))
    # also dutch stop words
    stopwords = set(ENGLISH_STOP_WORDS).union(dutch_stopwords)
    blacklist = create_blacklist(*args)
    # join together the stopwords and blacklist
    stopwords = stopwords.union(blacklist)
    # Create a Counter to store document frequencies
    doc_freq = Counter()

    for posting in job_postings:
        ngram_set = preprocess_and_extract_ngrams(posting, stopwords, ngram_range)
        doc_freq.update(ngram_set)

    # Filter by doc frequency threshold
    common_ngrams = {term: freq for term, freq in doc_freq.items() if freq >= min_doc_freq}

    return dict(sorted(common_ngrams.items(), key=lambda item: item[1], reverse=True))