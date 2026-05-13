from datetime import datetime
import json
import logging
import ast
import re
import time
from core.config import config
from openai import OpenAI

log = logging.getLogger(__name__)

def get_news_sentiment(symbol, txt):
    try:
        client = OpenAI()

        # clean text by removing quotes
        txt = txt.replace('"', '').replace("'", "")
        # make sentiment request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, to the hundredths place, analyze the sentiment of the following news article summary using a range from -1 to 1 with less than or equal to -0.35 being bearish, greater than -0.35 but less than or equal to -0.15 being somewhat bearish, greater than -0.15 but less than 0.15 being neutral (but never return 0), greater than or equal to 0.15 but less than 0.35 being somewhat bullish, and greater than or equal to 0.35 being bullish; respond with only the number score. Input: '{txt}'"}
            ]
        )
        # response can be string or float
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d.\-]", "", content)  
        
        flt_content = float(sanitized_content)
        time.sleep(float(config.time_buffer))
        return flt_content
    except Exception:
        log.warning("warning: news sentiment request returned a non float")
        return None

def get_press_sentiment(symbol, txt):
    try:
        client = OpenAI()

        # clean text by removing quotes
        txt = txt.replace('"', '').replace("'", "")

        # make sentiment request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, to the hundredths place, analyze the sentiment of the following press release description using a range from -1 to 1 with less than or equal to -0.35 being bearish, greater than -0.35 but less than or equal to -0.15 being somewhat bearish, greater than -0.15 but less than 0.15 being neutral (but never return 0), greater than or equal to 0.15 but less than 0.35 being somewhat bullish, and greater than or equal to 0.35 being bullish; respond with only the number score. Input: '{txt}'"}
            ]
        )
        # response can be string or float
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d.\-]", "", content)  
        
        flt_content = float(sanitized_content)
        time.sleep(float(config.time_buffer))
        return flt_content
    except Exception:
        log.warning("warning: press sentiment request returned a non float")
        return None

def get_press_toneshift(symbol, txt):
    try:
        client = OpenAI()

        # clean text by removing quotes
        txt = txt.replace('"', '').replace("'", "")

        # make sentiment request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, analyze the sentiment shift in tone within the following financial press release. Output a number to the hundredths decimal place between -1 and 1, where: ≤ -0.35: strong downward shift in tone, -0.35 to ≤ -0.15: moderate downward shift, -0.15 to < 0.15: neutral tone (but never return 0), ≥ 0.15 to < 0.35: moderate upward shift, ≥ 0.35: strong upward shift. Evaluate based on changes in language that reflect optimism, pessimism, confidence, or caution. Respond only with the score and no explanation. Text: '{txt}'"}
            ]
        )
        # response can be string or float
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d.\-]", "", content)  
        
        flt_content = float(sanitized_content)
        time.sleep(float(config.time_buffer))
        return flt_content
    except Exception:
        log.warning("warning: press toneshift request returned a non float")
        return None
    
def get_press_origin(symbol, description, headline):
    try:
        client = OpenAI()

        description = (description or "").replace('"', "").replace("'", "")
        headline = (headline or "").replace('"', "").replace("'", "")

        message = f"""You are a financial assistant. Given the press release information below, determine whether it was issued directly by the company (first party) or by a third party (like a law firm or news outlet). 
            Return only one of the following (without quotes or explanation):
            first
            third

            Headline: {headline}  
            Description: {description}  
            Issuer: {symbol}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            temperature=0,
        )

        time.sleep(float(config.time_buffer))

        party = response.choices[0].message.content.strip().lower()

        if party in {"first", "third"}:
            return party
        else:
            log.warning(f"warning: gpt returned an invalid party response: {party}")
            return None
    except Exception as err:
        log.warning(f"warning: press origin returned invalid response: {err}")
        return None 

def get_press_schedule(symbol, description, headline, origin):
    try:
        client = OpenAI()

        description = (description or "").replace('"', "").replace("'", "")
        headline = (headline or "").replace('"', "").replace("'", "")

        message = f"""Given the press release information below, classify it into exactly one of the following categories:
            - scheduled → a regularly recurring or pre-announced event (e.g., earnings release, investor call, dividend declaration)
            - unscheduled → an unexpected or one-off corporate announcement (e.g., merger, leadership change, litigation, guidance update, product launch)
            - announcement → a notice about an upcoming event or release (e.g., announcing the date of a future earnings call or investor day)

            Return only one word:
            scheduled
            unscheduled
            announcement

            Headline: {headline}
            Description: {description}
            Issuer: {symbol}
            Source type: {origin} (first-party or third-party)"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": message}],
            temperature=0,
        )

        time.sleep(float(config.time_buffer))

        schedule = response.choices[0].message.content.strip().lower()
        if schedule in {"scheduled", "unscheduled", "announcement"}:
            return schedule
        else:
            log.warning(f"warning: gpt returned an invalid schedule response: {schedule}")
            return None
    except Exception as err:
        log.warning(f"warning: press schedule returned invalid response: {err}")
        return None  

def get_earnings_toneshift(symbol, txt):
    try:
        client = OpenAI()

        # clean text by removing quotes
        txt = txt.replace('"', '').replace("'", "")

        # make sentiment request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, evaluate the change in tone within the following earnings call excerpt using a scale from -1.00 to 1.00 to two decimal places. Assess whether the tone shows increasing confidence, stability, or uncertainty. Return only a numerical score based on this scale: ≤ -0.35: clear negative momentum in tone, -0.35 to ≤ -0.15: slightly negative momentum, -0.15 to < 0.15: neutral tone (but never return 0), ≥ 0.15 to < 0.35: slightly positive momentum, ≥ 0.35: clear positive momentum. Respond only with the score. Do not include explanations. Input: '{txt}'"}
            ]
        )
        # response can be string or float
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d.\-]", "", content)  
        
        flt_content = float(sanitized_content)
        time.sleep(float(config.time_buffer))
        return flt_content
    except Exception:
        log.warning("warning: earnings toneshift request returned a non float")
        return None

def get_earnings_sentiment(symbol, txt):
    try:
        client = OpenAI()

        # clean text by removing quotes
        txt = txt.replace('"', '').replace("'", "")

        # make sentiment request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, to the hundredths place, analyze the sentiment of the following earnings call transcript using a range from -1 to 1 with less than or equal to -0.35 being bearish, greater than -0.35 but less than or equal to -0.15 being somewhat bearish, greater than -0.15 but less than 0.15 being neutral (but never return 0), greater than or equal to 0.15 but less than 0.35 being somewhat bullish, and greater than or equal to 0.35 being bullish; respond with only the number score. Input: '{txt}'"}
            ]
        )
        # response can be string or float
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d.\-]", "", content)  
        
        flt_content = float(sanitized_content)
        time.sleep(float(config.time_buffer))
        return flt_content
    except Exception:
        log.warning("warning: earnings sentiment request returned a non float")
        return None
    

def get_press_announcement_date(symbol, headline, url):
    try:
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"user", "content":f"Regarding this stock ticker: {symbol}, what is the announcement date from the following press release text. Return the date in YYYY-MM-DD format. If no date is found, return 'unknown'. Do not include any other text. Headline: {headline} URL: {url}"}
            ]
        )
    
        # clean response - remove white spaces and invalid chars
        content = response.choices[0].message.content.strip()  
        sanitized_content = re.sub(r"[^\d\-\w]", "", content)  
        log.debug(f"sanitized announcement date content: {sanitized_content}")
        
        if re.match(r"\d{4}-\d{2}-\d{2}", sanitized_content):
            time.sleep(float(config.time_buffer))
            return datetime.strptime(sanitized_content, '%Y-%m-%d')
        else:
            log.warning(f"warning: gpt returned an invalid date response: {sanitized_content}")
            return None
    except Exception as err:
        log.warning(f"warning: press announcement date returned invalid response: {err}")
        return None