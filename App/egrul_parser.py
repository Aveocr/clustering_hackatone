import json
from urllib.parse import urljoin
from dataclasses import dataclass
from datetime import datetime
from time import sleep

from requests.exceptions import RequestException, JSONDecodeError
import requests

BASE_URL =  'https://egrul.nalog.ru'
GET_TOKEN_PATH = '/'
SEARCH_PATH = '/search-result/'


@dataclass
class SearchResult:
    ip_start_date: datetime
    ip_end_date: datetime


class ParseError(BaseException):
    def __init__(self, inn, msg):
        self.__inn = inn
        self.__msg = msg
    
    def __str__(self):
        return f'INN: {self.__inn}; {self.__msg}'


def search_data(query: str) -> list[SearchResult]:
    token_url = urljoin(BASE_URL, GET_TOKEN_PATH)
    token_form_data = {
        'query': query
    }

    try:
        res = requests.post(token_url, token_form_data)
    except RequestException as e:
        raise ParseError(query, f'Request error: {e}')

    try:
        j = res.json()
    except JSONDecodeError as e:
        raise ParseError(query, f'JSON parse error: {e}')

    try:
        token = j['t']
    except KeyError:
        raise ParseError(query, f'Unable to find \"t\" field: {json.dumps(j, indent=4)}')

    search_url = urljoin(BASE_URL, SEARCH_PATH + token)
    
    while True:
        try:
            res = requests.get(search_url)
        except RequestException as e:
            raise ParseError(query, f'Request error: {e}')
        
        try:
            j = res.json()
        except JSONDecodeError as e:
            raise ParseError(query, f'JSON parse error: {e}')

        if 'status' in j and j['status'] == 'wait':
            sleep(2)
            continue
        
        try:
            results = []
            for data in j['rows']:
                if 'r' not in data:
                    continue
                
                results.append(SearchResult(
                    ip_start_date=datetime.strptime(data['r'], '%d.%m.%Y'),
                    ip_end_date=datetime.strptime(data['e'], '%d.%m.%Y') \
                        if 'e' in data else None
                ))

            return results
        except (ValueError, KeyError):
            raise ParseError(query, f'Unable to parse JSON: {json.dumps(j, indent=4)}')
