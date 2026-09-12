import requests
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0"


def collect_tass_news(user_agent, max_pages=1):
    base_url = "https://tass.ru/tbp/api/v1/search"
    headers = {"User-Agent": user_agent}
    extracted_data = []

    # Курсор пагинации(временная метка)
    search_after = ""

    print("Начинаем сбор данных...")

    for page in range(max_pages):
        # Параметры запроса
        params = {
            "search": "", # Берем все подряд
            "limit": 30,
            "lang": "ru"
        }

        # Если не первый круг - к параметрам добавляется курсор
        if search_after:
            params["search_after"] = search_after

        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Ошибка сервера: {response.status_code}. Останавливаем сбор.")
            break

        data = response.json()
        # Из ключа contens достаем список из 30 новостей
        news_list = data.get('result', {}).get('contents', [])
        # Проверка не дошли ли мы до конца истории
        has_more = data.get('result', {}).get('has_more', False)

        if not news_list:
            print("Новостей больше нет.")
            break

        for news in news_list:
            title = news.get('title', '').strip()
            if title:
                extracted_data.append({
                    'Headline': title,
                })

        last_news = news_list[-1]
        # Дата публикации/обновления последней новости из списка
        search_after = last_news.get('es_updated_dt') or last_news.get('published_dt', '')

        print(f"Шаг {page + 1}/{max_pages}: Собрано с ТАСС {len(news_list)} новостей. Всего в копилке: {len(extracted_data)}")

        if not has_more:
            print("Сервер сообщил, что достигнут конец базы данных.")
            break

        time.sleep(1)

    return extracted_data

def clean_headline(text):
    text = text.replace('\a0', ' ')
    pattern = r'\d{2}:\d{2},?\s*\d{1,2}\s+[а-яА-Яa-zA-Z]+\s+\d{4}.*'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text.strip()
def collect_lenta_news(user_agent, days_back=5):
    headers = {"User-Agent": user_agent}
    extracted_data = []
    base_date = datetime.now() - timedelta(days=1)
    print(f"Сбор архива Lenta.ru за последние {days_back} дней")
    step = 0
    for i in range(days_back):
        current_date = base_date - timedelta(days=i)
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')

        url = f"https://lenta.ru/{year}/{month}/{day}/"
        response = requests.get(url, headers=headers)
        if response.status_code!=200:
            print(f"Не удалось загрузить Lenta.ru, код ошибки: {response.status_code}")
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=lambda x: x and ('/news/' in x or '/articles/' in x))
        day_counter = 0

        for link in links:
            raw_title = link.get_text(strip=True)
            if raw_title and len(raw_title) > 15:
                # Очищаем заголовок от времени
                clean_title = clean_headline(raw_title)

                extracted_data.append({
                    'Headline': clean_title
                })
                day_counter += 1
        step+=1
        print(f"Шаг {step}/{days_back} собрано {len(extracted_data)} с Lenta.ru")
        time.sleep(1)
    df_temp = pd.DataFrame(extracted_data).drop_duplicates(subset=['Headline'])
    clean_data = df_temp.to_dict('records')

    print(f"\nСбор из архива завершен. Итого уникальных заголовков с Lenta.ru: {len(clean_data)}")
    return clean_data

def collect_rain_news(user_agent, max_pages=5):
    headers = {"User-Agent": user_agent}
    base_url = "https://tvrain.tv/news/"
    extracted_data = []
    current_date = datetime.today().strftime('%Y-%m-%d')
    current_page = 1
    for page in range(max_pages):
        params = {
            "gatherLabel": "news_buttonmore",
            "gatherCategory": "sp_newsfilter",
            "lastDate": current_date,
            "page": current_page,
            "filterTermMore": "",
            "loadmore": 1,
            "YII_CSRF_TOKEN": "a56691512e7085cd50ac26c4166f5de2edb8756a"
        }

        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"Код ошибки: {response.status_code}, проверьте актуальность токена")

        data = response.json()
        if data.get('lastDate'):
            current_date = data['lastDate']
        soup = BeautifulSoup(data['html'], 'html.parser')

        titles = [h3.find('a').get_text(strip=True) for h3 in soup.find_all('h3', class_='newsline_tile__headTitle') if
                  h3.find('a')]
        for title in titles:
            if title:
                extracted_data.append({'Headline': title})
        print(f"Шаг {page + 1}/{max_pages}: Собрано {len(titles)} новостей с Дождя. Текущая дата отсечки: {current_date}")

        current_page += 1
        time.sleep(1)
    return extracted_data

def save_to_combined_csv(new_data, filename='combined_news_dataset.csv'):
    if not new_data:
        print("Нет данных для записи.")
        return

    df_new = pd.DataFrame(new_data)

    file_exists = os.path.isfile(filename)

    df_new.to_csv(filename, mode='a', index=False, encoding='utf-8-sig', header=not file_exists)
    print(f"Данные успешно добавлены в файл {filename}.")

    # Глобальная очистка
    full_df = pd.read_csv(filename)
    initial_len = len(full_df)

    full_df = full_df.drop_duplicates(subset=['Headline'])
    full_df.to_csv(filename, index=False, encoding='utf-8-sig')

    print(
        f"Удалено дублей: {initial_len - len(full_df)}. Итого строк в базе: {len(full_df)}\n")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TARGET_FILE = os.path.join(DATA_DIR, "test_dataset.csv") if os.path.exists(DATA_DIR) else "test_dataset.csv"

    # Шаг 1. Собираем данные с ТАСС
    tass_data = collect_tass_news(USER_AGENT, max_pages=5)
    save_to_combined_csv(tass_data, filename=TARGET_FILE)

    # Шаг 2. Собираем данные с Ленты
    lenta_data = collect_lenta_news(USER_AGENT, days_back=5)
    save_to_combined_csv(lenta_data, filename=TARGET_FILE)

    # Шаг 3. Собираем данные с Дождя
    rain_data = collect_rain_news(USER_AGENT, max_pages=5)
    save_to_combined_csv(rain_data, filename=TARGET_FILE)

    print(f"Скрипт полностью отработал! Итоговый чистый датасет лежит в '{TARGET_FILE}'")