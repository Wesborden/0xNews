# -*- coding: utf-8 -*-
import telebot  # подключение телеграмм бота
import feedparser  # парсинг RSS лент
import requests  # загружает HTML страницу с сайта
import os  # работа с значениями из переменных окружения
import re  # удаление ненужных HTML тегов
import time  # работа с временем обработки
import threading  # использовал для автоотправки сообщений, минуя команды (не до конца уверен как работает(?))
import json  # работа с JSON(JavaScript Object Notation) файлами
import random  # рандомайзер
import queue  # очередь
import logging  # логирование
import uuid
import json as _json
import difflib

from bs4 import BeautifulSoup  # парсит HTML страницу и забирает от туда данные
from dotenv import load_dotenv  # загрузка переменных .env в память
from selenium import webdriver  # user-agent бот, для имитации открытия браузера человеком
from urllib.parse import urljoin  # исправление муссорной ссылки на рабочую, кликабельную
from collections import deque  # дэк для записи заголовков
from xml.etree import ElementTree as ET
from urllib.parse import urlparse
from typing import List, Dict, Optional
from typing import Tuple, Any

load_dotenv()  # загружает переменные из .env

# конфигурации телеграмма
TOKEN = os.getenv("TOKEN")  # присваивание токена бота из переменной окружения
CHAT_ID = int(os.getenv("CHAT_ID"))  # присваивание ID чата
THREAD_ID = int(os.getenv("THREAD_ID"))  # присваивание ID темы в чате
PINNED_ID = int(os.getenv("PINNED_MESSAGE_ID"))  # ID закрепленного сообщения
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))  # ID канала
LINK = f"<a href=\"{os.getenv('LINK')}\">\n@0xnews</a>"

# конфигурации API, URL, интервал
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # API ключ мистрала
MISTRAL_API_URL = os.getenv("MISTRAL_API_URL")  # URL мистрала
INTERVAL_CHECK = int(os.getenv("INTERVAL_CHECK"))  # интервал проверка RSS
URL_CG = os.getenv("URL_CG")  # CoinGecko конечная точка
FEARGREED_API = os.getenv("FEARGREED_API")  # апи страха и жадности
PINNED_INTERVAL_CHECK = int(os.getenv("PINNED_INTERVAL_CHECK"))  # интервал обновления закрепа

# Константы и lock для сортировки по смыслу
SEMANTIC_MEMORY_FILE = os.getenv("SEMANTIC_MEMORY_FILE")
SEMANTIC_LOCK = os.getenv("SEMANTIC_LOCK")
SEMANTIC_COMPARE_LIMIT = int(os.getenv("SEMANTIC_COMPARE_LIMIT"))
SEMANTIC_THRESHOLD_DEFAULT = float(os.getenv("SEMANTIC_THRESHOLD_DEFAULT"))

# список RSS источников
html_urls = [
    "https://www.coindesk.com/arc/outboundfeeds/rss",
    "https://cointelegraph.com/rss",
    "https://finance.yahoo.com/news/rssindex",
    "https://thedefiant.io/api/feed",
    "https://smartliquidity.info/feed/"

]

post_queue = queue.Queue()
post_list = []
json_url_queue = "queue_json.json"
post_list_lock = threading.Lock()  # добавлено для защиты post_list

bot = telebot.TeleBot(TOKEN, parse_mode = "HTML")  # создание телеграмм бота

logging.basicConfig(
    level=logging.INFO,  # уровень логирования (DEBUG, INFO, WARNING и т.д.)
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),  # логи в файл
        logging.StreamHandler()  # одновременно логи в консоль
    ]
)

logger = logging.getLogger(__name__)

# Загрузка post_list из JSON при старте
if os.path.exists(json_url_queue):
    with open(json_url_queue, "r", encoding="utf-8") as f:
        post_list = json.load(f)
        logger.info(f"Loaded {len(post_list)} posts from file")
    for post in post_list:
        post_queue.put(post)
else:
    post_list = []


def get_page_text(url, retries=3):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/138.0.0.0 Safari/537.36"}
        
        for attempt in range(retries):
            response = requests.get(url, headers=headers, timeout=30)

            if 500 <= response.status_code <= 600:
                    logger.error(f"To many requests. Pause 1 minute before repeating (attempt: {attempt + 1})")
                    time.sleep(60)  # пауза перед повтором 
                    continue

            response.raise_for_status()  # ошибка если статус != 200

            # Парсим HTML и вытаскиваем текст
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.extract()  # удаляем js/css

            raw_text = soup.get_text(separator="\n")
            lines = [line.strip() for line in raw_text.splitlines() if line.strip]
            full_text = "\n".join(lines)

            time.sleep(15)

            return full_text[:20000]  # обрезаем до 4000 символов
        
        logger.error(f"Error 429: Too many requests. No more retries.")
        return None
    
    except Exception as e:
        logger.error(f"Error loading {url}: {e}")
        return None
    
def _first_child_text(elem: ET.Element, names: List[str]) -> Optional[str]:
    for child in elem:
        if child.tag is None:
            continue
        if child.tag.split("}")[-1] in names:
            return (child.text or "").strip()
    return None


def _extract_link(elem: ET.Element) -> Optional[str]:
    # 1) простой <link>text</link>
    text = _first_child_text(elem, ["link"])
    if text and text.startswith("http"):
        return text
    # 2) Atom style <link href="..."/>
    for child in elem:
        if child.tag.split("}")[-1] == "link":
            href = child.attrib.get("href") or child.attrib.get("url")
            if href:
                return href
    return None


def _extract_image(elem: ET.Element, content_html: Optional[str]) -> Optional[str]:
    for child in elem:
        local = child.tag.split("}")[-1]
        if local in ("content", "thumbnail", "enclosure"):
            url = child.attrib.get("url") or child.attrib.get("href") or child.attrib.get("src")
            if url and url.startswith("http"):
                return url
    if content_html:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content_html, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _get_content_text(elem: ET.Element) -> str:
    for child in elem:
        local = child.tag.split("}")[-1]
        if local in ("encoded", "content", "description", "summary"):
            return (child.text or "").strip()
    return ""


def _call_mistral(prompt: str, model: str = "mistral-medium-latest", timeout: int = 30) -> Optional[str]:
    if not MISTRAL_API_KEY or not MISTRAL_API_URL:
        logger.error("Mistral API config is missing.")
        return None
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500
    }
    try:
        r = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # совместим с форматом: choices[0].message.content или choices[0].text
        content = None
        if isinstance(data.get("choices"), list) and data["choices"]:
            ch0 = data["choices"][0]
            # разные реализации: message.content или text
            content = (ch0.get("message", {}) .get("content") if ch0.get("message") else ch0.get("text"))
        if not content:
            # возможный иной формат
            content = data.get("output") or None
        return content.strip() if content else None
    except requests.RequestException as e:
        logger.error("Mistral request failed: %s", str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected Mistral response handling error: %s", str(e))
        return None
    
def load_semantic_memory() -> list:
    try:
        if os.path.exists(SEMANTIC_MEMORY_FILE):
            with open(SEMANTIC_MEMORY_FILE, "r", encoding="utf-8") as f:
                return _json.load(f)
    except Exception as e:
        logger.warning("Failed to load semantic memory: %s", e)
    return []

def save_semantic_memory(mem: list) -> None:
    try:
        with SEMANTIC_LOCK:
            with open(SEMANTIC_MEMORY_FILE, "w", encoding="utf-8") as f:
                _json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to save semantic memory: %s", e)

def _mistral_compare(new_title: str, new_text: str, reps: list) -> Tuple[bool, int, float, Any]:
    # ограничиваем количество сравнений
    limited_reps = reps[-SEMANTIC_COMPARE_LIMIT:]
    # формируем компактный, строгий prompt (только JSON в ответ)
    enumerated = "\n".join([f"{i+1}. {r}" for i, r in enumerate(limited_reps)]) if limited_reps else "[]"
    prompt = f"""
        You are a strict semantic comparator. Input:
        NEW_TITLE: {new_title}
        NEW_TEXT: {new_text[:4000]}

        EXISTING_TITLES:
        {enumerated}

        Task: compare NEW_TITLE+NEW_TEXT to each EXISTING_TITLES by meaning (not by wording). If any existing title has the same news meaning (i.e. the same event/topic that would make them duplicates in a newsfeed), return a JSON object only, no extra text, with fields:
        {{"match": true/false, "index": <1-based index of matched EXISTING_TITLES if match, otherwise null>, "score": <similarity 0.0-1.0>, "reason": "<one-sentence explanation>"}}.
        Score must be between 0.0 and 1.0. Use a high threshold for clear duplicates; but do not invent matches. Output strictly JSON.
        """.replace('"', '\\"')
    try:
        resp = _call_mistral(prompt, model="mistral-medium-latest", timeout=30)
        if not resp:
            return False, None, 0.0, None 
        # постобработка: пытаемся распарсить JSON
        try:
            parsed = _json.loads(resp)
            match = bool(parsed.get("match"))
            idx = parsed.get("index")
            score = float(parsed.get("score") or 0.0)
            return match, (int(idx)-1 if idx is not None else None), score, parsed
        except Exception:
            # если Mistral не вернул правильно сформированный JSON — возвращаем raw
            logger.warning("Mistral compare returned non-json: %s", resp[:500])
            return False, None, 0.0, resp
    except Exception as e:
        logger.warning("Mistral compare error: %s", e)
        return False, None, 0.0, None

def _local_fallback_compare(new_title: str, new_text: str, reps: list) -> Tuple[bool, int, float]:
    best_score = 0.0
    best_idx = None
    new_text_short = (new_text or "")[:2000]
    for i, rep in enumerate(reps[-SEMANTIC_COMPARE_LIMIT:]):
        # сравнение заголовков
        title_score = difflib.SequenceMatcher(None, new_title, rep).ratio()
        # сравнение текстов (если есть)
        text_score = difflib.SequenceMatcher(None, new_text_short, rep).ratio()  # реп — заголовок; слабый, но всё же
        # агрегируем: даём больший вес заголовку
        combined = 0.7 * title_score + 0.3 * text_score
        if combined > best_score:
            best_score = combined
            best_idx = i
    match = best_score >= SEMANTIC_THRESHOLD_DEFAULT
    return match, best_idx, best_score

def is_semantic_duplicate(title: str, page_text: str, link: str = "", threshold: float = SEMANTIC_THRESHOLD_DEFAULT) -> bool:
    with SEMANTIC_LOCK:
        mem = load_semantic_memory()
        reps = [c.get("representative") for c in mem if c.get("representative")]
    
    # если память пуста — создаём первый кластер
    if not reps:
        new_cluster = {
            "id": uuid.uuid4().hex,
            "representative": title,
            "titles": [title],
            "created_at": int(time.time()),
            "links": [link] if link else []
        }
        with SEMANTIC_LOCK:
            mem.append(new_cluster)
            save_semantic_memory(mem)
        logger.info("Semantic memory empty — created first cluster for: %s", title)
        return False

    # пытаемся через Mistral
    match, idx, score, raw = _mistral_compare(title, page_text, reps)
    if match and idx is not None and score >= threshold:
        logger.info("Mistral detected semantic duplicate (score=%.3f) vs rep idx %s: %s", score, idx, reps[idx])
        # добавляем заголовок в кластер, если его там нет
        with SEMANTIC_LOCK:
            if title not in mem[idx]["titles"]:
                mem[idx]["titles"].append(title)
                if link:
                    mem[idx].setdefault("links", []).append(link)
                save_semantic_memory(mem)
        return True

    # fallback: локальное сравнение
    fallback_match, fallback_idx, fallback_score = _local_fallback_compare(title, page_text, reps)
    if fallback_match and fallback_score >= threshold:
        logger.info("Local fallback marked duplicate (score=%.3f) vs rep idx %s: %s", fallback_score, fallback_idx, reps[fallback_idx])
        with SEMANTIC_LOCK:
            if title not in mem[fallback_idx]["titles"]:
                mem[fallback_idx]["titles"].append(title)
                if link:
                    mem[fallback_idx].setdefault("links", []).append(link)
                save_semantic_memory(mem)
        return True

    # не дубль — создаём новый кластер
    new_cluster = {
        "id": uuid.uuid4().hex,
        "representative": title,
        "titles": [title],
        "created_at": int(time.time()),
        "links": [link] if link else []
    }
    with SEMANTIC_LOCK:
        mem.append(new_cluster)
        # возможно полезно ограничить длину файла — оставляем последние N кластеров, но это на ваше усмотрение
        save_semantic_memory(mem)
    logger.info("New semantic cluster created for: %s (score_mistral=%s, fallback_score=%.3f)", title, score if 'score' in locals() else None, fallback_score if 'fallback_score' in locals() else 0.0)
    return False

def thedefiant_parsing() -> Optional[List[Dict]]:
    try:
        url = "https://thedefiant.io/api/feed"  # поменяйте тут, если нужно другой feed
        limit = 10                       # сколько элементов брать
        sort_post: List[Dict] = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/atom+xml, text/xml, */*"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        parsed = urlparse(url)
        thedefiant_memory = f"{parsed.netloc}{parsed.path}".replace("/", "_").strip("_") + ".json"

        # load seen titles
        try:
            with open(thedefiant_memory, "r", encoding="utf-8") as f:
                seen = deque(json.load(f), maxlen=300)
        except Exception:
            seen = deque(maxlen=300)

        # find candidates: RSS <item> or Atom <entry>
        candidates = list(root.findall(".//item")) or list(root.findall(".//{http://www.w3.org/2005/Atom}entry")) or list(root.findall(".//entry"))
        if not candidates:
            # fallback: собрать все элементы с title+link
            candidates = [el for el in root.iter() if el.tag.split("}")[-1] in ("item", "entry")]

        for elem in candidates[:limit]:
            title = _first_child_text(elem, ["title"]) or ""
            title = title.strip()
            if not title:
                # попытка по текстовым узлам
                texts = [c.text for c in elem.iter() if c.text and len(c.text.strip()) > 5]
                title = texts[0].strip() if texts else ""
            if not title:
                logger.info("Skip element without title")
                continue

            if title in seen:
                logger.info("Skip duplicate: %s", title)
                continue

            link = _extract_link(elem) or ""
            content_html = _get_content_text(elem)
            image = _extract_image(elem, content_html)

            # Получаем полный текст через вашу функцию (соблюдает свои таймауты)
            page_text = None
            if link:
                try:
                    page_text = get_page_text(link)
                except Exception as e:
                    logger.warning("get_page_text failed for %s: %s", link, str(e))
                    page_text = None

            # Формируем prompt для Mistral — компактно, но информативно
            prompt = f'''
                You are a news filter for the crypto market. Use this information for analysis: {page_text}. Your task is to assign each news item one of three priorities—HIGH or LOW—based on its impact on the market. Follow these steps precisely:

            **Step 1: Analyze the News**

            - Make a short summary of the content, key data, and related themes.
            - Review the entire source, focusing on financial news, macroeconomic data, and events affecting the economy.
            - Identify themes tied to the criteria below.

            **Step 2: Assign Priority**

            - HIGH: News with direct, significant impact on the crypto market.
            - LOW: News creating a general informational or emotional background, without direct impact or unrelated to markets or with minimal impact.

            **Priority Criteria:**
            HIGH Priority

            Assign HIGH if the news includes any of these:

            Macroeconomic & Regulatory:

            - Decisions or statements from the US Fed, ECB, People’s Bank of China, or Bank of Japan on interest rates, inflation, or liquidity. For example, a statement from the US Fed about raising interest rates.
            - Crypto regulation launches or bans in the USA, EU, China, Hong Kong, or UAE. For example, the EU banning a specific cryptocurrency.
            - SEC or CFTC rulings on major crypto exchange cases (e.g., Ripple, Binance, Coinbase). For example, the SEC ruling against Ripple.

            Investments & Finance:

            - Approval or launch of Bitcoin-ETFs or other crypto-ETFs in major jurisdictions. For example, the approval of a Bitcoin-ETF in the USA.

            Partnerships & Integrations:

            - Collaborations between crypto projects and companies like Meta, Apple, Amazon, Google, or Microsoft. For example, Meta partnering with a crypto project to integrate blockchain technology.
            - Crypto or blockchain integration into payment systems like PayPal, Visa, or Mastercard. For example, Visa integrating a cryptocurrency into its payment system.

            Key Figures:

            - Public statements or tweets from Elon Musk, Jerome Powell, Gary Gensler, Changpeng Zhao, Vitalik Buterin, or similar influential figures. For example, Elon Musk tweeting about a cryptocurrency.

            Do not assign HIGH if the news is:

            - Predictions without evidence (e.g., “expert says prices will rise”). For example, an analyst predicting that Bitcoin will reach $100,000 without providing evidence.
            - Local regulatory news from minor countries (outside USA, EU, China). For example, a small country in Africa banning cryptocurrency.
            - Unverified rumors or social media speculation. For example, a rumor on Twitter about a major crypto exchange being hacked.

            LOW Priority

            Assign LOW if the news includes any of these:

            Technological Events:

            - Hard forks, soft forks, or testnet launches in top-50 crypto projects by market cap. For example, a hard fork in Ethereum.
            - Launch of new blockchain protocols. For example, the launch of a new blockchain protocol for decentralized finance.
            - Integration of AI, IoT, or Web3 into the crypto ecosystem. For example, a crypto project integrating AI to improve its services.

            Second-Tier Companies:

            - Partnerships or collaborations involving crypto firms. For example, two crypto firms partnering to develop a new product.
            - Launches of exchanges, wallets, or DeFi applications. For example, the launch of a new decentralized exchange.

            Reports & Lesser Markets:

            - Analytical publications from Messari, Glassnode, CoinShares, or similar agencies. For example, a report from Messari on the state of the crypto market.
            - Regulatory news from less significant regions (e.g., South America, Africa). For example, a country in South America introducing new crypto regulations.
            - Vague, lacking specifics (e.g., “planned for the future”). For example, a crypto project announcing plans to launch a new product without specifying when.
            - Airdrop promotions, contests, or marketing campaigns. For example, a crypto project announcing an airdrop to promote its token.
            - Tech-related but unconnected to crypto. For example, a news item about a new smartphone without any mention of crypto.

            Low-Capitalization Events:

            - NFT drops from obscure projects. For example, an unknown artist launching an NFT collection.
            - Token launches with market cap below $10 million. For example, a new token with a market cap of $5 million.
            - Fraud or scams involving minor projects. For example, a small crypto project being accused of fraud.

            Crypto Culture:

            - Memes or social media disputes among regular users. For example, a meme about a cryptocurrency going viral on Twitter.
            - News about influencers with negligible market influence. For example, a minor influencer endorsing a cryptocurrency.

            Unrelated or Minor:

            - Tech news without crypto impact. For example, a news item about a new programming language.
            - Small-scale regional events in insignificant jurisdictions. For example, a local crypto meetup in a small town.
            - Repetitions of prior news without new details. For example, a news item repeating information from a previous news item.
            - PR materials or press releases lacking substance. For example, a press release from a crypto project announcing a new partnership without providing details.
            - Texts not mentioning cryptocurrencies, blockchain, or financial markets. For example, a news item about a new movie.
            - Speculative or opinion-based articles without substantial data or market impact.
            - Interrogative sentences and questions.
            - General (NOT SPECIFIC) information without definite data.

            Whales

            - Whales transactions
            - Large investments

            **Step 3: Output**

            - First, provide a concise 2-3 sentence summary of the news and its key topics.
            - Then, give a brief analysis of the news and themes to determine priority.
            - State the priority level on a separate line using the word PRIORITY and colon «:».

            Avoid introductory phrases, labels, or explanations.

            Example:Solana launches new scalable DeFi protocol. Solana unveiled Solana Pay V2, a protocol enhancing DeFi applications with 30% lower transaction fees and AI-optimized smart contracts, set for release in August 2025. This development strengthens Solana’s position against competitors like Ethereum by improving scalability and user experience. The news, while significant for DeFi and technological innovation, lacks the broader market impact of regulatory changes or major investments. #Solana #DeFi

            The news qualifies as LOW priority due to its focus on a new blockchain protocol and AI integration, aligning with technological event criteria, but it does not involve regulatory decisions, large-scale investments, or influential figures required for HIGH priority. Improvements to the prompt include replacing "critical clarity" with "quantifiable data not concisely summarized," specifying news prioritization by market or adoption impact, adding a language instruction like "Use English unless specified," and including an example without a list for clarity.

            [PRIORITY: LOW]

            MAXIMUM : 100 words
            '''
            mistral_resp = _call_mistral(prompt)
            if not mistral_resp:
                logger.warning("No response from Mistral for: %s", title)
                # добавляем в seen чтобы не запрашивать снова сразу
                seen.append(title)
                continue

            # извлекаем приоритет
            m = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", mistral_resp.upper())
            priority = m.group(1) if m else None

            # для логики — берем только HIGH (как в вашем примере), можно расширить
            if priority == "HIGH":
                # семантическая дедупликация
                try:
                    is_dup = is_semantic_duplicate(title, page_text or content_html or "", link=link)
                except Exception as e:
                    logger.warning("Semantic duplicate check failed for %s: %s", title, e)
                    is_dup = False  # не блокируем постинг из-за ошибки дедупликации

                if is_dup:
                    logger.info("Skipped semantic duplicate: %s", title)
                else:
                    sort_post.append({
                        "title": title,
                        "link": link,
                        "image": image,
                        "page_text": (page_text or content_html)[:20000],
                        "mistral_raw": mistral_resp
                    })
                    logger.info("HIGH added (unique): %s", title)

            # всегда записываем заголовок в seen (чтобы не обрабатывать повторно)
            seen.append(title)

            # небольшой пауз — get_page_text уже содержит паузу, но здесь дополнительная защита от rate limit
            time.sleep(15)

        # persist seen
        try:
            with open(thedefiant_memory, "w", encoding="utf-8") as f:
                json.dump(list(seen), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write seen-file %s: %s", thedefiant_memory, str(e))

        logger.info("Collected %d HIGH items from %s", len(sort_post), url)
        return sort_post

    except requests.RequestException as e:
        logger.error("HTTP error for feed %s: %s", url, str(e))
        return None
    except ET.ParseError as e:
        logger.error("XML parse error for %s: %s", url, str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected error parsing feed %s: %s", url, str(e))
        return None
    

def smartliquidity_parsing() -> Optional[List[Dict]]:
    try:
        url = "https://smartliquidity.info/feed/"
        limit = 10                       # сколько элементов брать
        sort_post: List[Dict] = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/atom+xml, text/xml, */*"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        parsed = urlparse(url)
        smartliquidity_memory = f"{parsed.netloc}{parsed.path}".replace("/", "_").strip("_") + ".json"

        # load seen titles
        try:
            with open(smartliquidity_memory, "r", encoding="utf-8") as f:
                seen = deque(json.load(f), maxlen=300)
        except Exception:
            seen = deque(maxlen=300)

        # find candidates: RSS <item> or Atom <entry>
        candidates = list(root.findall(".//item")) or list(root.findall(".//{http://www.w3.org/2005/Atom}entry")) or list(root.findall(".//entry"))
        if not candidates:
            # fallback: собрать все элементы с title+link
            candidates = [el for el in root.iter() if el.tag.split("}")[-1] in ("item", "entry")]

        for elem in candidates[:limit]:
            title = _first_child_text(elem, ["title"]) or ""
            title = title.strip()
            if not title:
                # попытка по текстовым узлам
                texts = [c.text for c in elem.iter() if c.text and len(c.text.strip()) > 5]
                title = texts[0].strip() if texts else ""
            if not title:
                logger.info("Skip element without title")
                continue

            if title in seen:
                logger.info("Skip duplicate: %s", title)
                continue

            link = _extract_link(elem) or ""
            content_html = _get_content_text(elem)

            # Получаем полный текст через вашу функцию (соблюдает свои таймауты)
            page_text = None
            if link:
                try:
                    page_text = get_page_text(link)
                except Exception as e:
                    logger.warning("get_page_text failed for %s: %s", link, str(e))
                    page_text = None

            # Формируем prompt для Mistral — компактно, но информативно
            prompt = f'''
                You are a news filter for the crypto market. Use this information for analysis: {page_text}. Your task is to assign each news item one of three priorities—HIGH or LOW—based on its impact on the market. Follow these steps precisely:

            **Step 1: Analyze the News**

            - Make a short summary of the content, key data, and related themes.
            - Review the entire source, focusing on financial news, macroeconomic data, and events affecting the economy.
            - Identify themes tied to the criteria below.

            **Step 2: Assign Priority**

            - HIGH: News with direct, significant impact on the crypto market.
            - LOW: News creating a general informational or emotional background, without direct impact or unrelated to markets or with minimal impact.

            **Priority Criteria:**
            HIGH Priority

            Assign HIGH if the news includes any of these:

            Macroeconomic & Regulatory:

            - Decisions or statements from the US Fed, ECB, People’s Bank of China, or Bank of Japan on interest rates, inflation, or liquidity. For example, a statement from the US Fed about raising interest rates.
            - Crypto regulation launches or bans in the USA, EU, China, Hong Kong, or UAE. For example, the EU banning a specific cryptocurrency.
            - SEC or CFTC rulings on major crypto exchange cases (e.g., Ripple, Binance, Coinbase). For example, the SEC ruling against Ripple.

            Investments & Finance:

            - Approval or launch of Bitcoin-ETFs or other crypto-ETFs in major jurisdictions. For example, the approval of a Bitcoin-ETF in the USA.

            Partnerships & Integrations:

            - Collaborations between crypto projects and companies like Meta, Apple, Amazon, Google, or Microsoft. For example, Meta partnering with a crypto project to integrate blockchain technology.
            - Crypto or blockchain integration into payment systems like PayPal, Visa, or Mastercard. For example, Visa integrating a cryptocurrency into its payment system.

            Key Figures:

            - Public statements or tweets from Elon Musk, Jerome Powell, Gary Gensler, Changpeng Zhao, Vitalik Buterin, or similar influential figures. For example, Elon Musk tweeting about a cryptocurrency.

            Do not assign HIGH if the news is:

            - Predictions without evidence (e.g., “expert says prices will rise”). For example, an analyst predicting that Bitcoin will reach $100,000 without providing evidence.
            - Local regulatory news from minor countries (outside USA, EU, China). For example, a small country in Africa banning cryptocurrency.
            - Unverified rumors or social media speculation. For example, a rumor on Twitter about a major crypto exchange being hacked.

            LOW Priority

            Assign LOW if the news includes any of these:

            Technological Events:

            - Hard forks, soft forks, or testnet launches in top-50 crypto projects by market cap. For example, a hard fork in Ethereum.
            - Launch of new blockchain protocols. For example, the launch of a new blockchain protocol for decentralized finance.
            - Integration of AI, IoT, or Web3 into the crypto ecosystem. For example, a crypto project integrating AI to improve its services.

            Second-Tier Companies:

            - Partnerships or collaborations involving crypto firms. For example, two crypto firms partnering to develop a new product.
            - Launches of exchanges, wallets, or DeFi applications. For example, the launch of a new decentralized exchange.

            Reports & Lesser Markets:

            - Analytical publications from Messari, Glassnode, CoinShares, or similar agencies. For example, a report from Messari on the state of the crypto market.
            - Regulatory news from less significant regions (e.g., South America, Africa). For example, a country in South America introducing new crypto regulations.
            - Vague, lacking specifics (e.g., “planned for the future”). For example, a crypto project announcing plans to launch a new product without specifying when.
            - Airdrop promotions, contests, or marketing campaigns. For example, a crypto project announcing an airdrop to promote its token.
            - Tech-related but unconnected to crypto. For example, a news item about a new smartphone without any mention of crypto.

            Low-Capitalization Events:

            - NFT drops from obscure projects. For example, an unknown artist launching an NFT collection.
            - Token launches with market cap below $10 million. For example, a new token with a market cap of $5 million.
            - Fraud or scams involving minor projects. For example, a small crypto project being accused of fraud.

            Crypto Culture:

            - Memes or social media disputes among regular users. For example, a meme about a cryptocurrency going viral on Twitter.
            - News about influencers with negligible market influence. For example, a minor influencer endorsing a cryptocurrency.

            Unrelated or Minor:

            - Tech news without crypto impact. For example, a news item about a new programming language.
            - Small-scale regional events in insignificant jurisdictions. For example, a local crypto meetup in a small town.
            - Repetitions of prior news without new details. For example, a news item repeating information from a previous news item.
            - PR materials or press releases lacking substance. For example, a press release from a crypto project announcing a new partnership without providing details.
            - Texts not mentioning cryptocurrencies, blockchain, or financial markets. For example, a news item about a new movie.
            - Speculative or opinion-based articles without substantial data or market impact.
            - Interrogative sentences and questions.
            - General (NOT SPECIFIC) information without definite data.

            Whales

            - Whales transactions
            - Large investments

            **Step 3: Output**

            - First, provide a concise 2-3 sentence summary of the news and its key topics.
            - Then, give a brief analysis of the news and themes to determine priority.
            - State the priority level on a separate line using the word PRIORITY and colon «:».

            Avoid introductory phrases, labels, or explanations.

            Example:Solana launches new scalable DeFi protocol. Solana unveiled Solana Pay V2, a protocol enhancing DeFi applications with 30% lower transaction fees and AI-optimized smart contracts, set for release in August 2025. This development strengthens Solana’s position against competitors like Ethereum by improving scalability and user experience. The news, while significant for DeFi and technological innovation, lacks the broader market impact of regulatory changes or major investments. #Solana #DeFi

            The news qualifies as LOW priority due to its focus on a new blockchain protocol and AI integration, aligning with technological event criteria, but it does not involve regulatory decisions, large-scale investments, or influential figures required for HIGH priority. Improvements to the prompt include replacing "critical clarity" with "quantifiable data not concisely summarized," specifying news prioritization by market or adoption impact, adding a language instruction like "Use English unless specified," and including an example without a list for clarity.

            [PRIORITY: LOW]

            MAXIMUM : 100 words
            '''
            mistral_resp = _call_mistral(prompt)
            if not mistral_resp:
                logger.warning("No response from Mistral for: %s", title)
                # добавляем в seen чтобы не запрашивать снова сразу
                seen.append(title)
                continue

            # извлекаем приоритет
            m = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", mistral_resp.upper())
            priority = m.group(1) if m else None

            # для логики — берем только HIGH (как в вашем примере), можно расширить
            if priority == "HIGH":
                # семантическая дедупликация
                try:
                    is_dup = is_semantic_duplicate(title, page_text or content_html or "", link=link)
                except Exception as e:
                    logger.warning("Semantic duplicate check failed for %s: %s", title, e)
                    is_dup = False  # не блокируем постинг из-за ошибки дедупликации

                if is_dup:
                    logger.info("Skipped semantic duplicate: %s", title)
                else:
                    sort_post.append({
                        "title": title,
                        "link": link,
                        "page_text": (page_text or content_html)[:20000],
                        "mistral_raw": mistral_resp
                    })
                    logger.info("HIGH added (unique): %s", title)

            # всегда записываем заголовок в seen (чтобы не обрабатывать повторно)
            seen.append(title)

            # небольшой пауз — get_page_text уже содержит паузу, но здесь дополнительная защита от rate limit
            time.sleep(15)

        # persist seen
        try:
            with open(smartliquidity_memory, "w", encoding="utf-8") as f:
                json.dump(list(seen), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write seen-file %s: %s", smartliquidity_memory, str(e))

        logger.info("Collected %d HIGH items from %s", len(sort_post), url)
        return sort_post

    except requests.RequestException as e:
        logger.error("HTTP error for feed %s: %s", url, str(e))
        return None
    except ET.ParseError as e:
        logger.error("XML parse error for %s: %s", url, str(e))
        return None
    except Exception as e:
        logger.exception("Unexpected error parsing feed %s: %s", url, str(e))
        return None


# функции парсинга
def desk_parsing():
    try:
        json_coindesk = "coindesk_memory.json"
        url = "https://www.coindesk.com/arc/outboundfeeds/rss"
        sort_post = []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml"
        }

        response = requests.get(url, headers=headers, timeout=40)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")

        for item in items[:10]:
            title_full = item.findtext("title")
            link_full = item.findtext("link")
            page_text = get_page_text(link_full)
            time.sleep(2)

            media_ns = "{http://search.yahoo.com/mrss/}"

            try:
                img_full = item.find(f"{media_ns}content").attrib.get("url")
            except (AttributeError, IndexError, KeyError):
                img_full = None

            try:
                with open(json_coindesk, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    title_org = deque(list(loaded), maxlen=30)
            except:
                title_org = deque(maxlen=30)

            if title_full in title_org:
                logger.info("Skip coindesk")
                continue
            else:
                title_org.append(title_full)
                with open(json_coindesk, "w", encoding="utf-8") as f:
                    json.dump(list(title_org), f, ensure_ascii=False, indent=1)

            headers_mistral = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }

            prompt = f'''
            You are a news filter for the crypto market. Use this information for analysis: {page_text}. Your task is to assign each news item one of three priorities—HIGH or LOW—based on its impact on the market. Follow these steps precisely:

            **Step 1: Analyze the News**

            - Make a short summary of the content, key data, and related themes.
            - Review the entire source, focusing on financial news, macroeconomic data, and events affecting the economy.
            - Identify themes tied to the criteria below.

            **Step 2: Assign Priority**

            - HIGH: News with direct, significant impact on the crypto market.
            - LOW: News creating a general informational or emotional background, without direct impact or unrelated to markets or with minimal impact.

            **Priority Criteria:**
            HIGH Priority

            Assign HIGH if the news includes any of these:

            Macroeconomic & Regulatory:

            - Decisions or statements from the US Fed, ECB, People’s Bank of China, or Bank of Japan on interest rates, inflation, or liquidity. For example, a statement from the US Fed about raising interest rates.
            - Crypto regulation launches or bans in the USA, EU, China, Hong Kong, or UAE. For example, the EU banning a specific cryptocurrency.
            - SEC or CFTC rulings on major crypto exchange cases (e.g., Ripple, Binance, Coinbase). For example, the SEC ruling against Ripple.

            Investments & Finance:

            - Approval or launch of Bitcoin-ETFs or other crypto-ETFs in major jurisdictions. For example, the approval of a Bitcoin-ETF in the USA.

            Partnerships & Integrations:

            - Collaborations between crypto projects and companies like Meta, Apple, Amazon, Google, or Microsoft. For example, Meta partnering with a crypto project to integrate blockchain technology.
            - Crypto or blockchain integration into payment systems like PayPal, Visa, or Mastercard. For example, Visa integrating a cryptocurrency into its payment system.

            Key Figures:

            - Public statements or tweets from Elon Musk, Jerome Powell, Gary Gensler, Changpeng Zhao, Vitalik Buterin, or similar influential figures. For example, Elon Musk tweeting about a cryptocurrency.

            Do not assign HIGH if the news is:

            - Predictions without evidence (e.g., “expert says prices will rise”). For example, an analyst predicting that Bitcoin will reach $100,000 without providing evidence.
            - Local regulatory news from minor countries (outside USA, EU, China). For example, a small country in Africa banning cryptocurrency.
            - Unverified rumors or social media speculation. For example, a rumor on Twitter about a major crypto exchange being hacked.

            LOW Priority

            Assign LOW if the news includes any of these:

            Technological Events:

            - Hard forks, soft forks, or testnet launches in top-50 crypto projects by market cap. For example, a hard fork in Ethereum.
            - Launch of new blockchain protocols. For example, the launch of a new blockchain protocol for decentralized finance.
            - Integration of AI, IoT, or Web3 into the crypto ecosystem. For example, a crypto project integrating AI to improve its services.

            Second-Tier Companies:

            - Partnerships or collaborations involving crypto firms. For example, two crypto firms partnering to develop a new product.
            - Launches of exchanges, wallets, or DeFi applications. For example, the launch of a new decentralized exchange.

            Reports & Lesser Markets:

            - Analytical publications from Messari, Glassnode, CoinShares, or similar agencies. For example, a report from Messari on the state of the crypto market.
            - Regulatory news from less significant regions (e.g., South America, Africa). For example, a country in South America introducing new crypto regulations.
            - Vague, lacking specifics (e.g., “planned for the future”). For example, a crypto project announcing plans to launch a new product without specifying when.
            - Airdrop promotions, contests, or marketing campaigns. For example, a crypto project announcing an airdrop to promote its token.
            - Tech-related but unconnected to crypto. For example, a news item about a new smartphone without any mention of crypto.

            Low-Capitalization Events:

            - NFT drops from obscure projects. For example, an unknown artist launching an NFT collection.
            - Token launches with market cap below $10 million. For example, a new token with a market cap of $5 million.
            - Fraud or scams involving minor projects. For example, a small crypto project being accused of fraud.

            Crypto Culture:

            - Memes or social media disputes among regular users. For example, a meme about a cryptocurrency going viral on Twitter.
            - News about influencers with negligible market influence. For example, a minor influencer endorsing a cryptocurrency.

            Unrelated or Minor:

            - Tech news without crypto impact. For example, a news item about a new programming language.
            - Small-scale regional events in insignificant jurisdictions. For example, a local crypto meetup in a small town.
            - Repetitions of prior news without new details. For example, a news item repeating information from a previous news item.
            - PR materials or press releases lacking substance. For example, a press release from a crypto project announcing a new partnership without providing details.
            - Texts not mentioning cryptocurrencies, blockchain, or financial markets. For example, a news item about a new movie.
            - Speculative or opinion-based articles without substantial data or market impact.
            - Interrogative sentences and questions.
            - General (NOT SPECIFIC) information without definite data.

            Whales

            - Whales transactions
            - Large investments

            **Step 3: Output**

            - First, provide a concise 2-3 sentence summary of the news and its key topics.
            - Then, give a brief analysis of the news and themes to determine priority.
            - State the priority level on a separate line using the word PRIORITY and colon «:».

            Avoid introductory phrases, labels, or explanations.

            Example:Solana launches new scalable DeFi protocol. Solana unveiled Solana Pay V2, a protocol enhancing DeFi applications with 30% lower transaction fees and AI-optimized smart contracts, set for release in August 2025. This development strengthens Solana’s position against competitors like Ethereum by improving scalability and user experience. The news, while significant for DeFi and technological innovation, lacks the broader market impact of regulatory changes or major investments. #Solana #DeFi

            The news qualifies as LOW priority due to its focus on a new blockchain protocol and AI integration, aligning with technological event criteria, but it does not involve regulatory decisions, large-scale investments, or influential figures required for HIGH priority. Improvements to the prompt include replacing "critical clarity" with "quantifiable data not concisely summarized," specifying news prioritization by market or adoption impact, adding a language instruction like "Use English unless specified," and including an example without a list for clarity.

            [PRIORITY: LOW]

            MAXIMUM : 100 words
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers_mistral, json=data, timeout=30)
                response.raise_for_status()
                full_response = response.json()['choices'][0]['message']['content'].strip()
                
                match = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", full_response.upper())
                if not match:
                    logger.error("Failed to extract degree:", full_response)
                    continue
                priority = match.group(1)

                if priority == "HIGH":
                    # семантическая дедупликация
                    try:
                        is_dup = is_semantic_duplicate(title_full, page_text or "", link=link_full)
                    except Exception as e:
                        logger.warning("Semantic duplicate check failed for %s: %s", title_full, e)
                        is_dup = False  # не блокируем постинг из-за ошибки дедупликации

                    if is_dup:
                        logger.info("Skipped semantic duplicate: %s", title_full)
                    else:
                        sort_post.append({
                            "title": title_full,
                            "link": link_full,
                            "page_text": (page_text)[:20000],
                            "mistral_raw": full_response
                        })
                        logger.info("HIGH added (unique): %s", title_full)

                else:
                    logger.info("BRUH PRIORITY")
                    continue

            except requests.exceptions.RequestException as e:
                logger.error("Error mistral (CoinDesk)", {str(e)})
                continue 

            time.sleep(20)

        logger.info(f"{len(sort_post)} news items collected (CoinDesk)")
        return sort_post

    except Exception as e:
        logger.error(f"Error {url}: {str(e)}")
        return None

def telegraph_parsing():
    try:
        json_telegraph = "telegraph_memory.json"
        url = "https://cointelegraph.com/rss"
        sort_post = []

        feed = feedparser.parse(url)

        if not feed.entries:
            logger.warning("No entries found in RSS feed.")
            return

        for entry in feed.entries[:10]:
            title_full = entry.title
            link_full = entry.link
            page_text = get_page_text(link_full)
            time.sleep(2)

            try:
                with open(json_telegraph, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    title_org = deque(list(loaded), maxlen=30)
            except:
                title_org = deque(maxlen=30)

            if title_full in title_org:
                logger.info("Skip telegraph")
                continue
            else:
                title_org.append(title_full)
                with open(json_telegraph, "w", encoding="utf-8") as f:
                    json.dump(list(title_org), f, ensure_ascii=False, indent=1)

            headers_mistral = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }

            prompt = f'''
            You are a news filter for the crypto market. Use this information for analysis: {page_text}. Your task is to assign each news item one of three priorities—HIGH or LOW—based on its impact on the market. Follow these steps precisely:

            **Step 1: Analyze the News**

            - Make a short summary of the content, key data, and related themes.
            - Review the entire source, focusing on financial news, macroeconomic data, and events affecting the economy.
            - Identify themes tied to the criteria below.

            **Step 2: Assign Priority**

            - HIGH: News with direct, significant impact on the crypto market.
            - LOW: News creating a general informational or emotional background, without direct impact or unrelated to markets or with minimal impact.

            **Priority Criteria:**
            HIGH Priority

            Assign HIGH if the news includes any of these:

            Macroeconomic & Regulatory:

            - Decisions or statements from the US Fed, ECB, People’s Bank of China, or Bank of Japan on interest rates, inflation, or liquidity. For example, a statement from the US Fed about raising interest rates.
            - Crypto regulation launches or bans in the USA, EU, China, Hong Kong, or UAE. For example, the EU banning a specific cryptocurrency.
            - SEC or CFTC rulings on major crypto exchange cases (e.g., Ripple, Binance, Coinbase). For example, the SEC ruling against Ripple.

            Investments & Finance:

            - Approval or launch of Bitcoin-ETFs or other crypto-ETFs in major jurisdictions. For example, the approval of a Bitcoin-ETF in the USA.

            Partnerships & Integrations:

            - Collaborations between crypto projects and companies like Meta, Apple, Amazon, Google, or Microsoft. For example, Meta partnering with a crypto project to integrate blockchain technology.
            - Crypto or blockchain integration into payment systems like PayPal, Visa, or Mastercard. For example, Visa integrating a cryptocurrency into its payment system.

            Key Figures:

            - Public statements or tweets from Elon Musk, Jerome Powell, Gary Gensler, Changpeng Zhao, Vitalik Buterin, or similar influential figures. For example, Elon Musk tweeting about a cryptocurrency.

            Do not assign HIGH if the news is:

            - Predictions without evidence (e.g., “expert says prices will rise”). For example, an analyst predicting that Bitcoin will reach $100,000 without providing evidence.
            - Local regulatory news from minor countries (outside USA, EU, China). For example, a small country in Africa banning cryptocurrency.
            - Unverified rumors or social media speculation. For example, a rumor on Twitter about a major crypto exchange being hacked.

            LOW Priority

            Assign LOW if the news includes any of these:

            Technological Events:

            - Hard forks, soft forks, or testnet launches in top-50 crypto projects by market cap. For example, a hard fork in Ethereum.
            - Launch of new blockchain protocols. For example, the launch of a new blockchain protocol for decentralized finance.
            - Integration of AI, IoT, or Web3 into the crypto ecosystem. For example, a crypto project integrating AI to improve its services.

            Second-Tier Companies:

            - Partnerships or collaborations involving crypto firms. For example, two crypto firms partnering to develop a new product.
            - Launches of exchanges, wallets, or DeFi applications. For example, the launch of a new decentralized exchange.

            Reports & Lesser Markets:

            - Analytical publications from Messari, Glassnode, CoinShares, or similar agencies. For example, a report from Messari on the state of the crypto market.
            - Regulatory news from less significant regions (e.g., South America, Africa). For example, a country in South America introducing new crypto regulations.
            - Vague, lacking specifics (e.g., “planned for the future”). For example, a crypto project announcing plans to launch a new product without specifying when.
            - Airdrop promotions, contests, or marketing campaigns. For example, a crypto project announcing an airdrop to promote its token.
            - Tech-related but unconnected to crypto. For example, a news item about a new smartphone without any mention of crypto.

            Low-Capitalization Events:

            - NFT drops from obscure projects. For example, an unknown artist launching an NFT collection.
            - Token launches with market cap below $10 million. For example, a new token with a market cap of $5 million.
            - Fraud or scams involving minor projects. For example, a small crypto project being accused of fraud.

            Crypto Culture:

            - Memes or social media disputes among regular users. For example, a meme about a cryptocurrency going viral on Twitter.
            - News about influencers with negligible market influence. For example, a minor influencer endorsing a cryptocurrency.

            Unrelated or Minor:

            - Tech news without crypto impact. For example, a news item about a new programming language.
            - Small-scale regional events in insignificant jurisdictions. For example, a local crypto meetup in a small town.
            - Repetitions of prior news without new details. For example, a news item repeating information from a previous news item.
            - PR materials or press releases lacking substance. For example, a press release from a crypto project announcing a new partnership without providing details.
            - Texts not mentioning cryptocurrencies, blockchain, or financial markets. For example, a news item about a new movie.
            - Speculative or opinion-based articles without substantial data or market impact.
            - Interrogative sentences and questions.
            - General (NOT SPECIFIC) information without definite data.

            Whales

            - Whales transactions
            - Large investments

            **Step 3: Output**

            - First, provide a concise 2-3 sentence summary of the news and its key topics.
            - Then, give a brief analysis of the news and themes to determine priority.
            - State the priority level on a separate line using the word PRIORITY and colon «:».

            Avoid introductory phrases, labels, or explanations.

            Example:Solana launches new scalable DeFi protocol. Solana unveiled Solana Pay V2, a protocol enhancing DeFi applications with 30% lower transaction fees and AI-optimized smart contracts, set for release in August 2025. This development strengthens Solana’s position against competitors like Ethereum by improving scalability and user experience. The news, while significant for DeFi and technological innovation, lacks the broader market impact of regulatory changes or major investments. #Solana #DeFi

            The news qualifies as LOW priority due to its focus on a new blockchain protocol and AI integration, aligning with technological event criteria, but it does not involve regulatory decisions, large-scale investments, or influential figures required for HIGH priority. Improvements to the prompt include replacing "critical clarity" with "quantifiable data not concisely summarized," specifying news prioritization by market or adoption impact, adding a language instruction like "Use English unless specified," and including an example without a list for clarity.

            [PRIORITY: LOW]

            MAXIMUM : 100 words
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers_mistral, json=data, timeout=30)
                response.raise_for_status()
                full_response = response.json()['choices'][0]['message']['content'].strip()
                
                match = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", full_response.upper())
                if not match:
                    logger.error("Failed to extract degree:", full_response)
                    continue
                priority = match.group(1)

                if priority == "HIGH":
                    # семантическая дедупликация
                    try:
                        is_dup = is_semantic_duplicate(title_full, page_text or "", link=link_full)
                    except Exception as e:
                        logger.warning("Semantic duplicate check failed for %s: %s", title_full, e)
                        is_dup = False  # не блокируем постинг из-за ошибки дедупликации

                    if is_dup:
                        logger.info("Skipped semantic duplicate: %s", title_full)
                    else:
                        sort_post.append({
                            "title": title_full,
                            "link": link_full,
                            "page_text": (page_text)[:20000],
                            "mistral_raw": full_response
                        })
                        logger.info("HIGH added (unique): %s", title_full)

                else:
                    logger.info("BRUH PRIORITY")
                    continue

            except requests.exceptions.RequestException as e:
                logger.error("Error mistral (telegraph)", {str(e)})
                continue 

            time.sleep(20)

        logger.info(f"{len(sort_post)} news items collected (telegraph)")
        return sort_post

    except Exception as e:
        logger.error(f"Error {url}: {str(e)}")
        return None
    
def yahoo_parsing():
    try:
        json_yahoo = "yahoo_memory.json"
        url = "https://finance.yahoo.com/news/rssindex"
        sort_post = []

        feed = feedparser.parse(url)

        if not feed.entries:
            logger.warning("No entries found in RSS feed.")
            return

        for entry in feed.entries[:10]:
            title_full = entry.title
            link_full = entry.link
            page_text = get_page_text(link_full)
            time.sleep(2)

            try:
                with open(json_yahoo, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    title_org = deque(list(loaded), maxlen=30)
            except:
                title_org = deque(maxlen=30)

            if title_full in title_org:
                logger.info("Skip yahoo")
                continue
            else:
                title_org.append(title_full)
                with open(json_yahoo, "w", encoding="utf-8") as f:
                    json.dump(list(title_org), f, ensure_ascii=False, indent=1)

            headers_mistral = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }

            prompt = f'''
            You are a news filter for the crypto market. Use this information for analysis: {page_text}. Your task is to assign each news item one of three priorities—HIGH or LOW—based on its impact on the market. Follow these steps precisely:

            **Step 1: Analyze the News**

            - Make a short summary of the content, key data, and related themes.
            - Review the entire source, focusing on financial news, macroeconomic data, and events affecting the economy.
            - Identify themes tied to the criteria below.

            **Step 2: Assign Priority**

            - HIGH: News with direct, significant impact on the crypto market.
            - LOW: News creating a general informational or emotional background, without direct impact or unrelated to markets or with minimal impact.

            **Priority Criteria:**
            HIGH Priority

            Assign HIGH if the news includes any of these:

            Macroeconomic & Regulatory:

            - Decisions or statements from the US Fed, ECB, People’s Bank of China, or Bank of Japan on interest rates, inflation, or liquidity. For example, a statement from the US Fed about raising interest rates.
            - Crypto regulation launches or bans in the USA, EU, China, Hong Kong, or UAE. For example, the EU banning a specific cryptocurrency.
            - SEC or CFTC rulings on major crypto exchange cases (e.g., Ripple, Binance, Coinbase). For example, the SEC ruling against Ripple.

            Investments & Finance:

            - Approval or launch of Bitcoin-ETFs or other crypto-ETFs in major jurisdictions. For example, the approval of a Bitcoin-ETF in the USA.

            Partnerships & Integrations:

            - Collaborations between crypto projects and companies like Meta, Apple, Amazon, Google, or Microsoft. For example, Meta partnering with a crypto project to integrate blockchain technology.
            - Crypto or blockchain integration into payment systems like PayPal, Visa, or Mastercard. For example, Visa integrating a cryptocurrency into its payment system.

            Key Figures:

            - Public statements or tweets from Elon Musk, Jerome Powell, Gary Gensler, Changpeng Zhao, Vitalik Buterin, or similar influential figures. For example, Elon Musk tweeting about a cryptocurrency.

            Do not assign HIGH if the news is:

            - Predictions without evidence (e.g., “expert says prices will rise”). For example, an analyst predicting that Bitcoin will reach $100,000 without providing evidence.
            - Local regulatory news from minor countries (outside USA, EU, China). For example, a small country in Africa banning cryptocurrency.
            - Unverified rumors or social media speculation. For example, a rumor on Twitter about a major crypto exchange being hacked.

            LOW Priority

            Assign LOW if the news includes any of these:

            Technological Events:

            - Hard forks, soft forks, or testnet launches in top-50 crypto projects by market cap. For example, a hard fork in Ethereum.
            - Launch of new blockchain protocols. For example, the launch of a new blockchain protocol for decentralized finance.
            - Integration of AI, IoT, or Web3 into the crypto ecosystem. For example, a crypto project integrating AI to improve its services.

            Second-Tier Companies:

            - Partnerships or collaborations involving crypto firms. For example, two crypto firms partnering to develop a new product.
            - Launches of exchanges, wallets, or DeFi applications. For example, the launch of a new decentralized exchange.

            Reports & Lesser Markets:

            - Analytical publications from Messari, Glassnode, CoinShares, or similar agencies. For example, a report from Messari on the state of the crypto market.
            - Regulatory news from less significant regions (e.g., South America, Africa). For example, a country in South America introducing new crypto regulations.
            - Vague, lacking specifics (e.g., “planned for the future”). For example, a crypto project announcing plans to launch a new product without specifying when.
            - Airdrop promotions, contests, or marketing campaigns. For example, a crypto project announcing an airdrop to promote its token.
            - Tech-related but unconnected to crypto. For example, a news item about a new smartphone without any mention of crypto.

            Low-Capitalization Events:

            - NFT drops from obscure projects. For example, an unknown artist launching an NFT collection.
            - Token launches with market cap below $10 million. For example, a new token with a market cap of $5 million.
            - Fraud or scams involving minor projects. For example, a small crypto project being accused of fraud.

            Crypto Culture:

            - Memes or social media disputes among regular users. For example, a meme about a cryptocurrency going viral on Twitter.
            - News about influencers with negligible market influence. For example, a minor influencer endorsing a cryptocurrency.

            Unrelated or Minor:

            - Tech news without crypto impact. For example, a news item about a new programming language.
            - Small-scale regional events in insignificant jurisdictions. For example, a local crypto meetup in a small town.
            - Repetitions of prior news without new details. For example, a news item repeating information from a previous news item.
            - PR materials or press releases lacking substance. For example, a press release from a crypto project announcing a new partnership without providing details.
            - Texts not mentioning cryptocurrencies, blockchain, or financial markets. For example, a news item about a new movie.
            - Speculative or opinion-based articles without substantial data or market impact.
            - Interrogative sentences and questions.
            - General (NOT SPECIFIC) information without definite data.

            Whales

            - Whales transactions
            - Large investments

            **Step 3: Output**

            - First, provide a concise 2-3 sentence summary of the news and its key topics.
            - Then, give a brief analysis of the news and themes to determine priority.
            - State the priority level on a separate line using the word PRIORITY and colon «:».

            Avoid introductory phrases, labels, or explanations.

            Example:Solana launches new scalable DeFi protocol. Solana unveiled Solana Pay V2, a protocol enhancing DeFi applications with 30% lower transaction fees and AI-optimized smart contracts, set for release in August 2025. This development strengthens Solana’s position against competitors like Ethereum by improving scalability and user experience. The news, while significant for DeFi and technological innovation, lacks the broader market impact of regulatory changes or major investments. #Solana #DeFi

            The news qualifies as LOW priority due to its focus on a new blockchain protocol and AI integration, aligning with technological event criteria, but it does not involve regulatory decisions, large-scale investments, or influential figures required for HIGH priority. Improvements to the prompt include replacing "critical clarity" with "quantifiable data not concisely summarized," specifying news prioritization by market or adoption impact, adding a language instruction like "Use English unless specified," and including an example without a list for clarity.

            [PRIORITY: LOW]

            MAXIMUM : 100 words
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers_mistral, json=data, timeout=30)
                response.raise_for_status()
                full_response = response.json()['choices'][0]['message']['content'].strip()
                
                match = re.search(r"PRIORITY:\s*(HIGH|MEDIUM|LOW)", full_response.upper())
                if not match:
                    logger.error("Failed to extract degree:", full_response)
                    continue
                priority = match.group(1)

                if priority == "HIGH":
                    # семантическая дедупликация
                    try:
                        is_dup = is_semantic_duplicate(title_full, page_text or "", link=link_full)
                    except Exception as e:
                        logger.warning("Semantic duplicate check failed for %s: %s", title_full, e)
                        is_dup = False  # не блокируем постинг из-за ошибки дедупликации

                    if is_dup:
                        logger.info("Skipped semantic duplicate: %s", title_full)
                    else:
                        sort_post.append({
                            "title": title_full,
                            "link": link_full,
                            "page_text": (page_text)[:20000],
                            "mistral_raw": full_response
                        })
                        logger.info("HIGH added (unique): %s", title_full)

                else:
                    logger.info("BRUH PRIORITY")
                    continue

            except requests.exceptions.RequestException as e:
                logger.error("Error mistral (yahoo)", {str(e)})
                continue 

            time.sleep(20)

        logger.info(f"{len(sort_post)} news items collected (yahoo)")
        return sort_post

    except Exception as e:
        logger.error(f"Error {url}: {str(e)}")
        return None
                
def coin_parsing():
    coin_full_course = []
    url = URL_CG
    
    params = {
        "vs_currency": "usd",
        "ids": "bitcoin,ethereum"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        try:
            for coin in data:
                name = coin.get("name")
                price = coin.get("current_price")
                symbol = coin.get("symbol")

                entry = {
                    "name": name,
                    "price": price,
                    "symbol": symbol
                }
                
                coin_full_course.append(entry)

                time.sleep(2)
                logger.info(f"coin good parse: {name}")

            return coin_full_course

        except Exception as e:
            logger.error(f"loose parse coin: {name}: {str(e)}")
    
    except Exception as e:
        logger.error(f"coin_parsing Error: {str(e)}")
        return

def trend_parsing(threshold: float = 0.5) -> str:
    url = "https://api.coingecko.com/api/v3/global"

    try:
        response = requests.get(url)
        data = response.json()
        
        change = data['data']['market_cap_change_percentage_24h_usd']
        
        if change >= threshold:
            trend = f"Восходящий 📈"
        elif change <= -threshold:
            trend = f"Нисходящий 📉"
        else:
            trend = f"Нейтральный 💤"

        logger.info("trend received")

        return trend
    
    except Exception as e:
        logger.error(f"Failed to get trend: {str(e)}")
        return

def index_parsing():
    url = FEARGREED_API

    params = {
        "limit": 1,
        "format": "json"
    }

    try:
        response = requests.get(url=url, params=params)
        data = response.json()

        value = data.get("data")[0].get("value")
        value_class = data.get("data")[0].get("value_classification")

        logger.info("index received")

        return value, value_class
    
    except Exception as e:
        logger.error(f"Failed to get index: {str(e)}")
        return

def usd_parsing():
    url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"

    try:
        response = requests.get(url)
        data = response.json()
        usd = data["usd"]["rub"]

        logger.info("USD course received")

        return usd
    
    except Exception as e:
        logger.error(f"Failed to get USD course: {str(e)}")
        return

def sp500_parsing():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers)

    try:
        data = response.json()
        value = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return value
    
    except Exception as e:
        logger.error(f"Failed to get USD course: {str(e)}")
        return
    

def generate_thedefiant_post(sort_post):
    try:
        url = "https://thedefiant.io/api/feed"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        if not sort_post:
            logger.error("not sort post in generate_thedefiant_post")
            return []
        
        generated_message = []

        for post in sort_post:
            comment = post.get("comment")
            link = post.get("link")
            page_text = post.get("page_text")
            if not page_text:
                logger.warning("No page text from thedefiant")
                continue
            title = post.get("title")
            image = post.get("image")

            prompt = f'''
            You're a professional journalist and cryptanalyst. Paraphrase {page_text} and make a suitable format, don't change the numeric and other data. Never write something that isn't in given text or make anything up. Your main task is to paraphrase, format and translate. The output must be solely the Telegram post in Russian, strictly formatted according to the rules below.

            Example: DeFi Development Company, неофициально именуемая «MSTR of Solana», подала заявку на размещение акций на сумму $1 млрд. #DeFi

            CONTENT FOCUS REQUIREMENTS - CRITICAL:

            - Focus on the actual NEWS EVENT or ACTION, not how the source reported it or analyse it.
            - DO NOT write about "источник сообщает", "по данным", "как стало известно" - write about the actual event.
            - Extract the core business event, decision, launch, or development that happened.
            - IGNORE meta-information about how the news was discovered or reported.
            - Never mention the name of the source even if it is mentioned in the article.
            - Extract and present only information explicitly stated in the text. Do not infer, extrapolate, or include details not directly provided, ensuring all content is current and accurate as per the article’s text.
            - End of the last sentence — add 1-3 hashtags on the topic of the news (use more specific hashtags like «DeFi» or «Airdrop», DON’T and NEVER use hashtags with general meaning like «Крипторынок», «Криптовалюта», «Финансы», «Crypto», «Blockchain» etc.)
            - Note: Hashtags must always appear on the same line as the last sentence, no line breaks before hashtags.
            - Token marking - use ticker hashtags when tokens are mentioned: #BTC #ETH #SOL #BNB #XRP etc.

            OPTIONAL HTML FORMATTING FOR MULTI-SENTENCE POSTS:
            You MAY (but don't have to do it every time) use this format, but ONLY if your post contains 2 or more sentences
            You MUST ALWAYS follow this exact structure:

            1. <b>ENTIRE FIRST SENTENCE</b>
            2. line break (press Enter)
            3. OTHER SENTENCES and hashtags

            You can make some paragraphs with line breaks, but DO NOT make any line breaks before hashtags

            CORRECT multi-sentence example:
            <b> Meta совместно с Oakley новые смарт-очки с искусственным интеллектом для спортсменов.</b>

            Устройство получило улучшенные возможности видеозаписи и интеграцию с социальными платформами. #Web3

            WRONG formatting examples to AVOID:

            - NO <b></b> tags around first sentence
            - No line breaks between sentences
            - Bold formatting on second sentence
            - Missing HTML structure
            - line breaks before hashtags

            NEWS CONTENT AND NARRATIVE STRUCTURE REQUIREMENTS:

            - Use active voice construction with clear subject-verb-object structure.
            - Identify the actual actor/company/person performing the action as the grammatical subject.
            - Use dynamic action verbs (запустила, объявила, представила, начала, завершила).
            - Avoid passive voice constructions like "было объявлено", "представляются", "запускаются".
            - Present the main business event, not the reporting process.
            - Include key details (amounts, dates, partnerships, prices) naturally within the sentence structure.
            - Pay close attention to numbers, rates, and costs and mention numerical data in the post.
            - The post should not contain any information other than from the given text
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.

            LANGUAGE AND CONTENT REQUIREMENTS:

            - ALL posts MUST be in Russian language.
            - Write in natural Russian with correct grammar and punctuation.
            - Translate foreign terms using accepted Russian crypto terminology.
            - Maintain natural Russian sentence structure and word order.
            - Present only factual information without opinions or analysis.
            - Proper punctuation according to Russian language standards.
            - Do not decline token names; keep them as tickers (e.g., ETH, BTC, XRP etc.).
            - Never write something that isn't in the given text and do not make anything up.

            CRITICAL RESTRICTIONS:

            - Do not include any additional text, explanations, or meta-information.
            - Do not use introductory phrases or editorial comments.
            - Focus on the actual news event, not how it was reported.
            - Maintain journalistic objectivity and factual accuracy.
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.
            - Never write something that isn't in text or make anything up.
            - Never duplicate the information

            Output ONLY the Telegram post itself in Russian, strictly adhering to the provided formatting and content requirements. Do not include any additional commentary, explanations, compliance notes, or meta-information about the post creation process, even if it describes adherence to the rules.
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                logger.info("post from mistral received")
                generated_message.append(content)
    
            except Exception as e:
                logger.error(f"Error generation (thedefiant): {str(e)}")
                return []
            
            time.sleep(15)

        return generated_message
    
    except Exception as e:
        logger.error(f"Error generate_thedefiant_post: {str(e)}")
        return None
    

def generate_smartliquidity_post(sort_post):
    try:
        url = "https://smartliquidity.info/feed/"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        if not sort_post:
            logger.error("not sort post in generate_smartliquidity_post")
            return []
        
        generated_message = []

        for post in sort_post:
            comment = post.get("comment")
            link = post.get("link")
            page_text = post.get("page_text")
            if not page_text:
                logger.warning("No page text from smartliquidity")
                continue
            title = post.get("title")

            prompt = f'''
            You're a professional journalist and cryptanalyst. Paraphrase {page_text} and make a suitable format, don't change the numeric and other data. Never write something that isn't in given text or make anything up. Your main task is to paraphrase, format and translate. The output must be solely the Telegram post in Russian, strictly formatted according to the rules below.

            Example: DeFi Development Company, неофициально именуемая «MSTR of Solana», подала заявку на размещение акций на сумму $1 млрд. #DeFi

            CONTENT FOCUS REQUIREMENTS - CRITICAL:

            - Focus on the actual NEWS EVENT or ACTION, not how the source reported it or analyse it.
            - DO NOT write about "источник сообщает", "по данным", "как стало известно" - write about the actual event.
            - Extract the core business event, decision, launch, or development that happened.
            - IGNORE meta-information about how the news was discovered or reported.
            - Never mention the name of the source even if it is mentioned in the article.
            - Extract and present only information explicitly stated in the text. Do not infer, extrapolate, or include details not directly provided, ensuring all content is current and accurate as per the article’s text.
            - End of the last sentence — add 1-3 hashtags on the topic of the news (use more specific hashtags like «DeFi» or «Airdrop», DON’T and NEVER use hashtags with general meaning like «Крипторынок», «Криптовалюта», «Финансы», «Crypto», «Blockchain» etc.)
            - Note: Hashtags must always appear on the same line as the last sentence, no line breaks before hashtags.
            - Token marking - use ticker hashtags when tokens are mentioned: #BTC #ETH #SOL #BNB #XRP etc.

            OPTIONAL HTML FORMATTING FOR MULTI-SENTENCE POSTS:
            You MAY (but don't have to do it every time) use this format, but ONLY if your post contains 2 or more sentences
            You MUST ALWAYS follow this exact structure:

            1. <b>ENTIRE FIRST SENTENCE</b>
            2. line break (press Enter)
            3. OTHER SENTENCES and hashtags

            You can make some paragraphs with line breaks, but DO NOT make any line breaks before hashtags

            CORRECT multi-sentence example:
            <b> Meta совместно с Oakley новые смарт-очки с искусственным интеллектом для спортсменов.</b>

            Устройство получило улучшенные возможности видеозаписи и интеграцию с социальными платформами. #Web3

            WRONG formatting examples to AVOID:

            - NO <b></b> tags around first sentence
            - No line breaks between sentences
            - Bold formatting on second sentence
            - Missing HTML structure
            - line breaks before hashtags

            NEWS CONTENT AND NARRATIVE STRUCTURE REQUIREMENTS:

            - Use active voice construction with clear subject-verb-object structure.
            - Identify the actual actor/company/person performing the action as the grammatical subject.
            - Use dynamic action verbs (запустила, объявила, представила, начала, завершила).
            - Avoid passive voice constructions like "было объявлено", "представляются", "запускаются".
            - Present the main business event, not the reporting process.
            - Include key details (amounts, dates, partnerships, prices) naturally within the sentence structure.
            - Pay close attention to numbers, rates, and costs and mention numerical data in the post.
            - The post should not contain any information other than from the given text
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.

            LANGUAGE AND CONTENT REQUIREMENTS:

            - ALL posts MUST be in Russian language.
            - Write in natural Russian with correct grammar and punctuation.
            - Translate foreign terms using accepted Russian crypto terminology.
            - Maintain natural Russian sentence structure and word order.
            - Present only factual information without opinions or analysis.
            - Proper punctuation according to Russian language standards.
            - Do not decline token names; keep them as tickers (e.g., ETH, BTC, XRP etc.).
            - Never write something that isn't in the given text and do not make anything up.

            CRITICAL RESTRICTIONS:

            - Do not include any additional text, explanations, or meta-information.
            - Do not use introductory phrases or editorial comments.
            - Focus on the actual news event, not how it was reported.
            - Maintain journalistic objectivity and factual accuracy.
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.
            - Never write something that isn't in text or make anything up.
            - Never duplicate the information

            Output ONLY the Telegram post itself in Russian, strictly adhering to the provided formatting and content requirements. Do not include any additional commentary, explanations, compliance notes, or meta-information about the post creation process, even if it describes adherence to the rules.
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                logger.info("post from mistral received")
                generated_message.append(content)
    
            except Exception as e:
                logger.error(f"Error generation (smartliquidity): {str(e)}")
                return []
              
            time.sleep(15)

        return generated_message
    
    except Exception as e:
        logger.error(f"Error generate_smartliquidity_post: {str(e)}")
        return None

    
def generate_coindesk_post(sort_post):
    try:
        url = "https://www.coindesk.com/arc/outboundfeeds/rss"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        if not sort_post:
            logger.error("not sort post in generate_coindesk_post")
            return []
        
        generated_message = []

        for post in sort_post:
            comment = post.get("comment")
            link = post.get("link")
            page_text = post.get("page_text")
            if not page_text:
                logger.warning("No page text from CoinDesk")
                continue
            title = post.get("title")

            prompt = f'''
            You're a professional journalist and cryptanalyst. Paraphrase {page_text} and make a suitable format, don't change the numeric and other data. Never write something that isn't in given text or make anything up. Your main task is to paraphrase, format and translate. The output must be solely the Telegram post in Russian, strictly formatted according to the rules below.

            Example: DeFi Development Company, неофициально именуемая «MSTR of Solana», подала заявку на размещение акций на сумму $1 млрд. #DeFi

            CONTENT FOCUS REQUIREMENTS - CRITICAL:

            - Focus on the actual NEWS EVENT or ACTION, not how the source reported it or analyse it.
            - DO NOT write about "источник сообщает", "по данным", "как стало известно" - write about the actual event.
            - Extract the core business event, decision, launch, or development that happened.
            - IGNORE meta-information about how the news was discovered or reported.
            - Never mention the name of the source even if it is mentioned in the article.
            - Extract and present only information explicitly stated in the text. Do not infer, extrapolate, or include details not directly provided, ensuring all content is current and accurate as per the article’s text.
            - End of the last sentence — add 1-3 hashtags on the topic of the news (use more specific hashtags like «DeFi» or «Airdrop», DON’T and NEVER use hashtags with general meaning like «Крипторынок», «Криптовалюта», «Финансы», «Crypto», «Blockchain» etc.)
            - Note: Hashtags must always appear on the same line as the last sentence, no line breaks before hashtags.
            - Token marking - use ticker hashtags when tokens are mentioned: #BTC #ETH #SOL #BNB #XRP etc.

            OPTIONAL HTML FORMATTING FOR MULTI-SENTENCE POSTS:
            You MAY (but don't have to do it every time) use this format, but ONLY if your post contains 2 or more sentences
            You MUST ALWAYS follow this exact structure:

            1. <b>ENTIRE FIRST SENTENCE</b>
            2. line break (press Enter)
            3. OTHER SENTENCES and hashtags

            You can make some paragraphs with line breaks, but DO NOT make any line breaks before hashtags

            CORRECT multi-sentence example:
            <b> Meta совместно с Oakley новые смарт-очки с искусственным интеллектом для спортсменов.</b>

            Устройство получило улучшенные возможности видеозаписи и интеграцию с социальными платформами. #Web3

            WRONG formatting examples to AVOID:

            - NO <b></b> tags around first sentence
            - No line breaks between sentences
            - Bold formatting on second sentence
            - Missing HTML structure
            - line breaks before hashtags

            NEWS CONTENT AND NARRATIVE STRUCTURE REQUIREMENTS:

            - Use active voice construction with clear subject-verb-object structure.
            - Identify the actual actor/company/person performing the action as the grammatical subject.
            - Use dynamic action verbs (запустила, объявила, представила, начала, завершила).
            - Avoid passive voice constructions like "было объявлено", "представляются", "запускаются".
            - Present the main business event, not the reporting process.
            - Include key details (amounts, dates, partnerships, prices) naturally within the sentence structure.
            - Pay close attention to numbers, rates, and costs and mention numerical data in the post.
            - The post should not contain any information other than from the given text
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.

            LANGUAGE AND CONTENT REQUIREMENTS:

            - ALL posts MUST be in Russian language.
            - Write in natural Russian with correct grammar and punctuation.
            - Translate foreign terms using accepted Russian crypto terminology.
            - Maintain natural Russian sentence structure and word order.
            - Present only factual information without opinions or analysis.
            - Proper punctuation according to Russian language standards.
            - Do not decline token names; keep them as tickers (e.g., ETH, BTC, XRP etc.).
            - Never write something that isn't in the given text and do not make anything up.

            CRITICAL RESTRICTIONS:

            - Do not include any additional text, explanations, or meta-information.
            - Do not use introductory phrases or editorial comments.
            - Focus on the actual news event, not how it was reported.
            - Maintain journalistic objectivity and factual accuracy.
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.
            - Never write something that isn't in text or make anything up.
            - Never duplicate the information

            Output ONLY the Telegram post itself in Russian, strictly adhering to the provided formatting and content requirements. Do not include any additional commentary, explanations, compliance notes, or meta-information about the post creation process, even if it describes adherence to the rules.
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                logger.info("post from mistral received")
                generated_message.append(content)
    
            except Exception as e:
                logger.error(f"Error generation (desk): {str(e)}")
                return []
            
            time.sleep(15)

        return generated_message
    
    except Exception as e:
        logger.error(f"Error generate_coindesk_post: {str(e)}")
        return None

    
def generate_telegraph_post(sort_post):
    try:
        url = "https://cointelegraph.com/rss"

        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        if not sort_post:
            logger.error("not sort post in generate_telegraph_post")
            return []
        
        generated_message = []

        for post in sort_post:
            page_text = post.get("page_text")
            if not page_text:
                logger.warning("No page text from telegraph")
                continue

            prompt = f'''
            You're a professional journalist and cryptanalyst. Paraphrase {page_text} and make a suitable format, don't change the numeric and other data. Never write something that isn't in given text or make anything up. Your main task is to paraphrase, format and translate. The output must be solely the Telegram post in Russian, strictly formatted according to the rules below.

            Example: DeFi Development Company, неофициально именуемая «MSTR of Solana», подала заявку на размещение акций на сумму $1 млрд. #DeFi

            CONTENT FOCUS REQUIREMENTS - CRITICAL:

            - Focus on the actual NEWS EVENT or ACTION, not how the source reported it or analyse it.
            - DO NOT write about "источник сообщает", "по данным", "как стало известно" - write about the actual event.
            - Extract the core business event, decision, launch, or development that happened.
            - IGNORE meta-information about how the news was discovered or reported.
            - Never mention the name of the source even if it is mentioned in the article.
            - Extract and present only information explicitly stated in the text. Do not infer, extrapolate, or include details not directly provided, ensuring all content is current and accurate as per the article’s text.
            - End of the last sentence — add 1-3 hashtags on the topic of the news (use more specific hashtags like «DeFi» or «Airdrop», DON’T and NEVER use hashtags with general meaning like «Крипторынок», «Криптовалюта», «Финансы», «Crypto», «Blockchain» etc.)
            - Note: Hashtags must always appear on the same line as the last sentence, no line breaks before hashtags.
            - Token marking - use ticker hashtags when tokens are mentioned: #BTC #ETH #SOL #BNB #XRP etc.

            OPTIONAL HTML FORMATTING FOR MULTI-SENTENCE POSTS:
            You MAY (but don't have to do it every time) use this format, but ONLY if your post contains 2 or more sentences
            You MUST ALWAYS follow this exact structure:

            1. <b>ENTIRE FIRST SENTENCE</b>
            2. line break (press Enter)
            3. OTHER SENTENCES and hashtags

            You can make some paragraphs with line breaks, but DO NOT make any line breaks before hashtags

            CORRECT multi-sentence example:
            <b> Meta совместно с Oakley новые смарт-очки с искусственным интеллектом для спортсменов.</b>

            Устройство получило улучшенные возможности видеозаписи и интеграцию с социальными платформами. #Web3

            WRONG formatting examples to AVOID:

            - NO <b></b> tags around first sentence
            - No line breaks between sentences
            - Bold formatting on second sentence
            - Missing HTML structure
            - line breaks before hashtags

            NEWS CONTENT AND NARRATIVE STRUCTURE REQUIREMENTS:

            - Use active voice construction with clear subject-verb-object structure.
            - Identify the actual actor/company/person performing the action as the grammatical subject.
            - Use dynamic action verbs (запустила, объявила, представила, начала, завершила).
            - Avoid passive voice constructions like "было объявлено", "представляются", "запускаются".
            - Present the main business event, not the reporting process.
            - Include key details (amounts, dates, partnerships, prices) naturally within the sentence structure.
            - Pay close attention to numbers, rates, and costs and mention numerical data in the post.
            - The post should not contain any information other than from the given text
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.

            LANGUAGE AND CONTENT REQUIREMENTS:

            - ALL posts MUST be in Russian language.
            - Write in natural Russian with correct grammar and punctuation.
            - Translate foreign terms using accepted Russian crypto terminology.
            - Maintain natural Russian sentence structure and word order.
            - Present only factual information without opinions or analysis.
            - Proper punctuation according to Russian language standards.
            - Do not decline token names; keep them as tickers (e.g., ETH, BTC, XRP etc.).
            - Never write something that isn't in the given text and do not make anything up.

            CRITICAL RESTRICTIONS:

            - Do not include any additional text, explanations, or meta-information.
            - Do not use introductory phrases or editorial comments.
            - Focus on the actual news event, not how it was reported.
            - Maintain journalistic objectivity and factual accuracy.
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.
            - Never write something that isn't in text or make anything up.
            - Never duplicate the information

            Output ONLY the Telegram post itself in Russian, strictly adhering to the provided formatting and content requirements. Do not include any additional commentary, explanations, compliance notes, or meta-information about the post creation process, even if it describes adherence to the rules.
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                logger.info("post from mistral received")
                generated_message.append(content)
    
            except Exception as e:
                logger.error(f"Error generation (telegraph): {str(e)}")
                return []
            
            time.sleep(15)

        return generated_message
    
    except Exception as e:
        logger.error(f"Error generate_telegraph_post: {str(e)}")
        return None
    
def generate_yahoo_post(sort_post):
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }

        if not sort_post:
            logger.error("not sort post in generate_yahoo_post")
            return []
        
        generated_message = []

        for post in sort_post:
            page_text = post.get("page_text")
            if not page_text:
                logger.warning("No page text from yahoo")
                continue

            prompt = f'''
            You're a professional journalist and cryptanalyst. Paraphrase {page_text} and make a suitable format, don't change the numeric and other data. Never write something that isn't in given text or make anything up. Your main task is to paraphrase, format and translate. The output must be solely the Telegram post in Russian, strictly formatted according to the rules below.

            Example: DeFi Development Company, неофициально именуемая «MSTR of Solana», подала заявку на размещение акций на сумму $1 млрд. #DeFi

            CONTENT FOCUS REQUIREMENTS - CRITICAL:

            - Focus on the actual NEWS EVENT or ACTION, not how the source reported it or analyse it.
            - DO NOT write about "источник сообщает", "по данным", "как стало известно" - write about the actual event.
            - Extract the core business event, decision, launch, or development that happened.
            - IGNORE meta-information about how the news was discovered or reported.
            - Never mention the name of the source even if it is mentioned in the article.
            - Extract and present only information explicitly stated in the text. Do not infer, extrapolate, or include details not directly provided, ensuring all content is current and accurate as per the article’s text.
            - End of the last sentence — add 1-3 hashtags on the topic of the news (use more specific hashtags like «DeFi» or «Airdrop», DON’T and NEVER use hashtags with general meaning like «Крипторынок», «Криптовалюта», «Финансы», «Crypto», «Blockchain» etc.)
            - Note: Hashtags must always appear on the same line as the last sentence, no line breaks before hashtags.
            - Token marking - use ticker hashtags when tokens are mentioned: #BTC #ETH #SOL #BNB #XRP etc.

            OPTIONAL HTML FORMATTING FOR MULTI-SENTENCE POSTS:
            You MAY (but don't have to do it every time) use this format, but ONLY if your post contains 2 or more sentences
            You MUST ALWAYS follow this exact structure:

            1. <b>ENTIRE FIRST SENTENCE</b>
            2. line break (press Enter)
            3. OTHER SENTENCES and hashtags

            You can make some paragraphs with line breaks, but DO NOT make any line breaks before hashtags

            CORRECT multi-sentence example:
            <b> Meta совместно с Oakley новые смарт-очки с искусственным интеллектом для спортсменов.</b>

            Устройство получило улучшенные возможности видеозаписи и интеграцию с социальными платформами. #Web3

            WRONG formatting examples to AVOID:

            - NO <b></b> tags around first sentence
            - No line breaks between sentences
            - Bold formatting on second sentence
            - Missing HTML structure
            - line breaks before hashtags

            NEWS CONTENT AND NARRATIVE STRUCTURE REQUIREMENTS:

            - Use active voice construction with clear subject-verb-object structure.
            - Identify the actual actor/company/person performing the action as the grammatical subject.
            - Use dynamic action verbs (запустила, объявила, представила, начала, завершила).
            - Avoid passive voice constructions like "было объявлено", "представляются", "запускаются".
            - Present the main business event, not the reporting process.
            - Include key details (amounts, dates, partnerships, prices) naturally within the sentence structure.
            - Pay close attention to numbers, rates, and costs and mention numerical data in the post.
            - The post should not contain any information other than from the given text
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.

            LANGUAGE AND CONTENT REQUIREMENTS:

            - ALL posts MUST be in Russian language.
            - Write in natural Russian with correct grammar and punctuation.
            - Translate foreign terms using accepted Russian crypto terminology.
            - Maintain natural Russian sentence structure and word order.
            - Present only factual information without opinions or analysis.
            - Proper punctuation according to Russian language standards.
            - Do not decline token names; keep them as tickers (e.g., ETH, BTC, XRP etc.).
            - Never write something that isn't in the given text and do not make anything up.

            CRITICAL RESTRICTIONS:

            - Do not include any additional text, explanations, or meta-information.
            - Do not use introductory phrases or editorial comments.
            - Focus on the actual news event, not how it was reported.
            - Maintain journalistic objectivity and factual accuracy.
            - Hashtags must ALWAYS appear on the same line as the last sentence, NEVER use line breaks before hashtags.
            - Never write something that isn't in text or make anything up.
            - Never duplicate the information

            Output ONLY the Telegram post itself in Russian, strictly adhering to the provided formatting and content requirements. Do not include any additional commentary, explanations, compliance notes, or meta-information about the post creation process, even if it describes adherence to the rules.
            '''

            data = {
                "model": "mistral-medium-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 500
            }

            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                content = response.json()['choices'][0]['message']['content'].strip()
                logger.info("post from mistral received")
                generated_message.append(content)
    
            except Exception as e:
                logger.error(f"Error generation (yahoo): {str(e)}")
                return []
            
            time.sleep(15)

        return generated_message
    
    except Exception as e:
        logger.error(f"Error generate_yahoo_post: {str(e)}")
        return None
    

def post_thedefiant():
    posts = thedefiant_parsing()
    if not posts:
        logger.info("not posts list from thedefiant")
        return []
    
    messages = generate_thedefiant_post(posts)
    result = []

    # Сообщаем если длины не совпадают — zip будет брать по минимальной длине
    if len(messages) != len(posts):
        logger.warning("messages/posts length mismatch: %d messages, %d posts. Pairing by zip (min)." ,
                       len(messages), len(posts))

    # Парсим один к одному: предыдущая версия делала произведение всех сообщений и всех постов
    if messages:
        for message, post in zip(messages, posts):
            if message:
                # Берём картинку из полей image/ photo (совместимость)
                image = post.get("image") or post.get("photo") or None
                caption = (message or "").strip()

                if image:
                    result.append({
                        "website": "thedefiant",
                        "type": "photo",
                        "photo": image,       # publisher ожидает поле "photo"
                        "caption": caption    # publisher делает post.get("caption") + LINK
                    })
                else:
                    # fallback — если нет изображения, отправляем обычный текст-пост
                    result.append({
                        "website": "thedefiant",
                        "type": "text",
                        "text": caption       # publisher ожидает поле "text" для текстовых постов
                    })
    
        return result
    

def post_smartliquidity():
    posts = smartliquidity_parsing()
    if not posts:
        logger.info("not posts list from smartliquidity")
        return []
    
    messages = generate_smartliquidity_post(posts)
    result = []

    # Сообщаем если длины не совпадают — zip будет брать по минимальной длине
    if len(messages) != len(posts):
        logger.warning("messages/posts length mismatch: %d messages, %d posts. Pairing by zip (min)." ,
                       len(messages), len(posts))

    # Парсим один к одному: предыдущая версия делала произведение всех сообщений и всех постов
    if messages:
        for message, post in zip(messages, posts):
            if message:
                # Берём картинку из полей image/ photo (совместимость)
                image = post.get("image") or post.get("photo") or None
                caption = (message or "").strip()

                if image:
                    result.append({
                        "website": "smartliquidity",
                        "type": "photo",
                        "photo": image,       # publisher ожидает поле "photo"
                        "caption": caption    # publisher делает post.get("caption") + LINK
                    })
                else:
                    # fallback — если нет изображения, отправляем обычный текст-пост
                    result.append({
                        "website": "smartliquidity",
                        "type": "text",
                        "text": caption       # publisher ожидает поле "text" для текстовых постов
                    })
    
        return result


def post_coindesk():
    posts = desk_parsing()
    if not posts:
        logger.info("not posts list from coindesk")
        return []
    
    messages = generate_coindesk_post(posts)
    result = []
    if messages:
        for message in messages or []:
            if message:
                result.append({
                    "website": "coindesk",
                    "type": "text",
                    "text": message
                })

        return result
    
def post_telegraph():
    posts = telegraph_parsing()
    if not posts:
        logger.info("not posts list from telegraph")
        return []
    
    messages = generate_telegraph_post(posts)
    result = []
    if messages:
        for message in messages or []:
            if message:
                result.append({
                    "website": "telegraph",
                    "type": "text",
                    "text": message
                })

        return result
    
def post_yahoo():
    posts = yahoo_parsing()
    if not posts:
        logger.info("not posts list from yahoo")
        return []
    
    messages = generate_yahoo_post(posts)
    result = []
    if messages:
        for message in messages or []:
            if message:
                result.append({
                    "website": "yahoo",
                    "type": "text",
                    "text": message
                })

        return result


source_handlers = {
    "www.coindesk.com": post_coindesk,
    "cointelegraph.com": post_telegraph,
    "finance.yahoo.com": post_yahoo,
    "thedefiant.io": post_thedefiant,
    "smartliquidity.info": post_smartliquidity
}


def publisher():
    try:
        while True:
            post = post_queue.get()

            try:
                post_list.remove(post)
            except ValueError:
                logger.warning("Post not found in post_list")
            
            if post["type"] == "photo":
                bot.send_photo(
                    chat_id=CHANNEL_ID,
                    photo=post.get("photo"),
                    caption=post.get("caption") + LINK
                )
                logger.info("---------Publish photo - post---------")

            elif post["type"] == "text":
                bot.send_message(
                    chat_id=CHANNEL_ID,
                    text=post.get("text") + LINK
                )
                logger.info("---------Publish text - post---------")

            post_queue.task_done()

            with open(json_url_queue, "r+", encoding="utf-8") as f:
                f.seek(0)
                f.truncate()
                json.dump(post_list, f, ensure_ascii=False, indent=2)
                logger.info(f"Updated queue file with {len(post_list)} posts")

            time.sleep(random.uniform(60, 90))

    except Exception as e:
        logger.error(f"Publisher error: {str(e)}")


def parse_and_queue():
    all_posts = []
    for url in html_urls:
        for key, handler in source_handlers.items():
            if key in url:
                posts = handler() or []
                all_posts.extend(posts)
                break

    random.shuffle(all_posts)

    for post in all_posts:
        post_queue.put(post)
        post_list.append(post)

    with open(json_url_queue, "w", encoding="utf-8") as f:
        json.dump(post_list, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(post_list)} posts to file")


def pinned_message():
    text_list = []

    coin_full_course = coin_parsing()
    res = index_parsing()
    usd = usd_parsing()
    sp500 = sp500_parsing()
    trend = trend_parsing()

    if not coin_full_course:
        logger.error("no coin course")

    if not res:
        logger.error("no fear and greed index")
        return
    fg_value, fg_value_class = res

    classification = fg_value_class
    if fg_value_class == "Extreme Fear":
        classification = "Экстремальный страх"
    elif fg_value_class == "Fear":
        classification = "Страх"
    elif fg_value_class == "Neutral":
        classification = "Нейтрально"
    elif fg_value_class == "Greed":
        classification = "Жадность"
    elif fg_value_class == "Extreme Greed":
        classification = "Экстремальная жадность"
    
    if 0 <= int(fg_value) <= 24:
        emoji = "🟥"
    elif 25 <= int(fg_value) <= 49:
        emoji = "🟧"
    elif 50 <= int(fg_value) <= 65:
        emoji = "🟨"
    elif 66 <= int(fg_value) <= 100:
        emoji = "🟩"
    
    if not usd:
        logger.error("no USD course")

    if not sp500:
        logger.error("no S&P500 course")

    if not trend:
        logger.error("no trand")

    text_list.append("💸")
    for coin in coin_full_course:
        name = coin.get("name")
        price = coin.get("price")
        symbol = coin.get("symbol").upper()
        text_list.append(f"<b>{symbol}:</b> <code>${price}</code>")
    text_list.append(f"<b>USD:</b> <code>₽{usd:.2f}</code>")
    text_list.append(f"<b>S&P500:</b> <code>${sp500}</code>        ")

    course_text = "\n• ".join(text_list)
    index_text = f"\n\n{fg_value} — {classification} {emoji}        "
    trend_text = f"\n\n<b>Тренд:</b> {trend}"

    text = course_text + index_text + trend_text

    bot.edit_message_text(
        chat_id=CHANNEL_ID,
        message_id=PINNED_ID,
        text=text
    )
    logger.info("Edit pinned")

def parsing_loop():
    while True:
        try:
            parse_and_queue()
        except Exception as e:
            logger.warning(f"Loop error: {str(e)}")
        time.sleep(INTERVAL_CHECK)

def course_loop():
    while True:
        try:
            pinned_message()
        except Exception as e:
            logger.warning(f"Course loop error: {str(e)}")
        time.sleep(PINNED_INTERVAL_CHECK)

def main():
    threading.Thread(target=parsing_loop, daemon=True).start()
    threading.Thread(target=publisher, daemon=True).start()
    threading.Thread(target=course_loop, daemon=True).start()
    bot.infinity_polling()

if __name__ == '__main__':
    main()