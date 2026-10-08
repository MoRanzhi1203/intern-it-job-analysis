# -*- coding: utf-8 -*-
"""
实习僧并行抓取程序（item_text + detail_url 联合去重版 + 启动稳定性增强版）

功能：
1. 从 MySQL 输入任务表读取菜单搜索任务；
2. 使用 4 个独立进程并行抓取搜索结果页；
3. 每个进程内部固定使用两个标签页：
   - 列表页标签：负责搜索结果页、分页和详情链接提取；
   - 详情页标签：负责职位详情页访问和字段提取；
4. 将抓取结果、任务进度、详情进度统一写入 MySQL；
5. 支持断点续爬，避免重复抓取已完成的详情页；
6. 结果表与详情进度表统一按 item_text + detail_url 联合去重；
7. 主进程只解析一次 ChromeDriver 路径，避免 4 个子进程并发 install；
8. 增强 worker 启动日志、异常捕获、退出码输出，便于诊断“为什么只弹一个窗口/启动很慢”；
9. 支持“先静态分配，再动态补位”：每个窗口优先完成自己的初始任务，完成后继续帮助其他窗口处理剩余任务。
10. 专项优化窗口启动速度：优先显示窗口、延迟创建详情标签页、减少首屏资源加载、缩短错峰启动间隔。
11. 支持页级协作：当菜单任务变少时，空闲窗口可继续领取同一关键词任务的剩余页面，避免窗口空转。
"""

import os
import re
import time
import random
import traceback
import multiprocessing as mp
from datetime import datetime
from urllib.parse import urlparse, parse_qs, parse_qsl, urlencode, urlunparse, unquote

import mysql.connector
from mysql.connector import Error

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# =========================
# 全局数据库配置
# =========================
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "shixiseng_db",
    "charset": "utf8mb4",
}

# 输入任务表，由前置菜单构造脚本生成。
INPUT_TASK_TABLE = "shixiseng_internet_it_links"

# 最终详情结果表。
OUTPUT_RESULT_TABLE = "shixiseng_job_details"

# 任务进度表，记录每个菜单任务已完成到第几页。
TASK_PROGRESS_TABLE = "shixiseng_task_progress"

# 详情进度表，记录哪些 (item_text, detail_url) 已成功抓取。
DETAIL_PROGRESS_TABLE = "shixiseng_detail_progress"

# =========================
# 通用工具函数
# =========================
SEARCH_BASE_URL = "https://www.shixiseng.com/interns"
ALLOWED_SEARCH_PARAMS = ("keyword", "city", "type", "from", "page")


def normalize_text_param(value: str) -> str:
    """
    规范化 URL 查询参数文本。
    去除首尾空白字符，并进行 URL 解码，确保相同含义的参数具有一致的字符串表示。
    
    Args:
        value (str): 原始参数文本。
        
    Returns:
        str: 规范化后的参数文本。
    """

    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""
    return unquote(value).strip()


def canonicalize_search_url(item_url: str, page_no: int = None, keep_page: bool = False) -> str:
    """
    规范化实习僧搜索 URL：
    1. 统一 scheme/netloc/path
    2. 只保留 keyword/city/type/from/page
    3. 参数顺序固定
    4. page=1 时移除 page
    5. 统一编码，避免 Perl / Ruby 等关键词传参形式不同导致重复任务
    """

    if not item_url:
        return SEARCH_BASE_URL

    parsed = urlparse(item_url.strip())

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "www.shixiseng.com"
    path = parsed.path or "/interns"

    if "shixiseng.com" not in netloc:
        netloc = "www.shixiseng.com"

    if path != "/interns":
        path = "/interns"

    raw_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_map = {}
    for k, v in raw_pairs:
        k = (k or "").strip()
        if k not in ALLOWED_SEARCH_PARAMS:
            continue
        v = normalize_text_param(v)  # 规范化文本参数，避免编码差异导致同义 URL 不一致。
        if not v:
            continue
        query_map[k] = v

    if page_no is not None:
        page_no = max(1, int(page_no))
        if page_no > 1:
            query_map["page"] = str(page_no)
        else:
            query_map.pop("page", None)
    else:
        if not keep_page:
            query_map.pop("page", None)
        else:
            page_val = (query_map.get("page") or "").strip()
            if not page_val.isdigit() or int(page_val) <= 1:
                query_map.pop("page", None)
            else:
                query_map["page"] = str(int(page_val))

    ordered_pairs = []
    for key in ALLOWED_SEARCH_PARAMS:
        if key in query_map:
            ordered_pairs.append((key, query_map[key]))

    canonical_query = urlencode(ordered_pairs, doseq=False)
    return urlunparse((scheme, netloc, path, "", canonical_query, ""))


def canonical_task_url(item_url: str) -> str:
    """任务唯一键使用的 URL：不保留 page。"""
    return canonicalize_search_url(item_url, page_no=None, keep_page=False)


def normalize_detail_url(detail_url: str) -> str:
    """规范化详情页 URL，去掉 query 和 fragment，统一绝对路径。"""
    if not detail_url:
        return ""
    detail_url = detail_url.strip()

    if detail_url.startswith("//"):
        detail_url = "https:" + detail_url
    elif detail_url.startswith("/"):
        detail_url = "https://www.shixiseng.com" + detail_url

    parsed = urlparse(detail_url)

    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or "www.shixiseng.com"
    path = parsed.path or ""

    return urlunparse((scheme, netloc, path, "", "", ""))


def sleep_jitter(base=1.0, jitter_min=0.2, jitter_max=0.8):
    """
    带随机抖动的休眠函数，用于模拟人类操作行为，降低被反爬机制拦截的风险。
    
    Args:
        base (float): 基础休眠秒数。
        jitter_min (float): 随机抖动下限秒数。
        jitter_max (float): 随机抖动上限秒数。
        
    Returns:
        float: 实际休眠的总秒数。
    """
    """
    通用随机等待。

    最终等待时间 = base + [jitter_min, jitter_max] 区间内的随机扰动

    参数：
    - base: 基础等待时间
    - jitter_min / jitter_max: 在基础时间上叠加的随机抖动范围

    返回：
    - 实际等待的秒数
    """
    delay = base + random.uniform(jitter_min, jitter_max)
    time.sleep(delay)
    return delay


def sleep_short():
    """ 
    短间隔睡眠，用于等待页面加载、元素出现等场景。
    """
    return sleep_jitter(base=0.15, jitter_min=0.05, jitter_max=0.15)


def sleep_medium():
    """
    中间隔睡眠，用于等待列表页加载、分页等场景。
    """
    return sleep_jitter(base=0.35, jitter_min=0.10, jitter_max=0.35)


def sleep_detail_gap():
    """
    详情页之间的等待，用于等待详情页加载、字段提取等场景。
    """
    return sleep_jitter(base=0.45, jitter_min=0.15, jitter_max=0.55)


def sleep_page_gap():
    """
    每完成一页搜索结果后的等待，用于等待页面加载、分页等场景。
    """
    return sleep_jitter(base=0.70, jitter_min=0.20, jitter_max=0.80)


def sleep_task_gap():
    """
    每个任务之间的等待，用于等待任务加载、字段提取等场景。
    """
    return sleep_jitter(base=1.00, jitter_min=0.30, jitter_max=1.20)


def sleep_retry_backoff(attempt_index):
    """
    重试等待，用于等待重试操作。
    """
    base = min(0.8 + attempt_index * 0.9, 1.5)
    return sleep_jitter(base=base, jitter_min=0.15, jitter_max=0.80)


def now_str() -> str:
    """
    获取当前时间的标准格式化字符串。
    
    Returns:
        str: 格式如 "YYYY-MM-DD HH:MM:SS" 的当前时间字符串。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    """
    全局标准化日志输出函数。
    统一附加当前时间戳，并强制刷新输出流，确保多进程下日志不丢失。
    
    Args:
        msg (str): 需要输出的日志内容（通常应包含 Worker-ID、任务名、状态等）。
    """
    print(f"[{now_str()}] {msg}", flush=True)


def monotonic_now():
    """
    当前时间戳，用于记录程序运行时间。
    """
    return time.perf_counter()


def format_seconds(seconds):
    """
    格式化秒数，用于记录程序运行时间。
    """
    return f"{seconds:.2f}s"


def resolve_chromedriver_path_fast():
    """优先复用本地已有 driver，找不到时再走 webdriver_manager.install()。"""
    candidates = []  # 初始化列表，用于保存候选的 ChromeDriver 路径。

    env_path = os.environ.get("CHROMEDRIVER") or os.environ.get("WEBDRIVER_CHROME_DRIVER")
    if env_path:
        candidates.append(env_path)

    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, ".wdm", "drivers", "chromedriver"),
        os.path.join(home, ".cache", "selenium", "chromedriver"),
    ])

    for base in candidates:
        if not base:
            continue
        if os.path.isfile(base):
            return base
        if os.path.isdir(base):
            for root, _dirs, files in os.walk(base):
                for name in files:
                    lower = name.lower()
                    if lower == "chromedriver" or lower == "chromedriver.exe":
                        return os.path.join(root, name)

    return ChromeDriverManager().install()


# =========================
# MySQL 管理类
# =========================
class MySQLManager:
    """封装 MySQL 连接、建表、查询和写入操作。"""

    def __init__(self, config):
        self.config = config.copy()
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = mysql.connector.connect(**self.config)
            self.cursor = self.conn.cursor(dictionary=True)
            log(f"MySQL 已连接：{self.config.get('database')}")
        except Error as e:
            raise RuntimeError(f"MySQL 连接失败：{e}")

    def reconnect_if_needed(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.connect()
        except Exception:
            self.connect()

    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
        except Exception:
            pass

        try:
            if self.conn and self.conn.is_connected():
                self.conn.close()
                log("MySQL 已关闭。")
        except Exception:
            pass

    def create_tables(self):
        """创建程序运行所需的数据表，并统一修正唯一索引。"""

        create_job_details_sql = f"""
        CREATE TABLE IF NOT EXISTS `{OUTPUT_RESULT_TABLE}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            group_name VARCHAR(255) NOT NULL,
            item_text VARCHAR(255) NOT NULL,
            item_url TEXT NOT NULL,
            search_page INT DEFAULT NULL,
            intern_id VARCHAR(100) DEFAULT '',
            detail_url TEXT NOT NULL,
            job_title_detail VARCHAR(255) DEFAULT '',
            publish_time_detail VARCHAR(255) DEFAULT '',
            salary_detail VARCHAR(255) DEFAULT '',
            city_detail VARCHAR(255) DEFAULT '',
            degree_detail VARCHAR(255) DEFAULT '',
            days_per_week_detail VARCHAR(255) DEFAULT '',
            intern_months_detail VARCHAR(255) DEFAULT '',
            job_tags_detail TEXT,
            job_header_img_links TEXT,
            job_description LONGTEXT,
            resume_language VARCHAR(255) DEFAULT '',
            deadline_detail VARCHAR(255) DEFAULT '',
            job_address TEXT,
            company_name_detail VARCHAR(255) DEFAULT '',
            company_desc_detail TEXT,
            company_tags_detail TEXT,
            industry_detail VARCHAR(255) DEFAULT '',
            company_nature_detail VARCHAR(255) DEFAULT '',
            company_size_detail VARCHAR(255) DEFAULT '',
            company_location_detail VARCHAR(255) DEFAULT '',
            company_intro_img_links TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            KEY idx_item_text (item_text),
            KEY idx_detail_url (detail_url(255)),
            KEY idx_intern_id (intern_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        create_task_progress_sql = f"""
        CREATE TABLE IF NOT EXISTS `{TASK_PROGRESS_TABLE}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            group_name VARCHAR(255) NOT NULL,
            item_text VARCHAR(255) NOT NULL,
            item_url TEXT NOT NULL,
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            last_completed_page INT NOT NULL DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_task (group_name, item_text, item_url(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        create_detail_progress_sql = f"""
        CREATE TABLE IF NOT EXISTS `{DETAIL_PROGRESS_TABLE}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            detail_url TEXT NOT NULL,
            group_name VARCHAR(255) DEFAULT '',
            item_text VARCHAR(255) DEFAULT '',
            item_url TEXT,
            completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            KEY idx_detail_progress_url (detail_url(255)),
            KEY idx_detail_progress_item_text (item_text)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        self.cursor.execute(create_job_details_sql)
        self.cursor.execute(create_task_progress_sql)
        self.cursor.execute(create_detail_progress_sql)
        self.conn.commit()

        self._ensure_result_table_unique_index()
        self._ensure_detail_progress_unique_index()

        log("MySQL 数据表已就绪。")

    def _get_index_names(self, table_name):
        self.cursor.execute(f"SHOW INDEX FROM `{table_name}`")
        rows = self.cursor.fetchall()
        return {row.get("Key_name") for row in rows}

    def _drop_index_if_exists(self, table_name, index_name):
        try:
            index_names = self._get_index_names(table_name)
            if index_name in index_names:
                self.cursor.execute(f"ALTER TABLE `{table_name}` DROP INDEX `{index_name}`")
                self.conn.commit()
                log(f"已删除索引：{table_name}.{index_name}")
        except Exception as e:
            log(f"删除索引失败（忽略）：{table_name}.{index_name} -> {e}")

    def _ensure_result_table_unique_index(self):
        """
        结果表统一为：
        UNIQUE KEY uniq_item_text_detail_url (item_text, detail_url(255))
        """
        old_indexes = [
            "uniq_detail_url",
            "uniq_item_text_url",
            "uniq_group_item_url",
            "uniq_task_detail",
        ]
        for idx in old_indexes:
            self._drop_index_if_exists(OUTPUT_RESULT_TABLE, idx)

        index_names = self._get_index_names(OUTPUT_RESULT_TABLE)
        if "uniq_item_text_detail_url" not in index_names:
            self.cursor.execute(
                f"""
                ALTER TABLE `{OUTPUT_RESULT_TABLE}`
                ADD UNIQUE KEY `uniq_item_text_detail_url`
                (`item_text`, `detail_url`(255))
                """
            )
            self.conn.commit()
            log(f"已创建索引：{OUTPUT_RESULT_TABLE}.uniq_item_text_detail_url")

    def _ensure_detail_progress_unique_index(self):
        """
        详情进度表统一为：
        UNIQUE KEY uniq_item_text_detail_progress (item_text, detail_url(255))
        """
        old_indexes = [
            "uniq_detail_progress",
            "uniq_detail_url",
        ]
        for idx in old_indexes:
            self._drop_index_if_exists(DETAIL_PROGRESS_TABLE, idx)

        index_names = self._get_index_names(DETAIL_PROGRESS_TABLE)
        if "uniq_item_text_detail_progress" not in index_names:
            self.cursor.execute(
                f"""
                ALTER TABLE `{DETAIL_PROGRESS_TABLE}`
                ADD UNIQUE KEY `uniq_item_text_detail_progress`
                (`item_text`, `detail_url`(255))
                """
            )
            self.conn.commit()
            log(f"已创建索引：{DETAIL_PROGRESS_TABLE}.uniq_item_text_detail_progress")

    def load_input_tasks(self):
        """
        从输入任务表中读取待抓取任务，并对搜索 URL 做规范化去重。
        """
        sql = f"""
        SELECT DISTINCT group_name, item_text, item_url
        FROM `{INPUT_TASK_TABLE}`
        ORDER BY group_name ASC, item_text ASC, item_url ASC
        """
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()

        normalized = []
        seen = set()

        for row in rows:
            group_name = (row.get("group_name") or "").strip()
            item_text = (row.get("item_text") or "").strip()
            item_url = canonical_task_url(row.get("item_url") or "")

            key = (group_name, item_text, item_url)
            if key in seen:
                continue
            seen.add(key)

            normalized.append({
                "group_name": group_name,
                "item_text": item_text,
                "item_url": item_url,
            })

        return normalized

    def load_task_progress(self):
        """
        读取任务进度表，并转换为字典。
        """
        sql = f"""
        SELECT group_name, item_text, item_url, status, last_completed_page, updated_at
        FROM `{TASK_PROGRESS_TABLE}`
        """
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()

        progress = {}
        for row in rows:
            key = (
                (row.get("group_name") or "").strip(),
                (row.get("item_text") or "").strip(),
                canonical_task_url(row.get("item_url") or "")
            )
            progress[key] = {
                "status": row.get("status", ""),
                "last_completed_page": int(row.get("last_completed_page") or 0),
                "updated_at": str(row.get("updated_at") or "")
            }
        return progress

    def upsert_task_progress(self, group_name, item_text, item_url, status, last_completed_page):
        item_url = canonical_task_url(item_url)
        sql = f"""
        INSERT INTO `{TASK_PROGRESS_TABLE}`
            (group_name, item_text, item_url, status, last_completed_page, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            last_completed_page = VALUES(last_completed_page),
            updated_at = VALUES(updated_at)
        """
        self.cursor.execute(
            sql,
            (group_name, item_text, item_url, status, int(last_completed_page), now_str())
        )
        self.conn.commit()

    def load_done_detail_keys(self):
        """
        读取所有已成功抓取过的 (item_text, detail_url) 联合键。
        """
        done = set()

        sql1 = f"SELECT item_text, detail_url FROM `{OUTPUT_RESULT_TABLE}`"
        self.cursor.execute(sql1)
        for row in self.cursor.fetchall():
            item_text = (row.get("item_text") or "").strip()
            detail_url = normalize_detail_url((row.get("detail_url") or "").strip())
            if item_text and detail_url:
                done.add((item_text, detail_url))

        sql2 = f"SELECT item_text, detail_url FROM `{DETAIL_PROGRESS_TABLE}`"
        self.cursor.execute(sql2)
        for row in self.cursor.fetchall():
            item_text = (row.get("item_text") or "").strip()
            detail_url = normalize_detail_url((row.get("detail_url") or "").strip())
            if item_text and detail_url:
                done.add((item_text, detail_url))

        return done

    def upsert_result_row(self, row):
        """
        将单条职位详情记录写入结果表。
        结果表按 item_text + detail_url 联合去重。
        """
        sql = f"""
        INSERT INTO `{OUTPUT_RESULT_TABLE}` (
            group_name,
            item_text,
            item_url,
            search_page,
            intern_id,
            detail_url,
            job_title_detail,
            publish_time_detail,
            salary_detail,
            city_detail,
            degree_detail,
            days_per_week_detail,
            intern_months_detail,
            job_tags_detail,
            job_header_img_links,
            job_description,
            resume_language,
            deadline_detail,
            job_address,
            company_name_detail,
            company_desc_detail,
            company_tags_detail,
            industry_detail,
            company_nature_detail,
            company_size_detail,
            company_location_detail,
            company_intro_img_links
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            group_name = VALUES(group_name),
            item_url = VALUES(item_url),
            search_page = VALUES(search_page),
            intern_id = VALUES(intern_id),
            job_title_detail = VALUES(job_title_detail),
            publish_time_detail = VALUES(publish_time_detail),
            salary_detail = VALUES(salary_detail),
            city_detail = VALUES(city_detail),
            degree_detail = VALUES(degree_detail),
            days_per_week_detail = VALUES(days_per_week_detail),
            intern_months_detail = VALUES(intern_months_detail),
            job_tags_detail = VALUES(job_tags_detail),
            job_header_img_links = VALUES(job_header_img_links),
            job_description = VALUES(job_description),
            resume_language = VALUES(resume_language),
            deadline_detail = VALUES(deadline_detail),
            job_address = VALUES(job_address),
            company_name_detail = VALUES(company_name_detail),
            company_desc_detail = VALUES(company_desc_detail),
            company_tags_detail = VALUES(company_tags_detail),
            industry_detail = VALUES(industry_detail),
            company_nature_detail = VALUES(company_nature_detail),
            company_size_detail = VALUES(company_size_detail),
            company_location_detail = VALUES(company_location_detail),
            company_intro_img_links = VALUES(company_intro_img_links),
            updated_at = CURRENT_TIMESTAMP
        """

        values = (
            row.get("group_name", ""),
            row.get("item_text", ""),
            canonical_task_url(row.get("item_url", "")),
            int(row.get("search_page") or 0) if str(row.get("search_page", "")).strip() else None,
            row.get("intern_id", ""),
            row.get("detail_url", ""),
            row.get("job_title_detail", ""),
            row.get("publish_time_detail", ""),
            row.get("salary_detail", ""),
            row.get("city_detail", ""),
            row.get("degree_detail", ""),
            row.get("days_per_week_detail", ""),
            row.get("intern_months_detail", ""),
            row.get("job_tags_detail", ""),
            row.get("job_header_img_links", ""),
            row.get("job_description", ""),
            row.get("resume_language", ""),
            row.get("deadline_detail", ""),
            row.get("job_address", ""),
            row.get("company_name_detail", ""),
            row.get("company_desc_detail", ""),
            row.get("company_tags_detail", ""),
            row.get("industry_detail", ""),
            row.get("company_nature_detail", ""),
            row.get("company_size_detail", ""),
            row.get("company_location_detail", ""),
            row.get("company_intro_img_links", "")
        )
        self.cursor.execute(sql, values)
        self.conn.commit()

    def upsert_detail_progress(self, detail_url, group_name, item_text, item_url):
        """
        将单条 (item_text, detail_url) 的完成状态写入详情进度表。
        """
        detail_url = normalize_detail_url(detail_url)
        item_url = canonical_task_url(item_url)
        sql = f"""
        INSERT INTO `{DETAIL_PROGRESS_TABLE}`
            (detail_url, group_name, item_text, item_url, completed_at)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            group_name = VALUES(group_name),
            item_url = VALUES(item_url),
            completed_at = VALUES(completed_at)
        """
        self.cursor.execute(
            sql,
            (detail_url, group_name, item_text, item_url, now_str())
        )
        self.conn.commit()


# =========================
# Selenium Spider
# =========================
class ShiXiSengSeleniumSpider:
    """封装实习僧列表页和详情页的浏览器操作。"""

    def __init__(self, driver_path, headless=False, wait_seconds=15, window_rect=None):
        startup_begin = monotonic_now()
        options = webdriver.ChromeOptions()
        options.page_load_strategy = "eager"

        if headless:
            options.add_argument("--headless=new")

        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--lang=zh-CN")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-default-apps")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--window-size=1400,900")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )

        prefs = {
            "profile.default_content_setting_values.images": 2,
            "profile.managed_default_content_settings.images": 2,
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
        }
        options.add_experimental_option("prefs", prefs)

        # 降低“被弹到最前面”的体感
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)

        self.driver = webdriver.Chrome(
            service=Service(driver_path),
            options=options
        )
        self.driver.set_page_load_timeout(max(20, wait_seconds + 5))
        self.driver.set_script_timeout(max(20, wait_seconds + 5))

        self.wait = WebDriverWait(self.driver, wait_seconds)
        self.wait_quick = WebDriverWait(self.driver, min(6, wait_seconds))

        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                    """
                }
            )
        except Exception:
            pass

        if window_rect and not headless:
            width = max(700, int(window_rect["width"]))
            height = max(420, int(window_rect["height"]))
            x = int(window_rect["x"])
            y = int(window_rect["y"])

            try:
                self.driver.set_window_rect(x=x, y=y, width=width, height=height)
            except Exception:
                try:
                    self.driver.set_window_position(x, y)
                    self.driver.set_window_size(width, height)
                except Exception:
                    pass

        self.list_tab = self.driver.current_window_handle
        self.detail_tab = None  # 初始化详情标签页。
        self.switch_to_list_tab()  # 切换到列表标签页。

        log(f"浏览器窗口已显示，启动耗时：{format_seconds(monotonic_now() - startup_begin)}")

    def switch_to_list_tab(self):
        self.driver.switch_to.window(self.list_tab)

    def ensure_detail_tab(self):
        """
        确保详情标签页存在，不存在时创建。
        """
        if self.detail_tab and self.detail_tab in self.driver.window_handles:  # 如果详情标签页存在，则返回详情标签页。
            return self.detail_tab  # 返回详情标签页。

        try:
            self.driver.switch_to.new_window("tab")  # 创建新标签页。
        except Exception:
            self.driver.execute_script("window.open('about:blank', '_blank');")  # 执行 JavaScript 脚本。
            handles = self.driver.window_handles
            self.driver.switch_to.window(handles[-1])  # 切换到新创建的标签页。

        self.detail_tab = self.driver.current_window_handle
        return self.detail_tab  # 返回详情标签页。

    def switch_to_detail_tab(self):
        self.ensure_detail_tab()
        self.driver.switch_to.window(self.detail_tab)

    def _clean_text(self, text):
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _first_text(self, node, selector):
        target = node.select_one(selector)
        if not target:
            return ""
        return self._clean_text(target.get_text(" ", strip=True))

    def is_driver_alive(self):
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            return False

    def close_footer_login_popup_if_exists(self):
        try:
            close_btn = self.wait_quick.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "div.footer-login.footer-login--is-show img.footer-login__close")
                )
            )

            try:
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                    close_btn
                )

            except Exception:
                pass

            try:
                close_btn.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", close_btn)

            log("已关闭 footer-login 登录弹窗。")
        except Exception:
            pass

    def build_page_url(self, item_url, page_no):
        return canonicalize_search_url(item_url, page_no=page_no, keep_page=True)

    def open_list_page(self, url):
        self.switch_to_list_tab()
        try:
            self.driver.get(url)
        except TimeoutException:
            log(f"列表页加载超时，继续等待关键节点：{url}")
        self.close_footer_login_popup_if_exists()
        return self.wait_for_list_ready()

    def _is_list_page_empty(self):
        self.switch_to_list_tab()
        try:
            cards = self.driver.find_elements(By.CSS_SELECTOR, "div.intern-wrap.intern-item")
            if cards:
                return False
        except Exception:
            pass

        empty_selectors = [
            ".search_empty",
            ".empty",
            ".no-data",
            ".no-result",
            ".intern-empty",
            ".none",
        ]
        for selector in empty_selectors:
            try:
                nodes = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if nodes:
                    return True
            except Exception:
                pass

        try:
            body_text = self.driver.find_element(By.TAG_NAME, "body").text
        except Exception:
            body_text = self.driver.page_source or ""

        empty_markers = [
            "暂无",
            "没有找到",
            "无相关",
            "暂无实习",
            "暂无岗位",
            "暂无数据",
            "空空如也",
            "没有内容",
            "未找到",
        ]
        return any(marker in body_text for marker in empty_markers)

    def wait_for_list_ready(self):
        self.switch_to_list_tab()

        for _ in range(12):
            try:
                cards = self.driver.find_elements(By.CSS_SELECTOR, "div.intern-wrap.intern-item")
                if cards:
                    self.close_footer_login_popup_if_exists()
                    return "items"
            except Exception:
                pass

            if self._is_list_page_empty():
                self.close_footer_login_popup_if_exists()
                return "empty"

            time.sleep(0.4)

        if self._is_list_page_empty():
            self.close_footer_login_popup_if_exists()
            return "empty"

        try:
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.intern-wrap.intern-item"))
            )
            self.close_footer_login_popup_if_exists()
            return "items"
        except TimeoutException:
            if self._is_list_page_empty():
                self.close_footer_login_popup_if_exists()
                return "empty"
            return "timeout"

    def get_current_page_no(self):
        self.switch_to_list_tab()
        try:
            active = self.driver.find_element(
                By.CSS_SELECTOR,
                ".el-pagination .el-pager li.number.active"
            )
            return int(active.text.strip())
        except Exception:
            try:
                parsed = urlparse(self.driver.current_url)
                qs = parse_qs(parsed.query)
                page_val = qs.get("page", ["1"])[0]
                return int(page_val)
            except Exception:
                return 1

    def get_total_pages(self):
        self.switch_to_list_tab()
        try:
            nums = self.driver.find_elements(By.CSS_SELECTOR, ".el-pagination .el-pager li.number")
            page_nums = []
            for n in nums:
                txt = n.text.strip()
                if txt.isdigit():
                    page_nums.append(int(txt))
            return max(page_nums) if page_nums else 1
        except Exception:
            return 1

    def click_next_page(self):
        self.switch_to_list_tab()
        old_page = self.get_current_page_no()

        next_btn = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".el-pagination button.btn-next"))
        )

        if next_btn.get_attribute("disabled") is not None or not next_btn.is_enabled():
            return False

        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
        sleep_short()

        self.driver.execute_script("arguments[0].click();", next_btn)
        self.wait.until(lambda d: self.get_current_page_no() != old_page)

        self.wait_for_list_ready()
        self.close_footer_login_popup_if_exists()
        return True

    def parse_list_page_links(self, group_name, item_text, item_url, current_page):
        self.switch_to_list_tab()
        soup = BeautifulSoup(self.driver.page_source, "lxml")
        cards = soup.select("div.intern-wrap.intern-item")

        items = []

        for card in cards:
            intern_id = card.get("data-intern-id", "").strip()

            title_a = card.select_one(".intern-detail__job a.title")
            if not title_a:
                continue

            href = title_a.get("href", "").strip()

            if href.startswith("//"):
                detail_url = "https:" + href
            elif href.startswith("/"):
                detail_url = "https://www.shixiseng.com" + href
            else:
                detail_url = href

            detail_url = normalize_detail_url(detail_url)
            if not detail_url:
                continue

            items.append({
                "group_name": group_name,
                "item_text": item_text,
                "item_url": item_url,
                "search_page": current_page,
                "intern_id": intern_id,
                "detail_url": normalize_detail_url(detail_url)
            })

        return items

    def wait_for_detail_ready(self):
        self.switch_to_detail_tab()
        self.wait.until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.job-header, div.content_left, div.job-about")
            )
        )
        self.close_footer_login_popup_if_exists()

    def _extract_detail_from_html(self, soup):
        result = {
            "job_title_detail": "",
            "publish_time_detail": "",
            "salary_detail": "",
            "city_detail": "",
            "degree_detail": "",
            "days_per_week_detail": "",
            "intern_months_detail": "",
            "job_tags_detail": "",
            "job_header_img_links": "",
            "job_description": "",
            "resume_language": "",
            "deadline_detail": "",
            "job_address": "",
            "company_name_detail": "",
            "company_desc_detail": "",
            "company_tags_detail": "",
            "industry_detail": "",
            "company_nature_detail": "",
            "company_size_detail": "",
            "company_location_detail": "",
            "company_intro_img_links": ""
        }

        header = soup.select_one("div.job-header")  # 获取职位头部区域。
        if header:  # 如果职位头部区域存在，则获取职位头部区域的内容。
            result["job_title_detail"] = self._first_text(header, ".new_job_name span")  # 获取职位标题。
            result["publish_time_detail"] = self._first_text(header, ".job_date .cutom_font")  # 获取发布时间。
            result["salary_detail"] = self._first_text(header, ".job_msg .job_money")  # 获取薪资。
            result["city_detail"] = self._first_text(header, ".job_msg .job_position")  # 获取城市。
            result["degree_detail"] = self._first_text(header, ".job_msg .job_academic")  # 获取学历。
            result["days_per_week_detail"] = self._first_text(header, ".job_msg .job_week")  # 获取每周工作天数。
            result["intern_months_detail"] = self._first_text(header, ".job_msg .job_time")  # 获取实习月数。

            tag_nodes = header.select(".job_good_list span")
            tags = []
            for node in tag_nodes:
                txt = self._clean_text(node.get_text(" ", strip=True))
                if txt:
                    tags.append(txt)
            result["job_tags_detail"] = "、".join(tags)

            header_img_links = []
            new_job_name = header.select_one(".new_job_name")
            if new_job_name:
                title_span = new_job_name.select_one("span")
                if title_span:
                    for sib in title_span.next_siblings:
                        if getattr(sib, "name", None) == "img":
                            src = self._clean_text(sib.get("src", ""))
                            if src:
                                header_img_links.append(src)
            result["job_header_img_links"] = " | ".join(dict.fromkeys(header_img_links))

        content_left = soup.select_one("div.content_left")
        if content_left:
            con_jobs = content_left.select("div.con-job")

            for block in con_jobs:
                block_title = self._first_text(block, ".job_til")

                if "职位描述" in block_title:
                    detail_node = block.select_one(".job_detail")
                    if detail_node:
                        result["job_description"] = self._clean_text(
                            detail_node.get_text("\n", strip=True)
                        )

                elif "投递要求" in block_title:
                    block_text = self._clean_text(block.get_text(" ", strip=True))

                    m = re.search(r"简历要求：?\s*(.*?)(?:\s+截止日期|$)", block_text)
                    if m:
                        result["resume_language"] = self._clean_text(m.group(1))

                    deadline_node = block.select_one(".cutom_font")
                    if deadline_node:
                        deadline_text = self._clean_text(
                            deadline_node.get_text(" ", strip=True)
                        )
                        deadline_text = re.sub(r"^截止日期：\s*", "", deadline_text)
                        result["deadline_detail"] = deadline_text

                elif "工作地点" in block_title:
                    location_node = block.select_one(".com_position")
                    if location_node:
                        result["job_address"] = self._clean_text(
                            location_node.get_text(" ", strip=True)
                        )

        about = soup.select_one("div.job-about")
        if about:
            result["company_name_detail"] = self._first_text(about, ".com_intro .com-name")
            result["company_desc_detail"] = self._first_text(about, ".com_intro .com-desc")

            company_intro_img_links = []
            com_intro = about.select_one(".com_intro")
            if com_intro:
                started = False
                for child in com_intro.children:
                    child_name = getattr(child, "name", None)
                    if not child_name:
                        continue

                    classes = child.get("class", []) or []

                    if child_name == "a" and (
                            "com-name" in classes or "com-name--with-label" in classes
                    ):
                        started = True
                        continue

                    if child_name == "div" and "com-desc" in classes:
                        break

                    if started and child_name == "img":
                        src = self._clean_text(child.get("src", ""))
                        if src:
                            company_intro_img_links.append(src)

            result["company_intro_img_links"] = " | ".join(dict.fromkeys(company_intro_img_links))

            tag_nodes = about.select(".com-tags > div")
            company_tags = []
            for node in tag_nodes:
                txt = self._clean_text(node.get_text(" ", strip=True))
                if txt:
                    company_tags.append(txt)
            result["company_tags_detail"] = "、".join(company_tags)

            detail_nodes = about.select(".com-detail > div")
            detail_values = []
            for node in detail_nodes:
                txt = self._clean_text(node.get_text(" ", strip=True))
                if txt:
                    detail_values.append(txt)

            if len(detail_values) >= 1:
                result["industry_detail"] = detail_values[0]
            if len(detail_values) >= 2:
                result["company_nature_detail"] = detail_values[1]
            if len(detail_values) >= 3:
                result["company_size_detail"] = detail_values[2]
            if len(detail_values) >= 4:
                result["company_location_detail"] = detail_values[3]

        return result

    def visit_detail_page_and_extract_no_back(self, detail_url):
        self.switch_to_detail_tab()
        try:
            self.driver.get(detail_url)
        except TimeoutException:
            log(f"详情页加载超时，继续提取：{detail_url}")
        self.close_footer_login_popup_if_exists()
        self.wait_for_detail_ready()

        soup = BeautifulSoup(self.driver.page_source, "lxml")
        detail_data = self._extract_detail_from_html(soup)
        return detail_data

    def safe_visit_detail_page_and_extract_no_back(self, detail_url, retry=2):
        last_error = ""
        for i in range(retry + 1):
            try:
                return self.visit_detail_page_and_extract_no_back(detail_url)
            except Exception as e:
                last_error = str(e)
                log(f"访问详情页失败，第 {i + 1} 次：{detail_url} -> {e}")
                sleep_retry_backoff(i)

                if not self.is_driver_alive():
                    raise

        return {"error": last_error}

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


# =========================
# 任务与窗口分配
# =========================
def split_list_keep_worker_count(data, worker_count):
    result = [[] for _ in range(worker_count)]
    for idx, item in enumerate(data):
        result[idx % worker_count].append(item)
    return result


def make_task_key(task):
    return (
        (task.get("group_name") or "").strip(),
        (task.get("item_text") or "").strip(),
        canonical_task_url((task.get("item_url") or "").strip()),
    )


def task_key_to_str(task_key):
    return "|||".join(task_key)


def make_page_task(task, page_no, total_pages):
    return {
        "group_name": (task.get("group_name") or "").strip(),
        "item_text": (task.get("item_text") or "").strip(),
        "item_url": (task.get("item_url") or "").strip(),
        "page_no": int(page_no),
        "total_pages": int(total_pages),
    }


def claim_specific_task(shared_task_pool, shared_task_pool_lock, task_key):
    with shared_task_pool_lock:
        for idx, task in enumerate(list(shared_task_pool)):
            if make_task_key(task) == task_key:
                return shared_task_pool.pop(idx)
    return None


def claim_next_task(shared_task_pool, shared_task_pool_lock):
    with shared_task_pool_lock:
        if len(shared_task_pool) == 0:
            return None
        return shared_task_pool.pop(0)


def requeue_task(shared_task_pool, shared_task_pool_lock, task, worker_id=None, reason=""):
    task_key = make_task_key(task)
    with shared_task_pool_lock:
        for existing in list(shared_task_pool):
            if make_task_key(existing) == task_key:
                return False
        shared_task_pool.append(task)

    prefix = f"[Worker-{worker_id}] " if worker_id is not None else ""
    suffix = f" | 原因：{reason}" if reason else ""
    log(f"{prefix}任务重新放回共享池：{task_key[0]} | {task_key[1]}{suffix}")
    return True


def claim_next_page_task(shared_page_task_pool, shared_page_task_pool_lock):
    with shared_page_task_pool_lock:
        if len(shared_page_task_pool) == 0:
            return None
        return shared_page_task_pool.pop(0)


def requeue_page_task(shared_page_task_pool, shared_page_task_pool_lock, page_task, worker_id=None, reason=""):
    page_key = make_task_key(page_task) + (int(page_task.get("page_no") or 0),)
    with shared_page_task_pool_lock:
        for existing in list(shared_page_task_pool):
            existing_key = make_task_key(existing) + (int(existing.get("page_no") or 0),)
            if existing_key == page_key:
                return False
        shared_page_task_pool.append(page_task)

    prefix = f"[Worker-{worker_id}] " if worker_id is not None else ""
    suffix = f" | 原因：{reason}" if reason else ""
    log(
        f"{prefix}页面任务重新放回共享池：{page_task.get('group_name', '')} | "
        f"{page_task.get('item_text', '')} | 第 {page_task.get('page_no', '')} 页{suffix}"
    )
    return True


def get_shared_task_pool_size(shared_task_pool, shared_task_pool_lock):
    with shared_task_pool_lock:
        return len(shared_task_pool)


def get_shared_page_task_pool_size(shared_page_task_pool, shared_page_task_pool_lock):
    with shared_page_task_pool_lock:
        return len(shared_page_task_pool)


def get_or_init_runtime_state(shared_task_runtime, key_str):
    state = shared_task_runtime.get(key_str)
    if isinstance(state, dict):
        return dict(state)
    return {
        "status": "new",
        "total_pages": 0,
        "remaining_pages": 0,
        "resume_page": 1,
        "init_fail_count": 0,
        "page_fail_count": 0,
        "last_reason": "",
    }


def update_runtime_state(shared_task_runtime, key_str, **kwargs):
    state = get_or_init_runtime_state(shared_task_runtime, key_str)
    state.update(kwargs)
    shared_task_runtime[key_str] = state
    return state


def ensure_task_pages_initialized(
        worker_id,
        task,
        spider,
        db,
        shared_task_runtime,
        shared_task_runtime_lock,
        shared_page_task_pool,
        shared_page_task_pool_lock,
):
    task_key = make_task_key(task)
    key_str = task_key_to_str(task_key)
    max_init_fail_count = 3

    with shared_task_runtime_lock:
        state = get_or_init_runtime_state(shared_task_runtime, key_str)
        if state.get("status") in {"ready", "running", "done", "aborted"}:
            return state.get("status")
        if state.get("status") == "initializing":
            log(f"[Worker-{worker_id}] 页面任务正在由其他窗口初始化：{task_key[0]} | {task_key[1]}")
            return "busy"
        update_runtime_state(shared_task_runtime, key_str, status="initializing")

    group_name, item_text, item_url = task_key

    try:
        local_task_progress = db.load_task_progress()
        task_state = local_task_progress.get(task_key, {})
        if task_state.get("status") == "done":
            update_runtime_state(shared_task_runtime, key_str, status="done", remaining_pages=0)
            log(f"[Worker-{worker_id}] 跳过已完成任务：{group_name} | {item_text}")
            return "done"

        last_completed_page = int(task_state.get("last_completed_page", 0) or 0)
        resume_page = last_completed_page + 1

        probe_url = spider.build_page_url(item_url, 1)
        log(f"[Worker-{worker_id}] 初始化页面任务，先探测真实总页数：{probe_url}")
        page_state = spider.open_list_page(probe_url)

        if page_state == "empty":
            db.upsert_task_progress(
                group_name=group_name,
                item_text=item_text,
                item_url=item_url,
                status="done",
                last_completed_page=0
            )
            update_runtime_state(
                shared_task_runtime,
                key_str,
                status="done",
                total_pages=0,
                remaining_pages=0,
                resume_page=1,
                init_fail_count=0,
                last_reason="empty_task",
            )
            log(f"[Worker-{worker_id}] 首次进入任务即为空任务，直接标记完成：{group_name} | {item_text}")
            return "done"

        if page_state != "items":
            with shared_task_runtime_lock:
                state = get_or_init_runtime_state(shared_task_runtime, key_str)
                fail_count = int(state.get("init_fail_count", 0) or 0) + 1
                state.update({
                    "status": "new",
                    "init_fail_count": fail_count,
                    "last_reason": f"init_{page_state}",
                })
                shared_task_runtime[key_str] = state

            log(
                f"[Worker-{worker_id}] 初始化页面任务失败（状态={page_state}）：{group_name} | {item_text} | "
                f"累计失败 {fail_count}/{max_init_fail_count}"
            )

            if fail_count >= max_init_fail_count:
                db.upsert_task_progress(
                    group_name=group_name,
                    item_text=item_text,
                    item_url=item_url,
                    status="done",
                    last_completed_page=0
                )
                update_runtime_state(
                    shared_task_runtime,
                    key_str,
                    status="aborted",
                    total_pages=0,
                    remaining_pages=0,
                    resume_page=1,
                    init_fail_count=fail_count,
                    last_reason=f"init_{page_state}",
                )
                log(
                    f"[Worker-{worker_id}] 初始化连续失败达到上限，终止该任务避免无限循环："
                    f"{group_name} | {item_text}"
                )
                return "aborted"

            return "retry"

        total_pages = max(1, spider.get_total_pages())

        safe_resume_page = min(max(1, resume_page), total_pages)
        if safe_resume_page != resume_page:
            log(
                f"[Worker-{worker_id}] 断点页 {resume_page} 超过当前总页数 {total_pages}，"
                f"自动修正为第 {safe_resume_page} 页"
            )
            resume_page = safe_resume_page
            last_completed_page = max(0, resume_page - 1)
            db.upsert_task_progress(
                group_name=group_name,
                item_text=item_text,
                item_url=item_url,
                status="running",
                last_completed_page=last_completed_page
            )

        page_tasks = [make_page_task(task, page_no, total_pages) for page_no in range(resume_page, total_pages + 1)]

        if not page_tasks:
            db.upsert_task_progress(
                group_name=group_name,
                item_text=item_text,
                item_url=item_url,
                status="done",
                last_completed_page=total_pages
            )
            update_runtime_state(
                shared_task_runtime,
                key_str,
                status="done",
                total_pages=total_pages,
                remaining_pages=0,
                resume_page=resume_page,
                init_fail_count=0,
                last_reason="no_page_tasks",
            )
            log(f"[Worker-{worker_id}] 当前任务无可分配页，直接标记完成：{group_name} | {item_text}")
            return "done"

        with shared_page_task_pool_lock:
            shared_page_task_pool.extend(page_tasks)

        update_runtime_state(
            shared_task_runtime,
            key_str,
            status="ready",
            total_pages=total_pages,
            remaining_pages=len(page_tasks),
            resume_page=resume_page,
            init_fail_count=0,
            last_reason="initialized",
        )
        log(
            f"[Worker-{worker_id}] 已将页面任务加入共享池：{group_name} | {item_text} | "
            f"从第 {resume_page} 页到第 {total_pages} 页，共 {len(page_tasks)} 页"
        )
        return "ready"

    except Exception as e:
        with shared_task_runtime_lock:
            state = get_or_init_runtime_state(shared_task_runtime, key_str)
            fail_count = int(state.get("init_fail_count", 0) or 0) + 1
            state.update({
                "status": "new",
                "init_fail_count": fail_count,
                "last_reason": "init_exception",
            })
            shared_task_runtime[key_str] = state

        log(f"[Worker-{worker_id}] 初始化页面任务失败：{group_name} | {item_text} -> {e}")
        log(traceback.format_exc())

        if fail_count >= max_init_fail_count:
            db.upsert_task_progress(
                group_name=group_name,
                item_text=item_text,
                item_url=item_url,
                status="done",
                last_completed_page=0
            )
            update_runtime_state(
                shared_task_runtime,
                key_str,
                status="aborted",
                total_pages=0,
                remaining_pages=0,
                resume_page=1,
                init_fail_count=fail_count,
                last_reason="init_exception",
            )
            log(
                f"[Worker-{worker_id}] 初始化异常连续达到上限，终止该任务避免无限循环："
                f"{group_name} | {item_text}"
            )
            return "aborted"

        return "retry"


def finalize_task_if_needed(worker_id, page_task, db, shared_task_runtime, shared_task_runtime_lock):
    task_key = make_task_key(page_task)
    key_str = task_key_to_str(task_key)
    group_name, item_text, item_url = task_key

    with shared_task_runtime_lock:
        state = get_or_init_runtime_state(shared_task_runtime, key_str)
        remaining_pages = max(0, int(state.get("remaining_pages", 0)) - 1)
        state["remaining_pages"] = remaining_pages
        state["status"] = "running" if remaining_pages > 0 else "done"
        shared_task_runtime[key_str] = state
        total_pages = int(state.get("total_pages", page_task.get("total_pages") or 0) or 0)

    if remaining_pages == 0:
        db.upsert_task_progress(
            group_name=group_name,
            item_text=item_text,
            item_url=item_url,
            status="done",
            last_completed_page=max(1, total_pages)
        )
        log(f"[Worker-{worker_id}] 任务完成（页面任务全部清空）：{group_name} | {item_text}")


def force_finish_task(worker_id, page_task, db, shared_task_runtime, shared_task_runtime_lock, reason="",
                      last_completed_page=0):
    task_key = make_task_key(page_task)
    key_str = task_key_to_str(task_key)
    group_name, item_text, item_url = task_key

    with shared_task_runtime_lock:
        state = get_or_init_runtime_state(shared_task_runtime, key_str)
        state["remaining_pages"] = 0
        state["status"] = "done"
        if "total_pages" not in state or not state.get("total_pages"):
            state["total_pages"] = int(page_task.get("total_pages") or 0)
        shared_task_runtime[key_str] = state

    db.upsert_task_progress(
        group_name=group_name,
        item_text=item_text,
        item_url=item_url,
        status="done",
        last_completed_page=max(0, int(last_completed_page or 0))
    )

    suffix = f" | 原因：{reason}" if reason else ""
    log(f"[Worker-{worker_id}] 任务直接结束：{group_name} | {item_text}{suffix}")


def process_single_page_task(
        worker_id: int,
        page_task: dict,
        spider,
        db,
        shared_done_detail_dict: dict,
        shared_detail_lock,
        shared_task_runtime: dict,
        shared_task_runtime_lock
) -> bool:
    """
    处理单个页面任务。
    加载搜索列表页，提取详情链接，并逐个访问详情页进行数据抓取。
    
    Args:
        worker_id (int): 进程 ID。
        page_task (dict): 页面任务字典。
        spider (ShiXiSengSeleniumSpider): 爬虫实例。
        db (MySQLManager): 数据库管理实例。
        shared_done_detail_dict (dict): 共享的已完成详情进度。
        shared_detail_lock (Lock): 详情进度锁。
        shared_task_runtime (dict): 共享的任务运行时状态。
        shared_task_runtime_lock (Lock): 运行时状态锁。
        
    Returns:
        bool: 处理是否成功。
    """

    group_name = (page_task.get("group_name") or "").strip()
    item_text = (page_task.get("item_text") or "").strip()
    item_url = (page_task.get("item_url") or "").strip()
    current_page = int(page_task.get("page_no") or 1)
    total_pages = int(page_task.get("total_pages") or 1)

    try:
        page_url = spider.build_page_url(item_url, current_page)
        log(
            f"[Worker-{worker_id}] 页面协作任务：{group_name} | {item_text} | "
            f"第 {current_page}/{total_pages} 页"
        )
        page_state = spider.open_list_page(page_url)
        if page_state == "empty":
            stop_at_page = max(0, current_page - 1)
            reason = "第一页为空，判定为空任务" if current_page == 1 else f"第 {current_page} 页为空，停止该任务后续重试"
            force_finish_task(
                worker_id=worker_id,
                page_task=page_task,
                db=db,
                shared_task_runtime=shared_task_runtime,
                shared_task_runtime_lock=shared_task_runtime_lock,
                reason=reason,
                last_completed_page=stop_at_page,
            )
            return True

        page_items = spider.parse_list_page_links(
            group_name=group_name,
            item_text=item_text,
            item_url=item_url,
            current_page=current_page
        )

        log(
            f"[Worker-{worker_id}] 当前页解析到详情数：{len(page_items)} | "
            f"{group_name} | {item_text} | 第 {current_page} 页"
        )

        if not page_items:
            stop_at_page = max(0, current_page - 1)
            reason = "第一页解析结果为空，判定为空任务" if current_page == 1 else f"第 {current_page} 页解析结果为空，停止该任务后续重试"
            force_finish_task(
                worker_id=worker_id,
                page_task=page_task,
                db=db,
                shared_task_runtime=shared_task_runtime,
                shared_task_runtime_lock=shared_task_runtime_lock,
                reason=reason,
                last_completed_page=stop_at_page,
            )
            return True

        page_fully_done = True

        for idx, item in enumerate(page_items, start=1):
            detail_url = item.get("detail_url", "")
            detail_url = normalize_detail_url(detail_url)
            detail_key = (item_text, detail_url)

            with shared_detail_lock:
                already_done = detail_key in shared_done_detail_dict

            if already_done:
                log(
                    f"[Worker-{worker_id}] 已存在，跳过详情："
                    f"{idx}/{len(page_items)} {item_text} | {detail_url}"
                )
                continue

            log(
                f"[Worker-{worker_id}] 抓详情："
                f"{idx}/{len(page_items)} {item_text} | {detail_url}"
            )

            row = {
                "group_name": group_name,
                "item_text": item_text,
                "item_url": item_url,
                "search_page": current_page,
                "intern_id": item.get("intern_id", ""),
                "detail_url": detail_url
            }

            try:
                detail_data = spider.safe_visit_detail_page_and_extract_no_back(
                    detail_url,
                    retry=2
                )

                if detail_data.get("error"):
                    page_fully_done = False
                    log(
                        f"[Worker-{worker_id}] 详情失败，保留待重试："
                        f"{item_text} | {detail_url} | {detail_data.get('error', '')}"
                    )
                    continue

                row.update(detail_data)

                db.upsert_result_row(row)
                db.upsert_detail_progress(
                    detail_url=detail_url,
                    group_name=group_name,
                    item_text=item_text,
                    item_url=item_url
                )

                with shared_detail_lock:
                    shared_done_detail_dict[detail_key] = 1

                log(
                    f"[Worker-{worker_id}] 完成详情："
                    f"{row.get('job_title_detail', '')} | "
                    f"{row.get('salary_detail', '')} | "
                    f"{row.get('company_name_detail', '')}"
                )

            except Exception as e:
                page_fully_done = False
                log(f"[Worker-{worker_id}] 详情彻底失败：{item_text} | {detail_url} -> {e}")
                log(traceback.format_exc())

            sleep_detail_gap()

        if not page_fully_done:
            log(
                f"[Worker-{worker_id}] 本页未全部成功：{group_name} | {item_text} | 第 {current_page} 页，"
                f"稍后会重新排队"
            )
            return False

        finalize_task_if_needed(
            worker_id=worker_id,
            page_task=page_task,
            db=db,
            shared_task_runtime=shared_task_runtime,
            shared_task_runtime_lock=shared_task_runtime_lock,
        )
        sleep_page_gap()
        return True

    except Exception as e:
        log(f"[Worker-{worker_id}] 页面任务异常：{group_name} | {item_text} | 第 {current_page} 页 -> {e}")
        log(traceback.format_exc())
        return False


def get_screen_quadrants():
    work_left = 0
    work_top = 0
    work_right = 1920
    work_bottom = 1080

    try:
        import ctypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        SPI_GETWORKAREA = 48
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETWORKAREA, 0, ctypes.byref(rect), 0
        )

        if ok:
            work_left = rect.left
            work_top = rect.top
            work_right = rect.right
            work_bottom = rect.bottom

    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()

            work_left = 0
            work_top = 0
            work_right = screen_w
            work_bottom = max(800, screen_h - 80)
        except Exception:
            work_left = 0
            work_top = 0
            work_right = 1920
            work_bottom = 1000

    work_width = max(1200, work_right - work_left)
    work_height = max(800, work_bottom - work_top)

    outer_margin_x = 8
    outer_margin_y = 8
    inner_gap_x = 8
    inner_gap_y = 8

    usable_x = work_left + outer_margin_x
    usable_y = work_top + outer_margin_y
    usable_width = work_width - outer_margin_x * 2
    usable_height = work_height - outer_margin_y * 2

    left_width = max(600, (usable_width - inner_gap_x) // 2)
    right_width = max(600, usable_width - inner_gap_x - left_width)

    top_height = max(380, (usable_height - inner_gap_y) // 2)
    bottom_height = max(380, usable_height - inner_gap_y - top_height)

    return [
        {"x": usable_x, "y": usable_y, "width": left_width, "height": top_height},
        {"x": usable_x + left_width + inner_gap_x, "y": usable_y, "width": right_width, "height": top_height},
        {"x": usable_x, "y": usable_y + top_height + inner_gap_y, "width": left_width, "height": bottom_height},
        {"x": usable_x + left_width + inner_gap_x, "y": usable_y + top_height + inner_gap_y, "width": right_width,
         "height": bottom_height},
    ]


# =========================
# Worker
# =========================
def worker_process(
        worker_id: int,
        task_list: list,
        headless: bool,
        wait_seconds: int,
        window_rect: dict,
        mysql_config: dict,
        shared_done_detail_dict: dict,
        shared_detail_lock,
        shared_task_pool: list,
        shared_task_pool_lock,
        shared_page_task_pool: list,
        shared_page_task_pool_lock,
        shared_task_runtime: dict,
        shared_task_runtime_lock,
        driver_path: str
):
    """
    工作进程主函数。
    负责初始化 WebDriver、数据库连接，并循环从共享任务池中领取任务进行页面抓取。
    
    Args:
        worker_id (int): 进程 ID。
        task_list (list): 初始分配给该进程的静态任务列表。
        headless (bool): 是否使用无头模式。
        wait_seconds (int): 页面加载超时时间。
        window_rect (dict): 浏览器窗口位置和大小。
        mysql_config (dict): 数据库配置。
        shared_done_detail_dict (dict): 共享的已完成详情进度字典。
        shared_detail_lock (Lock): 详情进度字典锁。
        shared_task_pool (list): 共享的主任务池。
        shared_task_pool_lock (Lock): 主任务池锁。
        shared_page_task_pool (list): 共享的页面任务池。
        shared_page_task_pool_lock (Lock): 页面任务池锁。
        shared_task_runtime (dict): 共享的任务运行时状态。
        shared_task_runtime_lock (Lock): 运行时状态锁。
        driver_path (str): ChromeDriver 路径。
    """

    spider = None
    db = None

    try:
        pid = os.getpid()
        log(f"[Worker-{worker_id}] 子进程启动，PID={pid}")
        log(f"[Worker-{worker_id}] 初始分配任务数：{len(task_list)}")

        if not task_list and len(shared_task_pool) == 0 and len(shared_page_task_pool) == 0:
            log(f"[Worker-{worker_id}] 无任务，直接退出。")
            return

        browser_begin = monotonic_now()
        log(f"[Worker-{worker_id}] 开始初始化浏览器...")
        spider = ShiXiSengSeleniumSpider(
            driver_path=driver_path,
            headless=headless,
            wait_seconds=wait_seconds,
            window_rect=window_rect
        )
        log(f"[Worker-{worker_id}] 浏览器启动成功，总耗时：{format_seconds(monotonic_now() - browser_begin)}")

        log(f"[Worker-{worker_id}] 开始连接 MySQL...")
        db = MySQLManager(mysql_config)
        db.connect()
        log(f"[Worker-{worker_id}] MySQL 连接成功。")

        initialized_count = 0
        for task_index, raw_task in enumerate(task_list, start=1):
            task = claim_specific_task(
                shared_task_pool=shared_task_pool,
                shared_task_pool_lock=shared_task_pool_lock,
                task_key=make_task_key(raw_task)
            )
            if task is None:
                log(f"[Worker-{worker_id}] 初始任务 {task_index}/{len(task_list)} 已被其他窗口接手或完成，跳过。")
                continue

            log(f"[Worker-{worker_id}] 初始化初始任务 {task_index}/{len(task_list)} 的页面共享队列")
            init_state = ensure_task_pages_initialized(
                worker_id=worker_id,
                task=task,
                spider=spider,
                db=db,
                shared_task_runtime=shared_task_runtime,
                shared_task_runtime_lock=shared_task_runtime_lock,
                shared_page_task_pool=shared_page_task_pool,
                shared_page_task_pool_lock=shared_page_task_pool_lock,
            )
            if init_state in {"ready", "done"}:
                initialized_count += 1
            elif init_state == "retry":
                requeue_task(
                    shared_task_pool=shared_task_pool,
                    shared_task_pool_lock=shared_task_pool_lock,
                    task=task,
                    worker_id=worker_id,
                    reason="初始任务初始化失败，等待其他窗口后续接力"
                )

        helped_count = 0
        idle_rounds = 0
        max_idle_rounds = 15
        idle_sleep_seconds = 2.0

        while True:
            page_task = claim_next_page_task(shared_page_task_pool, shared_page_task_pool_lock)
            if page_task is not None:
                idle_rounds = 0
                helped_count += 1
                success = process_single_page_task(
                    worker_id=worker_id,
                    page_task=page_task,
                    spider=spider,
                    db=db,
                    shared_done_detail_dict=shared_done_detail_dict,
                    shared_detail_lock=shared_detail_lock,
                    shared_task_runtime=shared_task_runtime,
                    shared_task_runtime_lock=shared_task_runtime_lock,
                )
                if not success:
                    requeue_page_task(
                        shared_page_task_pool=shared_page_task_pool,
                        shared_page_task_pool_lock=shared_page_task_pool_lock,
                        page_task=page_task,
                        worker_id=worker_id,
                        reason="页面任务执行失败，重新排队"
                    )
                continue

            task = claim_next_task(shared_task_pool, shared_task_pool_lock)
            if task is not None:
                idle_rounds = 0
                key = make_task_key(task)
                log(
                    f"[Worker-{worker_id}] 开始支援其他窗口任务并初始化页级协作："
                    f"{key[0]} | {key[1]}"
                )
                init_state = ensure_task_pages_initialized(
                    worker_id=worker_id,
                    task=task,
                    spider=spider,
                    db=db,
                    shared_task_runtime=shared_task_runtime,
                    shared_task_runtime_lock=shared_task_runtime_lock,
                    shared_page_task_pool=shared_page_task_pool,
                    shared_page_task_pool_lock=shared_page_task_pool_lock,
                )
                if init_state == "retry":
                    requeue_task(
                        shared_task_pool=shared_task_pool,
                        shared_task_pool_lock=shared_task_pool_lock,
                        task=task,
                        worker_id=worker_id,
                        reason="支援任务初始化失败，重新排队"
                    )
                continue

            idle_rounds += 1
            remaining_menu = get_shared_task_pool_size(shared_task_pool, shared_task_pool_lock)
            remaining_pages = get_shared_page_task_pool_size(shared_page_task_pool, shared_page_task_pool_lock)
            if idle_rounds >= max_idle_rounds:
                log(
                    f"[Worker-{worker_id}] 页面共享池与任务池连续空闲 {idle_rounds} 轮，结束等待并退出。"
                )
                break

            log(
                f"[Worker-{worker_id}] 暂无可支援任务，等待第 {idle_rounds}/{max_idle_rounds} 轮，"
                f"{format_seconds(idle_sleep_seconds)} 后再检查。菜单任务数：{remaining_menu}，页面任务数：{remaining_pages}"
            )
            time.sleep(idle_sleep_seconds)

        log(
            f"[Worker-{worker_id}] 全部任务处理结束，初始化任务数：{initialized_count}，"
            f"处理页面任务数：{helped_count}"
        )

    except Exception as e:
        log(f"[Worker-{worker_id}] 启动或运行阶段发生未捕获异常：{e}")
        log(traceback.format_exc())

    finally:
        if spider is not None:
            try:
                spider.close()
                log(f"[Worker-{worker_id}] 浏览器已关闭。")
            except Exception as e:
                log(f"[Worker-{worker_id}] 关闭浏览器异常：{e}")

        if db is not None:
            try:
                db.close()
            except Exception as e:
                log(f"[Worker-{worker_id}] 关闭数据库异常：{e}")

        log(f"[Worker-{worker_id}] 进程结束。")


# =========================
# 主函数
# =========================
def main():
    """
    主函数。
    负责初始化数据库、加载任务、处理详情进度去重、启动并监控多进程工作池。
    """
    worker_count = 4  # 保留四个伪分布
    detail_headless = False
    wait_seconds = 15
    startup_stagger_seconds = 0.08

    log("主进程启动。")
    log("开始连接主 MySQL...")
    db = MySQLManager(MYSQL_CONFIG)
    db.connect()
    db.create_tables()

    all_tasks = db.load_input_tasks()
    if not all_tasks:
        log(
            f"未从表 {INPUT_TASK_TABLE} 读取到任何菜单任务。\n"
            "请先运行前面的菜单链接构造脚本，把 group_name/item_text/item_url 写入该表。"
        )
        db.close()
        return

    task_progress = db.load_task_progress()

    pending_tasks = []
    for task in all_tasks:
        key = (task["group_name"], task["item_text"], task["item_url"])
        state = task_progress.get(key, {})
        if state.get("status") != "done":
            pending_tasks.append(task)

    if not pending_tasks:
        log("所有菜单任务都已完成，无需继续爬取。")
        db.close()
        return

    log(f"总任务数：{len(all_tasks)}")
    log(f"待完成任务数：{len(pending_tasks)}")

    existing_done_detail_keys = db.load_done_detail_keys()
    log(f"已存在详情联合键数：{len(existing_done_detail_keys)}")

    db.close()

    # 主进程预解析 driver，只做一次
    log("主进程开始解析 ChromeDriver 路径...")
    try:
        driver_path = resolve_chromedriver_path_fast()
        log(f"ChromeDriver 路径解析成功：{driver_path}")
    except Exception as e:
        log(f"ChromeDriver 路径解析失败：{e}")
        log(traceback.format_exc())
        return

    window_rects = get_screen_quadrants()
    task_chunks = split_list_keep_worker_count(pending_tasks, worker_count)

    for i, chunk in enumerate(task_chunks, start=1):
        log(f"Worker-{i} 分配任务数：{len(chunk)}")

    manager = mp.Manager()
    shared_done_detail_dict = manager.dict({k: 1 for k in existing_done_detail_keys})
    shared_detail_lock = manager.Lock()
    shared_task_pool = manager.list(pending_tasks)
    shared_task_pool_lock = manager.Lock()
    shared_page_task_pool = manager.list()
    shared_page_task_pool_lock = manager.Lock()
    shared_task_runtime = manager.dict()
    shared_task_runtime_lock = manager.Lock()

    processes = []

    for i in range(worker_count):
        chunk = task_chunks[i]

        if not chunk:
            log(f"Worker-{i + 1} 无任务，跳过启动。")
            continue

        p = mp.Process(
            target=worker_process,
            args=(
                i + 1,
                chunk,
                detail_headless,
                wait_seconds,
                window_rects[i],
                MYSQL_CONFIG,
                shared_done_detail_dict,
                shared_detail_lock,
                shared_task_pool,
                shared_task_pool_lock,
                shared_page_task_pool,
                shared_page_task_pool_lock,
                shared_task_runtime,
                shared_task_runtime_lock,
                driver_path
            ),
            name=f"Worker-{i + 1}"
        )

        processes.append(p)

        log(f"准备启动 Worker-{i + 1} ...")
        p.start()
        log(f"Worker-{i + 1} 已启动，PID={p.pid}")

        # 轻量错峰：保留极短间隔，尽快把窗口都拉起来
        time.sleep(startup_stagger_seconds)

    for p in processes:
        p.join()
        log(f"{p.name} 已结束，exitcode={p.exitcode}")

    log("全部 worker 已结束。")
    log(f"输入任务表：{INPUT_TASK_TABLE}")
    log(f"结果表：{OUTPUT_RESULT_TABLE}")
    log(f"任务进度表：{TASK_PROGRESS_TABLE}")
    log(f"详情进度表：{DETAIL_PROGRESS_TABLE}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
