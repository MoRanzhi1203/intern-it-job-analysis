# -*- coding: utf-8 -*-
"""
实习僧“互联网IT”岗位分类链接采集与入库程序（全自动正式版）

功能概述：
1. 自动打开实习僧首页；
2. 自动关闭首页弹窗；
3. 自动悬停“互联网IT”一级菜单并展开子菜单；
4. 提取菜单中的分组标题及对应岗位名称；
5. 根据岗位名称构造实习搜索页链接；
6. 将采集结果写入 MySQL 数据库；
7. 输出采集结果供人工核验；
8. 程序结束后自动关闭浏览器与数据库连接。
"""

import os
import time
from urllib.parse import quote

import mysql.connector
from mysql.connector import Error

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# =========================
# 数据库配置
# =========================
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "shixiseng_db",
    "charset": "utf8mb4"
}


def resolve_chromedriver_path_fast() -> str:
    """
    解析并获取 ChromeDriver 的本地路径。
    优先复用本地环境变量或常见缓存目录中已有的 driver，找不到时再调用 webdriver_manager 进行安装。
    
    Returns:
        str: ChromeDriver 的绝对路径。
    """
    candidates = []

    env_path = os.environ.get("CHROMEDRIVER") or os.environ.get("WEBDRIVER_CHROME_DRIVER")
    if env_path:
        candidates.append(env_path)

    # 常见的本地路径缓存
    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, ".wdm", "drivers", "chromedriver"),
        os.path.join(home, ".cache", "selenium", "chromedriver"),
        os.getcwd(),
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

    # 若未找到本地缓存，则自动下载安装
    return ChromeDriverManager().install()


class MySQLManager:
    """
    MySQL 数据库管理器。
    负责处理数据库连接、数据表创建以及数据的批量 Upsert（插入或更新）操作。
    """

    def __init__(self, config: dict):
        """
        初始化数据库管理器。
        
        Args:
            config (dict): 包含 host, port, user, password, database, charset 的配置字典。
        """
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 3306)
        self.user = config.get("user", "root")
        self.password = config.get("password", "123456")
        self.database = config.get("database", "shixiseng_db")
        self.charset = config.get("charset", "utf8mb4")

        self.server_conn = None
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        建立 MySQL 数据库连接。如果目标数据库不存在，则自动创建。
        """
        try:
            # 首先连接到 MySQL 服务，不指定数据库，用于检查和创建数据库
            self.server_conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                charset=self.charset
            )

            if self.server_conn.is_connected():
                server_cursor = self.server_conn.cursor()
                server_cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                    f"DEFAULT CHARACTER SET {self.charset}"
                )
                server_cursor.close()
                self.server_conn.close()
                self.server_conn = None

            # 重新连接到目标数据库
            self.conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset
            )

            self.cursor = self.conn.cursor()
            print(f"[日志] MySQL 已成功连接至数据库：{self.database}")

        except Error as e:
            raise RuntimeError(f"[错误] MySQL 连接失败：{e}")

    def create_table(self):
        """
        创建用于存储实习僧菜单链接的数据表（如果不存在）。
        保证 group_name 和 item_text 联合唯一。
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS shixiseng_internet_it_links (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            group_name VARCHAR(255) NOT NULL,
            item_text VARCHAR(255) NOT NULL,
            item_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_group_item (group_name, item_text)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            self.cursor.execute(create_sql)
            self.conn.commit()
            print("[日志] 数据表 shixiseng_internet_it_links 检查/创建成功。")
        except Error as e:
            raise RuntimeError(f"[错误] 创建数据表失败：{e}")

    def upsert_rows(self, rows: list):
        """
        将采集到的菜单链接批量写入数据库。如果存在重复，则更新链接及时间戳。
        
        Args:
            rows (list): 包含待写入数据的字典列表。
        """
        if not rows:
            print("[日志] 暂无有效数据需要写入 MySQL。")
            return

        sql = """
        INSERT INTO shixiseng_internet_it_links (group_name, item_text, item_url)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            item_url = VALUES(item_url),
            updated_at = CURRENT_TIMESTAMP
        """

        values = [
            (row["group_name"], row["item_text"], row["item_url"])
            for row in rows
        ]

        try:
            self.cursor.executemany(sql, values)
            self.conn.commit()
            print(f"[日志] MySQL 写入成功：共计 {self.cursor.rowcount} 条变更（包含新增与更新）。")
        except Error as e:
            print(f"[错误] 数据批量写入 MySQL 失败：{e}")

    def close(self):
        """
        安全关闭数据库游标与连接。
        """
        try:
            if self.cursor:
                self.cursor.close()
        except Exception:
            pass

        try:
            if self.conn and self.conn.is_connected():
                self.conn.close()
                print("[日志] MySQL 连接已安全关闭。")
        except Exception:
            pass


class ShiXiSengCategoryUrlBuilder:
    """
    实习僧“互联网IT”分类链接采集器。
    封装了基于 Selenium 的浏览器自动化操作。
    """

    def __init__(self, driver_path: str, headless: bool = False, wait_seconds: int = 15):
        """
        初始化 Selenium WebDriver。
        
        Args:
            driver_path (str): ChromeDriver 可执行文件的路径。
            headless (bool): 是否启用无头模式。
            wait_seconds (int): 显式等待的最大超时秒数。
        """
        options = webdriver.ChromeOptions()

        if headless:
            options.add_argument("--headless=new")

        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--lang=zh-CN")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-sync")
        options.add_argument("--disable-notifications")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--log-level=3")
        options.add_argument("--window-size=1400,900")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )

        # 隐藏自动化特征
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)

        prefs = {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
        }
        options.add_experimental_option("prefs", prefs)

        service = Service(driver_path, log_output=os.devnull)

        self.driver = webdriver.Chrome(
            service=service,
            options=options
        )

        self.driver.set_page_load_timeout(max(20, wait_seconds + 5))
        self.driver.set_script_timeout(max(20, wait_seconds + 5))

        self.wait = WebDriverWait(self.driver, wait_seconds)
        self.actions = ActionChains(self.driver)
        self.base_url = "https://www.shixiseng.com/"

        # 注入脚本，进一步隐藏 WebDriver 特征
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

    def open_homepage(self):
        """
        打开实习僧官方首页。
        """
        self.driver.get(self.base_url)
        print(f"[日志] 已成功打开实习僧首页：{self.base_url}")

    def close_home_popup(self):
        """
        尝试关闭首页可能出现的广告或活动弹窗。
        """
        try:
            time.sleep(2)
            close_btn = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "div.openmask img.closeOpen"))
            )

            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                close_btn
            )
            time.sleep(0.3)

            try:
                close_btn.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", close_btn)

            self.wait.until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.openmask"))
            )
            print("[日志] 首页弹窗已成功关闭。")

        except Exception:
            print("[日志] 未检测到首页弹窗或无需关闭。")

    def open_internet_it_menu(self):
        """
        悬停并展开“互联网IT”一级菜单。
        
        Returns:
            WebElement: 包含“互联网IT”子菜单的父级元素对象。
        """
        try:
            internet_it = self.wait.until(
                EC.presence_of_element_located((
                    By.XPATH,
                    "//div[contains(@class,'type-item')]//a[contains(@class,'type-item-first') and normalize-space()='互联网IT']"
                ))
            )

            self.driver.execute_script(
                "arguments[0].scrollIntoView({block:'center', inline:'center'});",
                internet_it
            )
            time.sleep(0.5)

            self.actions.move_to_element(internet_it).pause(1).perform()
            print("[日志] 鼠标已悬停至“互联网IT”菜单。")

            parent = self.wait.until(
                EC.presence_of_element_located((
                    By.XPATH,
                    "//div[contains(@class,'type-item')][.//a[contains(@class,'type-item-first') and normalize-space()='互联网IT']]"
                ))
            )

            type_list = parent.find_element(By.CSS_SELECTOR, "div.type-list")

            # 等待菜单展开动画完成
            self.wait.until(
                lambda d: type_list.is_displayed() or "display: none" not in (type_list.get_attribute("style") or "")
            )
            print("[日志] “互联网IT”子菜单已完全展开。")
            return parent

        except Exception as e:
            raise RuntimeError(f"[错误] 展开“互联网IT”菜单失败：{e}")

    def build_search_url(self, keyword_text: str) -> str:
        """
        根据岗位关键词构造标准化的实习搜索页 URL。
        
        Args:
            keyword_text (str): 岗位关键词（如 "Java"、"Python" 等）。
            
        Returns:
            str: 构造好的搜索 URL。
        """
        encoded_keyword = quote(keyword_text)
        return (
            "https://www.shixiseng.com/interns"
            f"?keyword={encoded_keyword}&city=%E5%85%A8%E5%9B%BD&type=intern&from=menu"
        )

    def collect_grouped_links(self, parent_elem) -> list:
        """
        从展开的菜单中提取所有的岗位分组及对应的岗位名称。
        在采集阶段完成基于 item_url 的规范化去重处理。
        这样做是为了避免菜单结构重复、别名岗位或页面结构异常导致的重复链接入库问题。
        
        Args:
            parent_elem (WebElement): 包含子菜单的父级元素对象。
            
        Returns:
            list: 包含提取数据的字典列表。
        """
        results = []
        raw_candidate_count = 0
        duplicate_url_count = 0
        # 基于 item_url 在采集阶段完成首次出现保留的去重，避免重复链接继续进入结果列表或数据库。
        # 这样可防止菜单结构重复、别名岗位或页面结构异常导致的重复链接入库问题。
        seen_item_urls = set()

        try:
            type_list = parent_elem.find_element(By.CSS_SELECTOR, "div.type-list")
            groups = type_list.find_elements(By.XPATH, "./div")

            for group_idx, group in enumerate(groups, start=1):
                try:
                    group_title_elem = group.find_element(By.CSS_SELECTOR, "a.first-text.text")
                    group_title = group_title_elem.text.strip()
                except Exception:
                    group_title = f"未命名分组_{group_idx}"

                all_links = group.find_elements(By.CSS_SELECTOR, "a.text")

                for link in all_links:
                    classes = (link.get_attribute("class") or "").strip()
                    text = (link.text or "").strip()

                    # 排除空文本和一级标题本身
                    if not text or "first-text" in classes:
                        continue

                    # 构造标准化搜索 URL，并在采集阶段按 item_url 执行去重。
                    item_url = self.build_search_url(text)
                    raw_candidate_count += 1

                    if item_url not in seen_item_urls:
                        seen_item_urls.add(item_url)
                        results.append({
                            "group_name": group_title,
                            "item_text": text,
                            "item_url": item_url
                        })
                    else:
                        duplicate_url_count += 1

            print(
                "[日志] 菜单链接采集完成："
                f"分组数={len(groups)}，"
                f"去重前数量={raw_candidate_count}，"
                f"去重后数量={len(results)}，"
                f"去重掉数量={duplicate_url_count}。"
            )
            return results

        except Exception as e:
            print(f"[错误] 提取分组链接过程中发生异常：{e}")
            return results

    def print_results(self, rows: list):
        """
        在控制台结构化打印采集到的菜单数据，供人工核验。
        
        Args:
            rows (list): 提取的数据字典列表。
        """
        print("\n" + "=" * 20 + " 采集结果核验 " + "=" * 20)
        current_group = None

        for row in rows:
            if row["group_name"] != current_group:
                current_group = row["group_name"]
                print(f"\n【分组：{current_group}】")

            print(f"  - {row['item_text']} -> {row['item_url']}")
        print("=" * 54 + "\n")

    def close(self):
        """
        安全退出 Selenium 浏览器实例。
        """
        try:
            if self.driver:
                self.driver.quit()
                print("[日志] 浏览器进程已安全关闭。")
        except Exception:
            pass


def main():
    """
    程序主入口函数。
    负责初始化组件、串联核心业务逻辑，并进行全局异常捕获。
    """
    spider = None
    db = None

    try:
        print("[日志] 正在解析 ChromeDriver 路径...")
        driver_path = resolve_chromedriver_path_fast()
        print(f"[日志] 成功获取 ChromeDriver 路径：{driver_path}")

        spider = ShiXiSengCategoryUrlBuilder(
            driver_path=driver_path,
            headless=False,
            wait_seconds=15
        )

        db = MySQLManager(MYSQL_CONFIG)

        # 1. 数据库连接与建表
        db.connect()
        db.create_table()

        # 2. 网页自动化操作
        spider.open_homepage()
        spider.close_home_popup()
        parent_menu = spider.open_internet_it_menu()

        # 3. 提取与处理数据
        rows = spider.collect_grouped_links(parent_menu)

        # 4. 打印并入库
        spider.print_results(rows)
        db.upsert_rows(rows)

        print("[日志] 全部任务执行完成，准备退出。")

    except Exception as e:
        print(f"\n[致命错误] 程序运行中发生未捕获异常：{repr(e)}")

    finally:
        # 无论成功或失败，确保资源释放
        if spider:
            spider.close()
        if db:
            db.close()


if __name__ == "__main__":
    main()
