import asyncio
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from config import SCRAPER_CONFIG, IMAGES_DIR
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class PlaywrightBaseScraper(ABC):
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close_browser()

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abstractmethod
    def start_url(self) -> str:
        pass

    @abstractmethod
    async def extract_listings(self, page: Page) -> list[dict]:
        pass

    @abstractmethod
    async def extract_detail(self, page: Page, listing: dict) -> dict:
        pass

    async def _random_delay(self, min_s: float = None, max_s: float = None):
        delay = random.uniform(
            min_s or SCRAPER_CONFIG["politeness_delay_min"],
            max_s or SCRAPER_CONFIG["politeness_delay_max"],
        )
        logger.debug("[%s] Waiting %.1fs", self.source_name, delay)
        await asyncio.sleep(delay)

    async def _stealth_page(self, context: BrowserContext) -> Page:
        page = await context.new_page()
        await page.set_viewport_size({
            "width": random.randint(1200, 1600),
            "height": random.randint(800, 1000),
        })
        ua = random.choice(SCRAPER_CONFIG["user_agents"])
        await context.add_init_script(f"""
            Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined }});
            Object.defineProperty(navigator, 'plugins', {{ get: () => [1,2,3,4,5] }});
            Object.defineProperty(navigator, 'languages', {{ get: () => ['en-US', 'en'] }});
        """)
        return page

    async def _safe_goto(self, page: Page, url: str, retries: int = 3) -> bool:
        for attempt in range(1, retries + 1):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(random.uniform(2, 5))
                await page.wait_for_load_state("networkidle", timeout=30000)
                return True
            except Exception as e:
                logger.warning(
                    "[%s] goto %s failed (attempt %d/%d): %s",
                    self.source_name, url, attempt, retries, e,
                )
                if attempt < retries:
                    wait = (SCRAPER_CONFIG["backoff_base"] ** attempt) * 10
                    await asyncio.sleep(wait)
        return False

    async def _close_popups(self, page: Page):
        try:
            buttons = await page.query_selector_all(
                "button:has-text('Accept'), button:has-text('Agree'), "
                "button:has-text('Got it'), .cookie-consent button, "
                "[aria-label*='close'], .modal-close, .popup-close"
            )
            for btn in buttons:
                try:
                    await btn.click(timeout=3000)
                    await asyncio.sleep(0.5)
                except Exception:
                    pass
        except Exception:
            pass

    async def start_browser(self):
        p = await async_playwright().start()
        launched = False
        for _ in range(3):
            try:
                self._browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--single-process",
                    ],
                )
                launched = True
                break
            except Exception as e:
                logger.warning("[%s] Browser launch failed: %s, retrying...", self.source_name, e)
                await asyncio.sleep(5)
        if not launched:
            raise RuntimeError(f"[{self.source_name}] Could not launch Chromium browser")
        self._context = await self._browser.new_context(
            user_agent=random.choice(SCRAPER_CONFIG["user_agents"]),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
            permissions=["geolocation"],
        )
        logger.info("[%s] Browser started", self.source_name)

    async def close_browser(self):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        logger.info("[%s] Browser closed", self.source_name)

    def _resolve_url(self, url: str) -> str:
        if not url:
            return ""
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/"):
            from urllib.parse import urlsplit
            parts = urlsplit(self.start_url)
            return f"{parts.scheme}://{parts.netloc}{url}"
        if url.startswith("http://") or url.startswith("https://"):
            return url
        from urllib.parse import urljoin
        return urljoin(self.start_url, url)

    async def download_image(self, page: Page, image_url: str, filepath: Path) -> bool:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        abs_url = self._resolve_url(image_url)
        if not abs_url:
            return False
        try:
            resp = await page.context.request.get(abs_url, headers={
                "Referer": self.start_url,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })
            if resp.ok:
                data = await resp.body()
                with open(filepath, "wb") as f:
                    f.write(data)
                return True
            else:
                logger.warning(
                    "[%s] Download HTTP %d for %s", self.source_name, resp.status, abs_url,
                )
                return False
        except Exception as e:
            logger.error("[%s] Download error %s: %s", self.source_name, abs_url, e)
            return False

    def build_filepath(self, image_id: str) -> Path:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return IMAGES_DIR / self.source_name / today / f"{image_id}.jpg"

    async def scrape_loop(self):
        await self.start_browser()
        try:
            while True:
                try:
                    page = await self._stealth_page(self._context)
                    ok = await self._safe_goto(page, self.start_url)
                    if not ok:
                        await page.close()
                        await asyncio.sleep(60)
                        continue

                    await self._close_popups(page)
                    await asyncio.sleep(random.uniform(1, 3))
                    listings = await self.extract_listings(page)
                    await page.close()
                    logger.info(
                        "[%s] Found %d listings", self.source_name, len(listings),
                    )

                    for listing in listings:
                        img_url = listing.get("image_url", "")
                        if not img_url or self.db.image_exists(img_url):
                            continue

                        detail_page = await self._stealth_page(self._context)
                        detail_url = listing.get("detail_url", img_url)
                        ok = await self._safe_goto(detail_page, detail_url)
                        if ok:
                            await self._close_popups(detail_page)
                            detail = await self.extract_detail(detail_page, listing)
                        else:
                            detail = listing
                        await detail_page.close()

                        filepath = self.build_filepath(listing.get("image_id", str(hash(img_url))))
                        dl_page = await self._stealth_page(self._context)
                        dled = await self.download_image(dl_page, detail["image_url"], filepath)
                        await dl_page.close()
                        if not dled:
                            continue

                        record = {
                            "source_url": detail.get("detail_url", detail["image_url"]),
                            "image_url": detail["image_url"],
                            "local_filepath": str(filepath),
                            "aircraft_name": detail.get("aircraft_name", ""),
                            "description": detail.get("description", ""),
                            "source_site": self.source_name,
                        }
                        img_id = self.db.insert_image(record)
                        if img_id:
                            logger.info(
                                "[%s] Saved #%d: %s", self.source_name, img_id,
                                detail.get("aircraft_name", "unknown"),
                            )

                        await self._random_delay()

                    logger.info(
                        "[%s] Cycle complete. Sleeping before next round.",
                        self.source_name,
                    )
                    await asyncio.sleep(SCRAPER_CONFIG["politeness_delay_max"] * 2)
                except Exception as e:
                    logger.exception("[%s] Scrape cycle error: %s", self.source_name, e)
                    await asyncio.sleep(60)
        finally:
            await self.close_browser()
