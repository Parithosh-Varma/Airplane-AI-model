import asyncio
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from config import SCRAPER_CONFIG, IMAGES_DIR
from src.database.manager import DatabaseManager
from src.scraper.proxy_rotator import ProxyRotator

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    def __init__(
        self,
        db_manager: DatabaseManager,
        proxy_rotator: Optional[ProxyRotator] = None,
    ):
        self.db = db_manager
        self.proxy_rotator = proxy_rotator
        self.session: Optional[aiohttp.ClientSession] = None
        self._retry_count: dict[str, int] = {}

    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        pass

    @property
    @abstractmethod
    def latest_feed_path(self) -> str:
        pass

    @abstractmethod
    def parse_listing(self, html: str) -> list[dict]:
        pass

    @abstractmethod
    def parse_detail(self, html: str, listing_data: dict) -> dict:
        pass

    async def __aenter__(self):
        headers = {"User-Agent": random.choice(SCRAPER_CONFIG["user_agents"])}
        connector = aiohttp.TCPConnector(ssl=False, limit=5)
        self.session = aiohttp.ClientSession(headers=headers, connector=connector)
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def _polite_delay(self):
        delay = random.uniform(
            SCRAPER_CONFIG["politeness_delay_min"],
            SCRAPER_CONFIG["politeness_delay_max"],
        )
        logger.debug("Waiting %.1f seconds before next request...", delay)
        await asyncio.sleep(delay)

    def _get_headers(self) -> dict:
        return {
            "User-Agent": random.choice(SCRAPER_CONFIG["user_agents"]),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    async def _fetch(self, url: str, retries: int = None) -> Optional[str]:
        if retries is None:
            retries = SCRAPER_CONFIG["max_retries"]

        proxy = self.proxy_rotator.get_proxy() if self.proxy_rotator else None

        for attempt in range(1, retries + 1):
            try:
                async with self.session.get(
                    url,
                    headers=self._get_headers(),
                    proxy=proxy.get("http") if proxy else None,
                    timeout=aiohttp.ClientTimeout(total=SCRAPER_CONFIG["request_timeout"]),
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        self._retry_count[url] = 0
                        return text
                    elif resp.status == 429:
                        wait = (SCRAPER_CONFIG["backoff_base"] ** attempt) * 60
                        logger.warning(
                            "Rate limited (429) on %s. Waiting %ds (attempt %d/%d)",
                            url, wait, attempt, retries,
                        )
                        await asyncio.sleep(wait)
                        if self.proxy_rotator:
                            self.proxy_rotator.rotate()
                    elif resp.status == 403:
                        logger.warning("Forbidden (403) on %s. Rotating proxy.", url)
                        if self.proxy_rotator:
                            self.proxy_rotator.rotate()
                        await asyncio.sleep(random.uniform(10, 30))
                    else:
                        logger.warning(
                            "HTTP %d on %s (attempt %d/%d)",
                            resp.status, url, attempt, retries,
                        )
                        await asyncio.sleep(5)
            except asyncio.TimeoutError:
                logger.warning("Timeout on %s (attempt %d/%d)", url, attempt, retries)
                await asyncio.sleep(SCRAPER_CONFIG["backoff_base"] ** attempt * 10)
            except aiohttp.ClientError as e:
                logger.warning("Client error on %s: %s (attempt %d/%d)", url, e, attempt, retries)
                await asyncio.sleep(SCRAPER_CONFIG["backoff_base"] ** attempt * 5)

        logger.error("Failed to fetch %s after %d attempts", url, retries)
        return None

    async def fetch_latest_uploads(self) -> list[dict]:
        feed_url = f"{self.base_url}{self.latest_feed_path}"
        logger.info("Fetching latest uploads from %s", feed_url)
        html = await self._fetch(feed_url)
        if not html:
            return []
        listings = self.parse_listing(html)
        logger.info("Found %d listings on %s", len(listings), self.source_name)
        return listings

    async def fetch_detail(self, listing: dict) -> Optional[dict]:
        detail_url = listing.get("detail_url", listing.get("image_url"))
        if not detail_url:
            return None
        html = await self._fetch(detail_url)
        if not html:
            return None
        return self.parse_detail(html, listing)

    async def download_image(self, image_url: str, filepath: Path) -> bool:
        try:
            async with self.session.get(
                image_url,
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    with open(filepath, "wb") as f:
                        f.write(await resp.read())
                    return True
                else:
                    logger.warning("Failed to download %s: HTTP %d", image_url, resp.status)
                    return False
        except Exception as e:
            logger.error("Error downloading %s: %s", image_url, e)
            return False

    def build_filepath(self, listing: dict) -> Path:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        filename = f"{listing.get('image_id', 'unknown')}.jpg"
        return IMAGES_DIR / self.source_name / today / filename

    async def scrape_loop(self):
        while True:
            try:
                listings = await self.fetch_latest_uploads()
                for listing in listings:
                    if self.db.image_exists(listing.get("image_url", "")):
                        logger.debug("Skipping duplicate: %s", listing.get("image_url"))
                        continue

                    detail = await self.fetch_detail(listing)
                    if not detail:
                        continue

                    filepath = self.build_filepath(detail)
                    success = await self.download_image(detail["image_url"], filepath)
                    if not success:
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
                        logger.info("Saved image %d: %s", img_id, detail.get("aircraft_name", "unknown"))

                    await self._polite_delay()

                logger.info(
                    "Completed scrape cycle for %s. Sleeping before next cycle.",
                    self.source_name,
                )
                await asyncio.sleep(SCRAPER_CONFIG["politeness_delay_max"] * 3)
            except Exception as e:
                logger.exception("Unhandled error in %s scrape loop: %s", self.source_name, e)
                await asyncio.sleep(60)
