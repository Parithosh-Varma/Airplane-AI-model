import asyncio
import logging
import re

from playwright.async_api import Page

from src.scraper.playwright_base import PlaywrightBaseScraper

logger = logging.getLogger(__name__)


class PlaywrightJetPhotosScraper(PlaywrightBaseScraper):
    @property
    def source_name(self) -> str:
        return "jetphotos"

    @property
    def start_url(self) -> str:
        return "https://www.jetphotos.com/latest?page=1"

    async def extract_listings(self, page: Page) -> list[dict]:
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)
        links = await page.query_selector_all("a[href*='/photo/']")
        listings = []
        seen = set()
        for link in links:
            href = await link.get_attribute("href")
            if not href or href in seen:
                continue
            seen.add(href)
            full_url = "https://www.jetphotos.com" + href if href.startswith("/") else href
            img = await link.query_selector("img")
            img_url = ""
            if img:
                img_url = await img.get_attribute("data-src") or await img.get_attribute("src") or ""
            img_url = self._resolve_url(img_url)
            photo_id = None
            m = re.search(r"/photo/(\d+)", href)
            if m:
                photo_id = m.group(1)
            listings.append({
                "image_url": img_url,
                "detail_url": full_url,
                "image_id": photo_id or str(len(listings)),
            })
        return listings

    async def extract_detail(self, page: Page, listing: dict) -> dict:
        result = dict(listing)
        try:
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)
            h1 = await page.query_selector("h1")
            if h1:
                result["aircraft_name"] = (await h1.text_content() or "").strip()
            full = await page.query_selector("img[src*='photos']")
            if full:
                src = await full.get_attribute("src") or ""
                result["image_url"] = self._resolve_url(src)
            desc = await page.query_selector("meta[name='description']")
            if desc:
                result["description"] = await desc.get_attribute("content") or ""
        except Exception as e:
            logger.debug("[jetphotos] Detail parse: %s", e)
        return result
