import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from src.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class PlanespottersScraper(BaseScraper):
    @property
    def source_name(self) -> str:
        return "planespotters"

    @property
    def base_url(self) -> str:
        return "https://www.planespotters.net"

    @property
    def latest_feed_path(self) -> str:
        return "/photos/recent"

    def parse_listing(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        listings = []
        for img_tag in soup.select("img.photo-thumb, img[src*='thumbs']"):
            parent = img_tag.find_parent("a")
            if not parent:
                continue
            href = parent.get("href", "")
            if "/photo/" not in href:
                continue

            img_url = img_tag.get("data-src") or img_tag.get("src", "")
            if img_url.startswith("//"):
                img_url = "https:" + img_url

            photo_id = None
            match = re.search(r"/(\d+)", href)
            if match:
                photo_id = match.group(1)

            listings.append({
                "image_url": img_url,
                "detail_url": urljoin(self.base_url, href),
                "image_id": photo_id or img_url.split("/")[-1].split(".")[0],
            })
        return listings

    def parse_detail(self, html: str, listing_data: dict) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        result = dict(listing_data)

        title_el = soup.select_one("h1")
        result["aircraft_name"] = title_el.get_text(strip=True) if title_el else ""

        full_img = soup.select_one("img[src*='photos'], img[src*='planespotters']")
        if full_img:
            src = full_img.get("src", "")
            if src.startswith("//"):
                src = "https:" + src
            result["image_url"] = src

        desc_el = soup.select_one("meta[name='description'], meta[property='og:description']")
        if desc_el:
            result["description"] = desc_el.get("content", "")

        return result
