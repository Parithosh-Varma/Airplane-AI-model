import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from src.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class AirplanePicturesScraper(BaseScraper):
    @property
    def source_name(self) -> str:
        return "airplane-pictures"

    @property
    def base_url(self) -> str:
        return "https://airplane-pictures.net"

    @property
    def latest_feed_path(self) -> str:
        return "/"

    def parse_listing(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        listings = []
        for link_tag in soup.select("a[href*='/photo/']"):
            href = link_tag.get("href", "")
            img_tag = link_tag.select_one("img")
            if not img_tag:
                continue
            thumb = img_tag.get("src") or img_tag.get("data-src", "")
            if thumb.startswith("//"):
                thumb = "https:" + thumb

            photo_id = None
            match = re.search(r"/photo/(\d+)", href)
            if match:
                photo_id = match.group(1)

            listings.append({
                "image_url": thumb,
                "detail_url": urljoin(self.base_url, href),
                "image_id": photo_id or thumb.split("/")[-1].split(".")[0],
            })
        return listings

    def parse_detail(self, html: str, listing_data: dict) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        result = dict(listing_data)

        h1 = soup.select_one("h1")
        result["aircraft_name"] = h1.get_text(strip=True) if h1 else ""

        full = soup.select_one("img.detail-image, img#detail_image, a img")
        if full:
            src = full.get("src", "")
            if src.startswith("//"):
                src = "https:" + src
            result["image_url"] = src

        desc = soup.select_one("meta[name='description'], meta[property='og:description']")
        if desc:
            result["description"] = desc.get("content", "")

        return result
