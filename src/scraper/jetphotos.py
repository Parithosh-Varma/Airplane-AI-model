import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from src.scraper.base import BaseScraper

logger = logging.getLogger(__name__)


class JetPhotosScraper(BaseScraper):
    @property
    def source_name(self) -> str:
        return "jetphotos"

    @property
    def base_url(self) -> str:
        return "https://www.jetphotos.com"

    @property
    def latest_feed_path(self) -> str:
        return "/latest?page=1"

    def parse_listing(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        listings = []
        for img_tag in soup.select("img.lazy, img[src*='thumbs']"):
            parent_link = img_tag.find_parent("a")
            if not parent_link:
                continue
            href = parent_link.get("href", "")
            if not href or "/photo/" not in href:
                continue

            image_url = img_tag.get("data-src") or img_tag.get("src", "")
            if not image_url:
                continue
            if image_url.startswith("//"):
                image_url = "https:" + image_url

            photo_id = None
            match = re.search(r"/photo/(\d+)", href)
            if match:
                photo_id = match.group(1)

            listings.append({
                "image_url": image_url,
                "detail_url": urljoin(self.base_url, href),
                "image_id": photo_id or image_url.split("/")[-1].split(".")[0],
            })
        return listings

    def parse_detail(self, html: str, listing_data: dict) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        result = dict(listing_data)

        title_el = soup.select_one("h1")
        result["aircraft_name"] = title_el.get_text(strip=True) if title_el else ""

        desc_el = soup.select_one("meta[name='description']")
        if desc_el:
            result["description"] = desc_el.get("content", "")

        full_img = soup.select_one("img[src*='photos']")
        if full_img:
            src = full_img.get("src", "")
            if src.startswith("//"):
                src = "https:" + src
            result["image_url"] = src

        return result
