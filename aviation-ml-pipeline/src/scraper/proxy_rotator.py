import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProxyRotator:
    def __init__(self, rotation_url: str = "", api_key: str = ""):
        self.rotation_url = rotation_url
        self.api_key = api_key
        self._static_proxies: list[str] = []
        self._current_index = 0

    def set_static_proxies(self, proxies: list[str]):
        self._static_proxies = proxies
        random.shuffle(self._static_proxies)
        logger.info("Loaded %d static proxies", len(proxies))

    def get_proxy(self) -> Optional[dict]:
        if self.rotation_url:
            return self._residential_proxy()
        if self._static_proxies:
            return self._static_proxy()
        return None

    def _residential_proxy(self) -> dict:
        return {
            "http": self.rotation_url,
            "https": self.rotation_url,
        }

    def _static_proxy(self) -> dict:
        proxy = self._static_proxies[self._current_index % len(self._static_proxies)]
        self._current_index += 1
        return {
            "http": proxy,
            "https": proxy,
        }

    def rotate(self):
        if self._static_proxies:
            self._current_index = (self._current_index + 1) % len(self._static_proxies)
            logger.debug("Rotated to proxy index %d", self._current_index)
