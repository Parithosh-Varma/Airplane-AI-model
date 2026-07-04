import asyncio
import logging
import signal
import sys
from datetime import datetime
from contextlib import suppress

from aiohttp import web

from config import DATABASE_URL, LOGGING_CONFIG, SCRAPER_CONFIG, RAILWAY_PORT
from src.database.manager import DatabaseManager
from src.preprocessing.image_processor import ImagePreprocessor
from src.scraper.airplanepictures import AirplanePicturesScraper
from src.scraper.jetphotos import JetPhotosScraper
from src.scraper.planespotters import PlanespottersScraper
from src.scraper.proxy_rotator import ProxyRotator
from src.training.trainer import ModelTrainer

from pathlib import Path

_log_handlers = [logging.StreamHandler(sys.stdout)]
_log_file = Path("logs/pipeline.log")
try:
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _log_handlers.append(logging.FileHandler(str(_log_file)))
except OSError:
    pass

logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self.db = DatabaseManager(DATABASE_URL, echo=False)
        self.proxy_rotator = self._init_proxy_rotator()
        self._shutdown = asyncio.Event()

    def _init_proxy_rotator(self) -> ProxyRotator:
        rotator = ProxyRotator(
            rotation_url=SCRAPER_CONFIG["proxy_rotation_url"],
            api_key=SCRAPER_CONFIG["proxy_api_key"],
        )
        if rotator.rotation_url:
            logger.info("Residential proxy rotation configured")
        else:
            logger.info(
                "No proxy rotation configured. "
                "For 24/7 production use, set PROXY_ROTATION_URL and PROXY_API_KEY "
                "in .env to avoid IP bans. See https://proxyrotate.com or similar services."
            )
        return rotator

    def _handle_signal(self):
        logger.info("Shutdown signal received. Stopping pipeline gracefully...")
        self._shutdown.set()

    async def _healthcheck(self):
        if not RAILWAY_PORT:
            return
        app = web.Application()
        app.router.add_get("/health", lambda r: web.json_response({"status": "ok"}))

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", RAILWAY_PORT)
        await site.start()
        logger.info("Healthcheck server listening on 0.0.0.0:%d", RAILWAY_PORT)

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal)

        logger.info("=" * 60)
        logger.info("Aviation ML Pipeline starting at %s", datetime.utcnow())
        logger.info("=" * 60)

        scrapers = [
            JetPhotosScraper(self.db, self.proxy_rotator),
            AirplanePicturesScraper(self.db, self.proxy_rotator),
            PlanespottersScraper(self.db, self.proxy_rotator),
        ]

        preprocessor = ImagePreprocessor(self.db)
        trainer = ModelTrainer(self.db)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._watch_shutdown(), name="shutdown_watcher")
            tg.create_task(self._healthcheck(), name="healthcheck")

            for scraper in scrapers:
                tg.create_task(self._run_scraper(scraper), name=f"scraper_{scraper.source_name}")

            tg.create_task(preprocessor.preprocessing_loop(), name="preprocessor")
            tg.create_task(trainer.training_loop(), name="trainer")

            tg.create_task(self._stats_logger(), name="stats_logger")

    async def _watch_shutdown(self):
        await self._shutdown.wait()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        logger.info("All tasks cancelled. Goodbye.")

    async def _run_scraper(self, scraper):
        async with scraper as s:
            scrape_task = asyncio.create_task(s.scrape_loop())
            cancel_handle = asyncio.ensure_future(self._shutdown.wait())
            done, _ = await asyncio.wait(
                [scrape_task, cancel_handle],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_handle in done:
                scrape_task.cancel()

    async def _stats_logger(self):
        while not self._shutdown.is_set():
            try:
                stats = self.db.get_stats()
                logger.info("STATS: %s", stats)
            except Exception as e:
                logger.debug("Stats query failed: %s", e)
            await asyncio.sleep(600)


async def main():
    pipeline = Pipeline()
    try:
        await pipeline.run()
    except asyncio.CancelledError:
        logger.info("Pipeline cancelled.")
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    asyncio.run(main())
