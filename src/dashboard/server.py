import asyncio
import logging
from pathlib import Path

from aiohttp import web

from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aviation ML Pipeline</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✈</text></svg>">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,sans-serif; background: #0f172a; color: #e2e8f0; padding: 1.5rem; min-height: 100vh; }
  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: .25rem; display: flex; align-items: center; gap: .5rem; }
  .subtitle { color: #94a3b8; font-size: .9rem; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .card { background: #1e293b; border-radius: 12px; padding: 1.25rem; border: 1px solid #334155; }
  .card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; color: #64748b; margin-bottom: .4rem; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .value.green { color: #22c55e; }
  .card .value.blue { color: #3b82f6; }
  .card .value.yellow { color: #eab308; }
  .card .value.purple { color: #a855f7; }
  .card .value.red { color: #ef4444; }
  .source-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .source-card { background: #1e293b; border-radius: 12px; padding: 1.25rem; border: 1px solid #334155; }
  .source-card .name { font-size: 1.1rem; font-weight: 600; margin-bottom: .75rem; text-transform: capitalize; }
  .source-card .stat-row { display: flex; justify-content: space-between; padding: .3rem 0; font-size: .9rem; color: #cbd5e1; }
  .source-card .stat-row .num { font-weight: 600; }
  .section-title { font-size: 1rem; font-weight: 600; margin-bottom: .75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; }
  th { text-align: left; padding: .6rem .5rem; color: #64748b; font-weight: 500; border-bottom: 1px solid #334155; text-transform: uppercase; font-size: .7rem; letter-spacing: .05em; }
  td { padding: .6rem .5rem; border-bottom: 1px solid #1e293b; }
  .badge { display: inline-block; padding: .15rem .5rem; border-radius: 999px; font-size: .7rem; font-weight: 600; }
  .badge.yes { background: #166534; color: #22c55e; }
  .badge.no { background: #450a0a; color: #ef4444; }
  .badge.wait { background: #713f12; color: #eab308; }
  .time { color: #64748b; font-size: .8rem; }
  .footer { text-align: center; color: #475569; font-size: .8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #1e293b; }
  .spinner { display: inline-block; width: 1rem; height: 1rem; border: 2px solid #334155; border-top-color: #3b82f6; border-radius: 50%; animation: spin .6s linear infinite; margin-right: .5rem; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading { text-align: center; padding: 3rem; color: #64748b; }
  .error-state { text-align: center; padding: 2rem; color: #ef4444; }
</style>
</head>
<body>
  <h1>✈ Aviation ML Pipeline</h1>
  <div class="subtitle" id="subtitle">Loading...</div>
  <div class="grid" id="stats-grid"></div>
  <div class="section-title">Per Source</div>
  <div class="source-grid" id="source-grid"></div>
  <div class="section-title">Recent Images</div>
  <div style="background:#1e293b;border-radius:12px;border:1px solid #334155;overflow-x:auto">
    <table><thead><tr>
      <th>ID</th><th>Aircraft</th><th>Source</th><th>Preprocessed</th><th>Trained</th><th>Added</th>
    </tr></thead><tbody id="recent-tbody"></tbody></table>
  </div>
  <div class="footer" id="footer"></div>
<script>
async function load(){try{
  const r=await fetch('/api/stats');if(!r.ok)throw new Error('HTTP '+r.status);
  const d=await r.json();
  document.getElementById('subtitle').textContent='Last updated: '+new Date(d.now).toLocaleString()+'  ·  Uptime: '+d.uptime;
  const sg=document.getElementById('stats-grid');
  const cards=[
    {label:'Total Images',value:d.stats.total_images,cls:'blue'},
    {label:'Preprocessed',value:d.stats.preprocessed,cls:'green'},
    {label:'Trained',value:d.stats.trained,cls:'purple'},
    {label:'Awaiting Training',value:d.stats.untrained,cls:d.stats.untrained>0?'yellow':'green'},
  ];
  sg.innerHTML=cards.map(c=>'<div class=card><div class=label>'+c.label+'</div><div class="value '+c.cls+'">'+c.value+'</div></div>').join('');

  const srcg=document.getElementById('source-grid');
  const perSource=d.per_source&&d.per_source.length?d.per_source:[{source:'jetphotos',total:0,trained:0},{source:'airplane-pictures',total:0,trained:0},{source:'planespotters',total:0,trained:0}];
  srcg.innerHTML=perSource.map(s=>'<div class=source-card><div class=name>'+s.source+'</div><div class=stat-row><span>Images</span><span class=num>'+s.total+'</span></div><div class=stat-row><span>Trained</span><span class=num>'+s.trained+'</span></div></div>').join('');

  const tb=document.getElementById('recent-tbody');
  const rows=d.recent&&d.recent.length?d.recent.map(r=>'<tr><td>#'+r.id+'</td><td>'+(r.aircraft_name||'—')+'</td><td>'+r.source_site+'</td><td><span class="badge '+(r.is_preprocessed?'yes':'no')+'">'+(r.is_preprocessed?'✓':'✗')+'</span></td><td><span class="badge '+(r.is_trained?'yes':r.is_preprocessed?'wait':'no')+'">'+(r.is_trained?'✓':r.is_preprocessed?'pending':'—')+'</span></td><td class=time>'+(r.created_at?new Date(r.created_at).toLocaleString():'')+'</td></tr>').join(''):
    '<tr><td colspan=6 style="text-align:center;color:#64748b;padding:2rem">No images yet</td></tr>';
  tb.innerHTML=rows;
  document.getElementById('footer').textContent='Pipeline  ·  Aviation ML';
}catch(e){document.getElementById('stats-grid').innerHTML='<div class=error-state>Failed to load: '+e.message+'</div>';}}
load();setInterval(load,10000);
</script>
</body>
</html>"""


class DashboardServer:
    def __init__(self, db_manager: DatabaseManager, port: int, uptime_start: float):
        self.db = db_manager
        self.port = port
        self.uptime_start = uptime_start
        self._runner: web.AppRunner = None

    def _uptime(self) -> str:
        import time
        s = int(time.time() - self.uptime_start)
        h, r = divmod(s, 3600)
        m, s = divmod(r, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    async def _handle_index(self, _request):
        return web.Response(text=PAGE, content_type="text/html")

    async def _handle_health(self, _request):
        return web.json_response({"status": "ok"})

    async def _handle_stats(self, _request):
        import time
        try:
            stats = self.db.get_stats()
            per_source = self.db.get_per_source_stats()
            recent = self.db.get_recent_images(limit=30)
            return web.json_response({
                "stats": stats,
                "per_source": per_source,
                "recent": recent,
                "uptime": self._uptime(),
                "now": time.time(),
            })
        except Exception as e:
            logger.exception("Stats API error")
            return web.json_response({"error": str(e)}, status=500)

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/api/stats", self._handle_stats)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        logger.info("Dashboard live at http://0.0.0.0:%d", self.port)

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
