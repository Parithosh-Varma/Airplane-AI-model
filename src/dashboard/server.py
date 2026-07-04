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
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,sans-serif;background:#0f172a;color:#e2e8f0;padding:1.5rem;min-height:100vh}
  h1{font-size:1.6rem;font-weight:700;margin-bottom:.25rem}
  .sub{color:#94a3b8;font-size:.9rem;margin-bottom:1.5rem}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1.5rem}
  .card{background:#1e293b;border-radius:12px;padding:1.25rem;border:1px solid #334155}
  .card .l{font-size:.75rem;text-transform:uppercase;color:#64748b;margin-bottom:.4rem}
  .card .v{font-size:2rem;font-weight:700}
  .card .g{color:#22c55e}.card .b{color:#3b82f6}.card .y{color:#eab308}.card .p{color:#a855f7}.card .r{color:#ef4444}
  .sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-bottom:1.5rem}
  .scard{background:#1e293b;border-radius:12px;padding:1.25rem;border:1px solid #334155}
  .scard .n{font-size:1.1rem;font-weight:600;margin-bottom:.75rem;text-transform:capitalize}
  .scard .sr{display:flex;justify-content:space-between;padding:.3rem 0;font-size:.9rem;color:#cbd5e1}
  .scard .sr .nm{font-weight:600}
  .st{font-size:1rem;font-weight:600;margin-bottom:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th{text-align:left;padding:.6rem .5rem;color:#64748b;font-weight:500;border-bottom:1px solid #334155;text-transform:uppercase;font-size:.7rem;letter-spacing:.05em}
  td{padding:.6rem .5rem;border-bottom:1px solid #1e293b}
  .badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;font-size:.7rem;font-weight:600}
  .badge.y{background:#166534;color:#22c55e}.badge.n{background:#450a0a;color:#ef4444}.badge.w{background:#713f12;color:#eab308}
  .tm{color:#64748b;font-size:.8rem}
  .ft{text-align:center;color:#475569;font-size:.8rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b}
  .loading{text-align:center;padding:3rem;color:#64748b}
  .err{text-align:center;padding:2rem;color:#ef4444}
  .tblwrap{background:#1e293b;border-radius:12px;border:1px solid #334155;overflow-x:auto}
</style>
</head>
<body>
<h1>&#9992; Aviation ML Pipeline</h1>
<div class="sub" id="sub">Pipeline running &mdash; loading data...</div>
<div class="grid" id="stats"></div>
<div class="st">Per Source</div>
<div class="sgrid" id="sources"></div>
<div class="st">Recent Images</div>
<div class="tblwrap"><table><thead><tr>
<th>ID</th><th>Aircraft</th><th>Source</th><th>Preprocessed</th><th>Trained</th><th>Added</th>
</tr></thead><tbody id="recent"></tbody></table></div>
<div class="ft" id="ft">Aviation ML Pipeline</div>
<script>
async function load(){
  try{
    const r=await fetch('/api/stats');
    if(!r.ok) throw new Error('HTTP '+r.status);
    const d=await r.json();
    const now=new Date(d.now);
    document.getElementById('sub').innerHTML='Last updated: '+now.toLocaleString()+' \u00b7 Uptime: '+d.uptime;
    document.getElementById('stats').innerHTML=[
      {l:'Total Images',v:d.stats.total_images,c:'b'},
      {l:'Preprocessed',v:d.stats.preprocessed,c:'g'},
      {l:'Trained',v:d.stats.trained,c:'p'},
      {l:'Awaiting Training',v:d.stats.untrained,c:d.stats.untrained>0?'y':'g'},
    ].map(function(c){return '<div class=card><div class=l>'+c.l+'</div><div class="v '+c.c+'">'+c.v+'</div></div>'}).join('');
    var ps=d.per_source&&d.per_source.length?d.per_source:[{source:'jetphotos',total:0,trained:0},{source:'airplane-pictures',total:0,trained:0},{source:'planespotters',total:0,trained:0}];
    document.getElementById('sources').innerHTML=ps.map(function(s){return '<div class=scard><div class=n>'+s.source+'</div><div class=sr><span>Images</span><span class=nm>'+s.total+'</span></div><div class=sr><span>Trained</span><span class=nm>'+s.trained+'</span></div></div>'}).join('');
    var tb=document.getElementById('recent');
    if(d.recent&&d.recent.length){
      tb.innerHTML=d.recent.map(function(r){return '<tr><td>#'+r.id+'</td><td>'+(r.aircraft_name||'\u2014')+'</td><td>'+r.source_site+'</td><td><span class="badge '+(r.is_preprocessed?'y':'n')+'">'+(r.is_preprocessed?'OK':'--')+'</span></td><td><span class="badge '+(r.is_trained?'y':r.is_preprocessed?'w':'n')+'">'+(r.is_trained?'OK':r.is_preprocessed?'wait':'--')+'</span></td><td class=tm>'+(r.created_at?new Date(r.created_at).toLocaleString():'')+'</td></tr>'}).join('');
    } else {
      tb.innerHTML='<tr><td colspan=6 style="text-align:center;color:#64748b;padding:2rem">No images collected yet. Scrapers are running, waiting 30-120s between requests.</td></tr>';
    }
    document.getElementById('ft').innerHTML='Aviation ML Pipeline \u00b7 Running';
  }catch(e){
    document.getElementById('stats').innerHTML='<div class=err>Could not load stats: '+e.message+'</div>';
    document.getElementById('sub').textContent='Pipeline is running but stats are not available yet';
  }
}
load();
setInterval(load,10000);
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
                "now": int(time.time() * 1000),
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
        actual_port = site._server.sockets[0].getsockname()[1]
        logger.info("Dashboard live at http://0.0.0.0:%d", actual_port)

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
