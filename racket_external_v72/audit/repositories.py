"""Record public metadata searches for reusable impact datasets; count only successful queries."""
import concurrent.futures
import datetime
import json
import urllib.parse
import subprocess
from pathlib import Path

def query(provider, keywords):
    if provider == "DataCite":
        params = {"query": keywords, "resource-type-id": "dataset", "page[size]": "250"}
        endpoint = "https://api.datacite.org/dois"
    elif provider == "Zenodo":
        params = {"q": keywords, "size": "20"}
        endpoint = "https://zenodo.org/api/records"
    else:
        params = {"q": keywords, "per_page": "50"}
        endpoint = "https://api.github.com/search/repositories"
    url = endpoint + "?" + urllib.parse.urlencode(params,quote_via=urllib.parse.quote)
    rec = {"provider": provider, "query": keywords, "url": url}
    try:
        raw = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                       "--max-time", "22", "--user-agent", "racket-impact-research", url], stderr=subprocess.STDOUT)
        body = json.loads(raw)
        if provider == "DataCite":
            count = body["meta"]["total"]
            items = [(r["attributes"]["titles"][0]["title"], r["attributes"].get("url") or r["id"]) for r in body["data"]]
        elif provider == "Zenodo":
            count = body["hits"]["total"]
            items = [(r["metadata"]["title"], r["links"].get("self_html") or r.get("doi_url") or str(r["id"])) for r in body["hits"]["hits"]]
        else:
            count = body["total_count"]
            items = [(r["full_name"],r["html_url"]) for r in body["items"]]
        rec.update(total=count, records=items)
    except Exception as error:
        rec["error"] = (error.output.decode(errors="replace")[:400] if isinstance(error, subprocess.CalledProcessError) else str(error))
    return rec

queries = [(p,q) for p in ("DataCite","Zenodo") for q in ('"table tennis" AND (racket OR rubber) AND (rebound OR spin)', '"long pips" OR "long pimples" OR "anti-spin rubber"')]
queries += [("DataCite", 'titles.title:"table tennis"'), ("Zenodo", '"table tennis" AND "spin"')]
queries += [("GitHub", q) for q in ('"table tennis" racket spin', '"table tennis" rubber impact', '"long pips" table tennis')]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    records = list(pool.map(lambda v:query(*v),queries))
output = {"checked_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"queries":records}
Path(__file__).with_name("repositories.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
for record in records:
    print(record["provider"], record.get("total"), record.get("error",""), str(record.get("records",[])[:1])[:160])
