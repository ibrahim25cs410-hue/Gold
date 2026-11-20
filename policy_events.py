"""Archive original Federal Reserve releases with explicit publication timestamps."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FED = "https://www.federalreserve.gov"
CALENDAR = FED + "/monetarypolicy/fomccalendars.htm"


def parse_statement(url, payload):
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", payload.decode("utf-8"))).split())
    date = re.search(r"monetary(\d{8})a\.htm$", url)
    heading = re.search(r"For release at (\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)", text, re.I)
    action = re.search(r"decided to (raise|lower|maintain|keep) the target range", text, re.I)
    if not (date and heading and action):
        raise ValueError("Unverified statement date, release time, or policy action")
    hour = int(heading[1]) % 12 + (12 if heading[3].lower()=="p.m." else 0)
    observed = pd.to_datetime(date[1],format="%Y%m%d")
    published = (observed+pd.Timedelta(hours=hour,minutes=int(heading[2]))).tz_localize(
        "America/New_York").tz_convert("UTC")
    decision = {"raise":1,"lower":-1,"maintain":0,"keep":0}[action[1].lower()]
    return {"observation_date":str(observed.date()),"published_at":published.isoformat(),
            "action":decision,"source":url,"sha256":hashlib.sha256(payload).hexdigest()}


def fetch(folder):
    folder.mkdir(parents=True,exist_ok=False)
    def get(url):
        request=Request(url,headers={"User-Agent":"Mozilla/5.0","Accept":"text/html"})
        with urlopen(request,timeout=20) as response:
            return response.read()
    calendar = get(CALENDAR)
    (folder/"calendar.html").write_bytes(calendar)
    links = sorted(set(re.findall(rb"/newsevents/pressreleases/monetary20\d{6}a\.htm",
                                  calendar)))
    if not links:
        raise ValueError("No statement links found")
    def download(link):
        url = FED+link.decode()
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            payload=get(url)
            (folder/url.rsplit("/",1)[1]).write_bytes(payload)
            return {**parse_statement(url,payload),"retrieved_at":stamp,"status":"verified"}
        except Exception as exc:
            return {"source":url,"retrieved_at":stamp,"status":"rejected","error":repr(exc)}
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(download,links))
    pd.DataFrame(rows).to_csv(folder/"statements.csv",index=False)
    print(pd.Series([row["status"] for row in rows]).value_counts().to_string(),flush=True)


def policy_features(origins, releases):
    if not origins.is_monotonic_increasing:
        raise ValueError("Forecast origins must be ordered")
    verified = releases.loc[releases.status=="verified"].copy()
    verified["published_at"] = pd.to_datetime(verified.published_at,utc=True)
    if (verified.empty or verified.published_at.isna().any() or verified.published_at.duplicated().any()
        or not verified.action.isin([-1,0,1]).all()):
        raise ValueError("Invalid verified release archive")
    known = pd.merge_asof(pd.DataFrame({"origin":pd.to_datetime(origins,utc=True)}),
                          verified.sort_values("published_at"),left_on="origin",right_on="published_at",
                          allow_exact_matches=False)
    days = (known.origin-known.published_at).dt.total_seconds().to_numpy()/86400
    action = known.action.to_numpy(dtype=float)
    return pd.DataFrame({"policy_action":action,"post_event":np.exp(-days/5),
                         "signed_impulse":action*np.exp(-days/5)},index=origins)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    fetch(parser.parse_args().folder)
