"""Resume and verify the published Aura shadow checkpoints without touching serving."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from forecast_registry import home, sha

MODELS={
    "chronos":("amazon/chronos-2","29ec3766d36d6f73f0696f85560a422f50e8498c"),
    "timesfm":("google/timesfm-2.5-200m-transformers","5a9806b9b291fad9233b5249d88263f1846304d3"),
}
CHUNK=6*1024**2


def acquire(kind,workers):
    from huggingface_hub import HfApi
    from filelock import FileLock

    repo,revision=MODELS[kind]
    info=HfApi().model_info(repo,revision=revision,files_metadata=True)
    if info.sha!=revision:
        raise ValueError("Unexpected Hub revision")
    files={f.rfilename:f for f in info.siblings}
    weight=files["model.safetensors"]
    digest=weight.lfs.sha256
    root=home()/"foundation-checkpoints"/f"{kind}-{revision}"
    root.mkdir(parents=True,exist_ok=True)
    base=f"https://huggingface.co/{repo}/resolve/{revision}"
    with FileLock(str(root/"download.lock"),timeout=0):
        for name in ("config.json","generation_config.json"):
            if name not in files:
                continue
            with urlopen(base+"/"+name,timeout=30) as body:
                raw=body.read()
            blob=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
            if blob!=files[name].blob_id:
                raise ValueError(f"Invalid configuration hash: {name}")
            (root/name).write_bytes(raw)
        path=root/"model.safetensors"
        if not path.exists() or sha(path)!=digest:
            ranges=[(i,min(weight.size-1,i+CHUNK-1)) for i in range(0,weight.size,CHUNK)]
            chunks=root/"chunks"
            chunks.mkdir(exist_ok=True)
            def download(pair):
                start,end=pair
                local=chunks/f"{start}.part"
                for attempt in range(4):
                    offset=start+(local.stat().st_size if local.exists() else 0)
                    if offset==end+1:
                        return local
                    request=Request(f"{base}/model.safetensors?download=true&range={offset}",
                                    headers={"Range":f"bytes={offset}-{end}","Accept-Encoding":"identity"})
                    try:
                        with urlopen(request,timeout=65) as response:
                            if response.status!=206 or response.headers.get("Content-Range")!=f"bytes {offset}-{end}/{weight.size}":
                                raise ValueError("Official CDN did not honor the requested exact byte range")
                            with local.open("ab") as out:
                                remaining=end-offset+1
                                while remaining:
                                    block=response.read(min(remaining,256*1024))
                                    if not block:
                                        raise EOFError("Interrupted checkpoint range")
                                    out.write(block)
                                    remaining-=len(block)
                        return local
                    except ValueError:
                        raise
                    except Exception:
                        if attempt==3:
                            raise
                        time.sleep(2)
            print(f"{kind}: acquiring {weight.size} bytes in {len(ranges)} verified ranges",flush=True)
            with ThreadPoolExecutor(workers) as executor:
                futures={executor.submit(download,pair):pair[0] for pair in ranges}
                paths={}
                failures=[]
                for future in as_completed(futures):
                    try:
                        paths[futures[future]]=future.result()
                    except Exception as exc:
                        failures.append(type(exc).__name__)
                        continue
                    if len(paths)%10==0:
                        print(f"{kind}: {len(paths)}/{len(ranges)} ranges complete",flush=True)
            if failures:
                raise RuntimeError(f"{len(failures)} checkpoint ranges need retry: {sorted(set(failures))}")
            temporary=root/"model.assembling"
            with temporary.open("wb") as assembled:
                for local in (paths[start] for start,_ in ranges):
                    with local.open("rb") as source:
                        shutil.copyfileobj(source,assembled)
            if temporary.stat().st_size!=weight.size or sha(temporary)!=digest:
                raise ValueError("Invalid checkpoint checksum after assembling ranges")
            temporary.replace(path)
        manifest={"kind":kind,"model_id":repo,"revision":revision,"sha256":digest,
                  "size":weight.size,"directory":str(root)}
        (root/"verified.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
        print(f"{kind}: checkpoint verified at {root}",flush=True)
        return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=MODELS)
    parser.add_argument("--workers",type=int,choices=range(1,33),default=16)
    a=parser.parse_args()
    acquire(a.kind,a.workers)
