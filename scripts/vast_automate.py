#!/usr/bin/env python3
"""
Automate Vast.ai workflow for SAM 2.1 LoRA fine-tuning:
 - Enforce GPU guardrails (GPU in A100|A10|L40S, VRAM >=24 GB, price <=$1.20/hr unless overridden)
 - Generate Run Manifest (dataset counts, 5 train + 5 predict stems, model/config, hyperparams, GPU filters)
 - Launch instance (PyTorch 2.5.1 CUDA 12.4 image), upload workspace only (no external drives)
 - Install deps + SAM2, auto-fetch SAM2.1 checkpoint/config if missing
 - Train Round-1 (epochs=12, batch=2, lr=1e-4), predict 5 unlabeled, download artifacts, terminate instance

Usage:
    python scripts/vast_automate.py --api-key <KEY> --identity ~/.ssh/id_rsa

Notes:
- Run from repo root. Uses ./Maps and ./mask in the repo.
- Ensure your SSH public key is registered in Vast console.
- Set VASTAI_API_KEY env var to avoid passing --api-key.
"""
import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

from vastai_sdk import VastAI
import time, re

from scripts.run_manifest import build_manifest
from src.config_types import Config
from src.fs_utils import list_images


def _vram_gb(item: dict) -> float:
    for k in ("gpu_ram", "gpu_ram_gb", "gpu_memory", "mem_gb"):
        v = item.get(k)
        if v is None:
            continue
        try:
            val = float(v)
            # Vast commonly reports gpu_ram in MB; normalize to GB.
            # Heuristic: if value is greater than 128, assume MB and convert.
            if val > 128:
                return val / 1024.0
            return val
        except Exception:
            continue
    return 0.0


def pick_offer(vast: VastAI, allowed_regex: str, min_vram_gb: int, max_price: float, attempts: int = 60, sleep_s: int = 15,
               reliability: float = 0.95, allow_non_datacenter: bool = False, allow_unverified: bool = False) -> dict:
    # Keep the server-side query broad to avoid schema mismatches; refine client-side.
    q_parts = [f"reliability>={reliability}", "rentable=true", f"dph<={max_price}"]
    if not allow_unverified:
        q_parts.append("verified=true")
    if not allow_non_datacenter:
        q_parts.append("datacenter=true")
    if allowed_regex and str(allowed_regex).strip():
        q_parts.append(f"gpu_name~{allowed_regex}")
    q = " ".join(q_parts)
    for i in range(attempts):
        offers = vast.search_offers(query=q)
        if isinstance(offers, dict):
            offers = offers.get("offers", [])
        # Client-side refine: enforce VRAM and sort by effective price
        refined = []
        for o in (offers or []):
            price = _price_of(o)
            vram = _vram_gb(o)
            if price <= max_price and vram >= float(min_vram_gb):
                refined.append(o)
        refined.sort(key=_price_of)
        if refined:
            return refined[0]
        print(f"No offers match constraints (try {i+1}/{attempts}). Query: {q} | offers_seen={len(offers or [])}")
        time.sleep(sleep_s)
    raise RuntimeError(f"No offers match constraints after {attempts} attempts: {q}")


def _price_of(item: dict) -> float:
    for k in ("dph", "dph_total", "dph_base", "actual_dph"):
        v = item.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                continue
    return 9999.0


def pick_existing_running_instance(vast: VastAI, allowed_regex: str, min_vram_gb: int, max_price: float) -> dict | None:
    """Return the cheapest already-running instance owned by the user that matches constraints, else None."""
    try:
        lst = vast.show_instances() or []
    except Exception:
        return None
    pat = re.compile(allowed_regex) if (allowed_regex is not None) else re.compile("")
    candidates = []
    for it in lst:
        state = (it.get("state") or it.get("status") or "").lower()
        if state not in ("running", "active"):
            continue
        gpu_name = (it.get("gpu_name") or it.get("gpu_type") or "")
        if not pat.search(gpu_name):
            continue
        try:
            vram = float(it.get("gpu_ram", 0))
        except Exception:
            vram = _vram_gb(it)
        if vram < min_vram_gb:
            continue
        price = _price_of(it)
        if price > max_price:
            continue
        candidates.append(it)
    if not candidates:
        return None
    candidates.sort(key=_price_of)
    return candidates[0]


def pick_existing_stopped_instance(vast: VastAI, allowed_regex: str, min_vram_gb: int, max_price: float) -> dict | None:
    """Return the cheapest stopped/paused instance owned by the user that matches constraints, else None."""
    try:
        lst = vast.show_instances() or []
    except Exception:
        return None
    pat = re.compile(allowed_regex) if (allowed_regex is not None) else re.compile("")
    candidates = []
    for it in lst:
        state = (it.get("state") or it.get("status") or "").lower()
        if state not in ("stopped", "exited", "created"):
            continue
        gpu_name = (it.get("gpu_name") or it.get("gpu_type") or "")
        if not pat.search(gpu_name):
            continue
        try:
            vram = float(it.get("gpu_ram", 0))
        except Exception:
            vram = _vram_gb(it)
        if vram < min_vram_gb:
            continue
        price = _price_of(it)
        if price > max_price:
            continue
        candidates.append(it)
    if not candidates:
        return None
    candidates.sort(key=_price_of)
    return candidates[0]


def ensure_identity_and_register(vast: VastAI, identity_path: str) -> str:
    path = Path(identity_path).expanduser()
    pub = Path(str(path) + ".pub")
    if not path.exists():
        # generate key
        import subprocess
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["ssh-keygen", "-t", "rsa", "-b", "4096", "-N", "", "-f", str(path)])
    if not pub.exists():
        raise RuntimeError(f"Missing public key: {pub}")
    pub_text = pub.read_text().strip()
    # register if not present
    try:
        keys = vast.show_ssh_keys() or []
        already = any(pub_text in (k.get("ssh_key", "")) for k in keys)
        if not already:
            vast.create_ssh_key(ssh_key=pub_text)
    except Exception:
        # best-effort
        pass
    return str(path)


def ensure_repo_ready(repo: Path):
    # Make sure required folders exist
    for p in [repo / "Maps", repo / "mask", repo / "weights", repo / "outputs"]:
        p.mkdir(parents=True, exist_ok=True)


def wait_running(vast: VastAI, instance_id: int, timeout=900, poll=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            info = vast.show_instance(id=int(instance_id))
            state = (info.get("state") or info.get("status") or "").lower()
            print("State:", state)
            if state in ("running", "active"):
                return info
        except Exception as e:
            print("Poll error:", e)
        time.sleep(poll)
    raise TimeoutError("Instance did not reach running state in time")


def rexec(vast: VastAI, instance_id: int, cmd: str):
    print("$", cmd)
    out = vast.execute(ID=int(instance_id), COMMAND=f"bash -lc '{cmd}'")
    print(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("VASTAI_API_KEY", ""))
    ap.add_argument("--gpu", default="", help="Allowed GPUs regex (empty = no filter)")
    ap.add_argument("--min_vram", type=int, default=16)
    # Accept both --max_price and --max-price
    ap.add_argument("--max_price", "--max-price", dest="max_price", type=float, default=float(os.getenv("VAST_MAX_PRICE", 1.20)))
    ap.add_argument("--identity", default=os.path.expanduser("~/.ssh/id_rsa"))
    ap.add_argument("--disk", default="60")
    ap.add_argument("--image", default="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", help="Docker image to run.")
    ap.add_argument("--label", default="lorasam2-sam2-lora")
    ap.add_argument("--reuse_existing", action="store_true", default=True, help="Reuse an already running instance if it matches constraints")
    ap.add_argument("--reliability", type=float, default=0.70)
    ap.add_argument("--allow-non-datacenter", dest="allow_non_datacenter", action="store_true", default=True)
    ap.add_argument("--allow-unverified", dest="allow_unverified", action="store_true", default=True)
    # Round-1 hyperparams
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--only-search", dest="only_search", action="store_true", default=False, help="Only search and print the top 10 matching offers, then exit.")
    ap.add_argument("--use-cli-search", dest="use_cli_search", action="store_true", default=False, help="Use vast CLI (vastai) for searching offers instead of SDK.")
    args = ap.parse_args()

    if not args.api_key:
        print("Missing API key. Provide --api-key or set VASTAI_API_KEY env var.")
        sys.exit(1)
    # Ensure identity exists and is registered with Vast
    identity_path = ensure_identity_and_register(vast=VastAI(api_key=args.api_key), identity_path=args.identity)

    repo = Path.cwd()
    ensure_repo_ready(repo)

    vast = VastAI(api_key=args.api_key)

    # Build and print Run Manifest
    mani = build_manifest(images_dir=str(repo / "Maps"), masks_dir=str(repo / "mask"),
                          train_k=5, predict_k=5, model_name="SAM 2.1 (hiera-large)",
                          config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
                          epochs=args.epochs, batch=args.batch, lr=args.lr,
                          gpu_filter=args.gpu, min_vram=args.min_vram, max_price=args.max_price,
                          will_auto_fetch_ckpt=True)
    import json
    print(f"Using price ceiling: ${args.max_price:.2f}/hr")
    print("Run Manifest:\n" + json.dumps(mani, indent=2))

    instance_id = None
    created_new = False
    # Debug-only: search and print offers, then exit.
    if args.only_search:
        print("Searching for offers only (no launch)...")
        offers = []
        if args.use_cli_search:
            # Try the vast CLI (vastai). If not present, download vast.py as 'vast' into ~/.local/bin
            import shutil, subprocess
            cli = shutil.which("vast") or shutil.which("vastai")
            if not cli:
                try:
                    home = str(Path.home())
                    binp = Path(home) / ".local/bin"
                    binp.mkdir(parents=True, exist_ok=True)
                    subprocess.check_call(["bash", "-lc", "wget -q https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O ~/.local/bin/vast && chmod +x ~/.local/bin/vast"])
                    cli = str(binp / "vast")
                except Exception as e:
                    print("Failed to install vast CLI:", e)
            if cli:
                # Build CLI filters similar to our query
                flt = ["search", "offers", "--raw"]
                flt += ["--reliability", str(args.reliability), "--price_max", str(args.max_price), "--rentable"]
                if not args.allow_unverified:
                    flt += ["--verified"]
                if not args.allow_non_datacenter:
                    flt += ["--datacenter"]
                if args.gpu and str(args.gpu).strip():
                    flt += ["--gpu_name", args.gpu]
                try:
                    env = os.environ.copy()
                    if args.api_key:
                        env["VAST_API_KEY"] = args.api_key
                        env["VASTAI_API_KEY"] = args.api_key
                    out = subprocess.check_output([cli] + flt, env=env, text=True)
                    import json as _json
                    data = _json.loads(out)
                    offers = data.get("offers", data if isinstance(data, list) else [])
                except Exception as e:
                    print("vast CLI search failed:", e)
        if not offers:
            # Fallback to SDK
            q = f"reliability>={args.reliability} rentable=true dph<={args.max_price}"
            if not args.allow_unverified:
                q += " verified=true"
            if not args.allow_non_datacenter:
                q += " datacenter=true"
            if args.gpu and str(args.gpu).strip():
                q += f" gpu_name~{args.gpu}"
            offers = vast.search_offers(query=q)
            if isinstance(offers, dict):
                offers = offers.get("offers", [])
        # Refine locally and print top 20 (more visibility)
        refined = []
        for o in offers or []:
            pr = _price_of(o)
            vr = _vram_gb(o)
            if pr <= args.max_price and vr >= float(args.min_vram):
                refined.append(o)
        refined.sort(key=_price_of)
        print(f"Found {len(refined)} refined offers under ${args.max_price}/hr with >= {args.min_vram} GB VRAM")
        for i, o in enumerate(refined[:20]):
            try:
                print(f"#{i+1}: id={o.get('id')} gpu={o.get('gpu_name')} vram={_vram_gb(o)}GB price=${_price_of(o):.3f}/hr loc={o.get('geolocation','')} verified={o.get('verified')} datacenter={o.get('datacenter')}")
            except Exception as e:
                print("Error printing offer:", e, o)
        return
    if args.reuse_existing:
        existing = pick_existing_running_instance(vast, allowed_regex=args.gpu, min_vram_gb=args.min_vram, max_price=args.max_price)
        if existing:
            instance_id = existing.get("id")
            print("Reusing running instance:", existing)
        else:
            stopped = pick_existing_stopped_instance(vast, allowed_regex=args.gpu, min_vram_gb=args.min_vram, max_price=args.max_price)
            if stopped:
                iid = stopped.get("id")
                print("Starting stopped instance:", stopped)
                try:
                    vast.start_instance(id=int(iid))
                except Exception as e:
                    print("Failed to start instance:", e)
                instance_id = iid
    if not instance_id:
        offer = pick_offer(vast, allowed_regex=args.gpu, min_vram_gb=args.min_vram, max_price=args.max_price,
                           attempts=60, sleep_s=15, reliability=args.reliability,
                           allow_non_datacenter=args.allow_non_datacenter, allow_unverified=args.allow_unverified)
        print("Selected offer:", offer)

    if not instance_id:
        print("Launching instance...")
        resp = None
        # Prefer launching by ask_id to avoid gpu_name enum validation issues in SDK
        try:
            resp = vast.launch_instance(ask_id=offer.get("id"), image=args.image, disk=args.disk, label=args.label)
        except Exception as e:
            print("Launch by ask_id failed, retrying with gpu_name normalization:", e)
            # Normalize GPU name (e.g., 'RTX 4090' -> 'RTX_4090', 'Q RTX 6000' -> 'Q_RTX_6000')
            raw = offer.get("gpu_name", "") or ""
            norm = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
            try:
                resp = vast.launch_instance(num_gpus="1", gpu_name=norm, image=args.image, disk=args.disk, label=args.label)
            except Exception as e2:
                print(f"Launch with gpu_name={norm} failed:", e2)
                raise
        created_new = True
        if isinstance(resp, dict):
            instance_id = resp.get("new_contract") or resp.get("instance_id") or resp.get("id")
        if not instance_id:
            # fallback by label
            lst = vast.show_instances()
            for it in (lst or []):
                if it.get("label") == args.label:
                    instance_id = it.get("id")
                    break
        if not instance_id:
            print("Could not determine instance ID after launch.")
            sys.exit(1)

    # Attach SSH key explicitly (best-effort)
    try:
        pubkey = Path(identity_path + ".pub").read_text().strip()
        vast.attach_ssh(instance_id=int(instance_id), ssh_key=pubkey)
    except Exception:
        pass

    wait_running(vast, instance_id)

    workdir = "/workspace/lorasam2"
    # Tarball repo
    tar_path = Path("/tmp/lorasam2_repo.tgz").resolve()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(repo, arcname="lorasam2")
    print("Tar created:", tar_path)

    # Upload
    remote_tar = f"{instance_id}:/workspace/lorasam2.tgz"
    vast.copy(src=str(tar_path), dst=remote_tar, identity=args.identity)
    print("Upload done.")

    # Unpack + deps + train (SAM2)
    rexec(vast, instance_id, f"mkdir -p {workdir} && cd /workspace && tar xzf /workspace/lorasam2.tgz && rm -f /workspace/lorasam2.tgz && mv -f /workspace/lorasam2/* {workdir} || true")
    # Ensure correct pip + install reqs (torch already present in image); install SAM2
    rexec(vast, instance_id, f"cd {workdir} && python3 -m pip install --upgrade pip && pip install -r requirements.txt && pip install 'git+https://github.com/facebookresearch/sam2.git'")
    # Train Round-1
    rexec(vast, instance_id, f"cd {workdir} && python train_lora_sam2.py --images_dir ./Maps --masks_dir ./mask --out_dir ./outputs/sam2_lora --epochs {args.epochs} --batch {args.batch} --lr {args.lr}")
    # Predict 5 unlabeled
    rexec(vast, instance_id, f"cd {workdir} && python scripts/predict_batch_sam2.py --images_dir ./Maps --masks_dir ./mask --out_dir ./outputs/preds --lora ./outputs/sam2_lora/best.pt --n 5")

    # Download outputs
    local_out = (repo / "outputs_from_vast").resolve()
    local_out.mkdir(parents=True, exist_ok=True)
    vast.copy(src=f"{instance_id}:{workdir}/outputs", dst=str(local_out), identity=args.identity)
    print("Outputs downloaded to:", local_out)
    print("Done. Correct the 5 predicted masks, save in ./mask as <stem>_mask.png, then rerun training with --resume.")

    # Terminate only if we created it; otherwise leave user's instance running
    if created_new:
        try:
            vast.destroy_instance(id=int(instance_id))
            print("Instance terminated.")
        except Exception as e:
            print("Terminate failed:", e)


if __name__ == "__main__":
    main()
