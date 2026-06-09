#!/usr/bin/env python3
import numpy as np
import re, json, os
from collections import defaultdict

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN = True
except ImportError:
    SKLEARN = False

MODEL_PATH  = "/tmp/dns_clf.joblib"
SCALER_PATH = "/tmp/dns_scaler.joblib"

# Known YouTube ad / IMA SDK domains — highest confidence labels
YOUTUBE_AD_DOMAINS = {
    "imasdk.googleapis.com",
    "static.doubleclick.net",
    "googleads.g.doubleclick.net",
    "video-stats.l.google.com",
    "ad.youtube.com",
    "ads.youtube.com",
    "s.youtube.com",          # ad stats beacon
    "www.youtube.com",        # sometimes ad redirect
}

# Regex patterns that strongly suggest ad/tracker
AD_PATTERNS = re.compile(
    r"(^|\.)(ad|ads|adserver|adtrack|adnxs|doubleclick|googlesyndication"
    r"|googletagmanager|googletagservices|analytics|tracker|track|pixel"
    r"|beacon|telemetry|metric|stats|log|collect|event|imasdk"
    r"|adsystem|adservice|adtech|moatads|scorecardresearch"
    r"|omtrdc|2mdn|googlevideo\.com$)",
    re.IGNORECASE
)

def extract_features(d: dict) -> list:
    """
    Features:
    0  avg_pkt_size     - avg packet size in bytes
    1  std_pkt_size     - std dev of packet sizes
    2  freq             - queries per window
    3  clients          - unique client count
    4  latency_avg      - avg DNS response latency ms
    5  domain_depth     - subdomain count (a.b.c.com = 3)
    6  name_len         - domain name total length
    7  has_ad_keyword   - regex match on name
    8  is_youtube_family- 1 if *.youtube.com / *.googlevideo.com
    9  tld_numeric      - TLD encoded: .com=0 .net=1 .io=2 .org=3 other=4
    """
    name        = d.get("name", "")
    avg_pkt     = float(d.get("avg_pkt", d.get("avg_pkt_size", 0)))
    std_pkt     = float(d.get("std_pkt", d.get("std_pkt_size", 0)))
    freq        = float(d.get("freq", d.get("count", 1)))
    clients     = float(d.get("clients", 1))
    latency     = float(d.get("latency_avg", d.get("latency", 0)))
    parts       = name.rstrip(".").split(".")
    depth       = len(parts) - 2 if len(parts) > 2 else 0
    name_len    = len(name)
    has_ad      = 1.0 if AD_PATTERNS.search(name) else 0.0
    yt_family   = 1.0 if any(name.endswith(s) for s in
                    ("youtube.com","googlevideo.com","ytimg.com","yt.be")) else 0.0
    tld = parts[-1].lower() if parts else ""
    tld_num = {"com":0,"net":1,"io":2,"org":3}.get(tld, 4)

    return [avg_pkt, std_pkt, freq, clients, latency,
            depth, name_len, has_ad, yt_family, float(tld_num)]

def rule_based_label(name: str, avg_pkt: float) -> tuple:
    """Returns (label, confidence) using hard rules before ML."""
    if name in YOUTUBE_AD_DOMAINS:
        return "youtube_ad", 0.99
    if AD_PATTERNS.search(name):
        return "ad", 0.90
    # Small DNS packets (<80 bytes) with high freq = likely tracker beacon
    if avg_pkt < 80 and avg_pkt > 0:
        return "tracker", 0.70
    return None, 0.0

class DnsAdClassifier:
    def __init__(self):
        self.trained  = False
        self.clf      = None
        self.scaler   = None
        self._try_load()

    def _try_load(self):
        if not SKLEARN:
            return
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.clf     = joblib.load(MODEL_PATH)
                self.scaler  = joblib.load(SCALER_PATH)
                self.trained = True
            except Exception:
                pass

    def train(self, domains: list) -> dict:
        if not SKLEARN:
            return {"error": "sklearn not installed"}

        X, y = [], []
        for d in domains:
            feats   = extract_features(d)
            blocked = d.get("blocked", False)
            # Ground truth: AGH blocked = ad (1), not blocked = normal (0)
            label   = 1 if blocked else 0
            # Override with hard rules
            rl, conf = rule_based_label(d.get("name",""), feats[0])
            if rl in ("youtube_ad","ad","tracker"):
                label = 1
            X.append(feats)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Need at least 2 classes
        if len(set(y)) < 2:
            return {"error": "need both ad and normal samples", "samples": len(domains)}

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.clf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=42
        )
        self.clf.fit(Xs, y)
        self.trained = True

        joblib.dump(self.clf,    MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        importances = dict(zip(
            ["avg_pkt","std_pkt","freq","clients","latency",
             "depth","name_len","has_ad","yt_family","tld"],
            self.clf.feature_importances_.tolist()
        ))
        return {
            "trained": True,
            "samples": len(domains),
            "ad_count": int(sum(y)),
            "normal_count": int(len(y) - sum(y)),
            "feature_importance": importances
        }

    def classify(self, domains: list) -> list:
        results = []
        for d in domains:
            name    = d.get("name", "")
            feats   = extract_features(d)
            avg_pkt = feats[0]

            # Hard rules first
            rl, conf = rule_based_label(name, avg_pkt)
            if rl:
                results.append({
                    "name":     name,
                    "label":    rl,
                    "ad_prob":  conf,
                    "method":   "rule"
                })
                continue

            # ML if trained
            if self.trained and SKLEARN:
                try:
                    Xs   = self.scaler.transform([feats])
                    prob = float(self.clf.predict_proba(Xs)[0][1])
                    label = "ad" if prob >= 0.6 else "normal"
                    results.append({
                        "name":     name,
                        "label":    label,
                        "ad_prob":  round(prob, 3),
                        "method":   "ml"
                    })
                    continue
                except Exception:
                    pass

            # Fallback: heuristic score
            score = 0.0
            if avg_pkt < 80 and avg_pkt > 0:  score += 0.3
            if avg_pkt > 1200:                  score += 0.1  # large CDN = likely content
            if feats[6] > 40:                   score += 0.2  # long name
            if feats[5] > 3:                    score += 0.15 # deep subdomain
            score = min(score, 0.59)  # below threshold until trained
            results.append({
                "name":    name,
                "label":   "unknown",
                "ad_prob": round(score, 3),
                "method":  "heuristic"
            })

        return results

    def classify_youtube_ads(self, packets: list) -> list:
        """
        YouTube ad detection via packet size burst pattern.
        During ad: IMA SDK request (small ~200-400 byte DNS) followed by
        rapid googlevideo.com queries.
        """
        if not packets:
            return []

        # Group by domain
        by_domain = defaultdict(list)
        for p in packets:
            by_domain[p.get("domain","")].append({
                "size": p.get("size", 0),
                "ts":   p.get("ts_ms", 0)
            })

        results = []
        for domain, pkts in by_domain.items():
            sizes = [p["size"] for p in pkts]
            avg   = sum(sizes) / len(sizes) if sizes else 0
            std   = float(np.std(sizes)) if len(sizes) > 1 else 0

            # Signature: IMA SDK small request burst
            ima_sig   = domain == "imasdk.googleapis.com"
            # Signature: rapid small packets (ad manifest) then large (video)
            burst_sig = std > 200 and avg < 500 and len(pkts) > 3

            if ima_sig:
                label, prob = "youtube_ad", 0.98
            elif burst_sig and "googlevideo" in domain:
                label, prob = "youtube_ad_candidate", 0.65
            elif AD_PATTERNS.search(domain):
                label, prob = "ad", 0.88
            else:
                label, prob = "normal", 0.1

            results.append({
                "name":       domain,
                "label":      label,
                "ad_prob":    round(prob, 3),
                "avg_size":   round(avg),
                "std_size":   round(std),
                "pkt_count":  len(pkts),
                "method":     "youtube_burst"
            })

        return sorted(results, key=lambda x: -x["ad_prob"])
