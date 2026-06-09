#!/usr/bin/env python3
"""
DNS Ad Classifier — Render.com ML service
Classifies domains as ad/tracker vs normal using packet size + domain features.
Trained on AGH blocklist labels + heuristics.
"""
from flask import Flask, request, jsonify
import numpy as np
import re, os, json
from model import DnsAdClassifier

app = Flask(__name__)
clf = DnsAdClassifier()

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_trained": clf.trained})

@app.route("/train", methods=["POST"])
def train():
    """
    Body: { "domains": [ {name, avg_pkt, std_pkt, freq, blocked} ] }
    blocked=true means AGH blocklist already flagged it (ground truth label).
    """
    body = request.get_json(force=True)
    domains = body.get("domains", [])
    if len(domains) < 5:
        return jsonify({"error": "need >= 5 samples to train"}), 400
    result = clf.train(domains)
    return jsonify(result)

@app.route("/classify", methods=["POST"])
def classify():
    """
    Body: { "domains": [ {name, avg_pkt, std_pkt, freq, clients, latency_avg} ] }
    Returns each domain with ad_prob (0.0-1.0) and label (ad/normal/youtube_ad).
    """
    body = request.get_json(force=True)
    domains = body.get("domains", [])
    if not domains:
        return jsonify({"results": []})
    results = clf.classify(domains)
    return jsonify({"results": results})

@app.route("/classify_youtube", methods=["POST"])
def classify_youtube():
    """
    Specialized YouTube ad classifier using packet size bursts.
    Body: { "packets": [ {domain, size, ts_ms} ] }
    Looks for IMA SDK patterns + size anomalies during video playback.
    """
    body   = request.get_json(force=True)
    packets = body.get("packets", [])
    results = clf.classify_youtube_ads(packets)
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
