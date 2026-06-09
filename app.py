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
API_KEY = os.environ.get("API_KEY", "")

def require_key():
    if not API_KEY:
        return None  # no key set = open (dev mode)
    k = request.headers.get("X-Api-Key", "")
    if k != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return None

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_trained": clf.trained})

@app.route("/train", methods=["POST"])
def train():
    err = require_key()
    if err: return err
    body = request.get_json(force=True)
    domains = body.get("domains", [])
    if len(domains) < 5:
        return jsonify({"error": "need >= 5 samples to train"}), 400
    result = clf.train(domains)
    return jsonify(result)

@app.route("/classify", methods=["POST"])
def classify():
    err = require_key()
    if err: return err
    body = request.get_json(force=True)
    domains = body.get("domains", [])
    if not domains:
        return jsonify({"results": []})
    results = clf.classify(domains)
    return jsonify({"results": results})

@app.route("/classify_youtube", methods=["POST"])
def classify_youtube():
    err = require_key()
    if err: return err
    body   = request.get_json(force=True)
    packets = body.get("packets", [])
    results = clf.classify_youtube_ads(packets)
    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
