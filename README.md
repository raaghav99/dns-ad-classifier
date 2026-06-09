# DNS Ad Classifier — Render.com ML Service

Random Forest classifier for DNS ad/tracker detection based on packet sizes.

## Endpoints
- `GET  /health`            — liveness check (wake-on-send)
- `POST /train`             — train model on labeled domain data
- `POST /classify`          — classify domains, returns ad_prob 0.0–1.0
- `POST /classify_youtube`  — YouTube-specific burst pattern detector

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to render.com → New → Web Service → connect repo
3. Render auto-detects `render.yaml`
4. After deploy, copy your URL (e.g. `https://dns-ad-classifier.onrender.com`)
5. On phone: `export RENDER_ML_URL=https://dns-ad-classifier.onrender.com`
   Or add to start_pihole.sh before cron starts

## Phone Integration

Set in start_pihole.sh (before cron line):
```sh
export RENDER_ML_URL=https://your-app.onrender.com
```

Or set in cron env:
```
RENDER_ML_URL=https://your-app.onrender.com
*/15 * * * * root flock -n /tmp/ai_dns_monitor.lock python3 /usr/local/bin/ai_dns_monitor.py >> /var/log/ai_dns_monitor.log 2>&1
```
