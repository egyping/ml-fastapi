// frontend-layer/src/lib/api.js
export async function estimatePrice(payload) {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) {
      const msg = await res.text().catch(() => '')
      throw new Error(`HTTP ${res.status}: ${msg}`)
    }
    return res.json()
  }
  