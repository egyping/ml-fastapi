import { useState } from "react";
import { estimatePrice } from "./lib/api";

const TYPES = ["apartment", "duplex", "villa"];
const YES_NO = ["no", "yes"];

export default function EstimatorForm() {
  const [form, setForm] = useState({
    type: "apartment",   // required
    area: 120,           // required
    bedrooms: 3,         // required
    bathrooms: 2,        // required
    city: "cairo",       // required
    region: "",          // required
    level: "",           // optional
    furnished: "no",     // optional (default)
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const onChange = (e) => {
    const { name, value } = e.target;
    if (["area", "bedrooms", "bathrooms"].includes(name)) {
      setForm((f) => ({ ...f, [name]: value === "" ? "" : Number(value) }));
    } else {
      setForm((f) => ({ ...f, [name]: value }));
    }
  };

  const requiredOk =
    form.type &&
    form.area > 0 &&
    form.bedrooms !== "" &&
    form.bathrooms !== "" &&
    String(form.city || "").trim() &&
    String(form.region || "").trim();

  async function submit(e) {
    e.preventDefault();
    setErr(""); setResult(null);
    if (!requiredOk) {
      setErr("Please fill all required fields.");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        type: String(form.type).toLowerCase(),
        area: Number(form.area),
        bedrooms: Number(form.bedrooms),
        bathrooms: Number(form.bathrooms),
        level: String(form.level ?? ""),
        furnished: String(form.furnished ?? "no"),
        city: String(form.city).toLowerCase().trim(),
        region: String(form.region).toLowerCase().trim(),
      };
      const data = await estimatePrice(payload);
      setResult(data);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel">
      <header className="panel__header">
        <h1 className="title">
          EG Property Estimator <span className="badge">beta</span>
        </h1>
        <p className="subtitle">
          Fill the form and get an estimated price in EGP. Required fields are marked
          with <span className="req">*</span>.
        </p>
      </header>

      <form className="form" onSubmit={submit}>
        {/* Row 1 */}
        <div className="grid">
          <div className="field">
            <label className="label">Type <span className="req">*</span></label>
            <select name="type" value={form.type} onChange={onChange} className="input">
              {TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div className="field">
            <label className="label">Area (m²) <span className="req">*</span></label>
            <input name="area" type="number" min="1" value={form.area}
                   onChange={onChange} className="input" placeholder="e.g. 120" />
          </div>

          <div className="field">
            <label className="label">Level (optional)</label>
            <input name="level" value={form.level} onChange={onChange}
                   className="input" placeholder="e.g. 5 or 'ground'" />
          </div>
        </div>

        {/* Row 2 */}
        <div className="grid">
          <div className="field">
            <label className="label">Bedrooms <span className="req">*</span></label>
            <input name="bedrooms" type="number" min="0" value={form.bedrooms}
                   onChange={onChange} className="input" placeholder="e.g. 3" />
          </div>

          <div className="field">
            <label className="label">Bathrooms <span className="req">*</span></label>
            <input name="bathrooms" type="number" min="0" value={form.bathrooms}
                   onChange={onChange} className="input" placeholder="e.g. 2" />
          </div>

          <div className="field">
            <label className="label">Furnished (optional)</label>
            <select name="furnished" value={form.furnished} onChange={onChange} className="input">
              {YES_NO.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
        </div>

        {/* Row 3 */}
        <div className="grid2">
          <div className="field">
            <label className="label">City <span className="req">*</span></label>
            <input name="city" value={form.city} onChange={onChange}
                   className="input" placeholder="e.g. cairo" />
          </div>

          <div className="field">
            <label className="label">Region <span className="req">*</span></label>
            <input name="region" value={form.region} onChange={onChange}
                   className="input" placeholder="e.g. zahraa al maadi" />
          </div>
        </div>

        <div className="actions">
          <button type="submit" className="btn" disabled={loading || !requiredOk}>
            {loading ? "Estimating..." : "Estimate"}
          </button>
          {!requiredOk && <span className="hint">Fill required fields.</span>}
          {err && <span className="error">{err}</span>}
        </div>
      </form>

      {result && (
        <div className="result">
          <h3 className="result__title">
            Estimated Price: {Number(result.price).toLocaleString()} {result.currency}
          </h3>
          <p className="result__meta">
            Price per m² (model): <b>{Math.round(result.details.ppm2_model)}</b> ·{" "}
            Price per m² (baseline): <b>{Math.round(result.details.ppm2_baseline)}</b> ·{" "}
            Blend α: <b>{result.details.blend_alpha}</b>
          </p>
          <p className="result__note">
            This is a starting point. Accuracy improves as we add geo & text features.
          </p>
        </div>
      )}
    </section>
  );
}
