/* Shared utilities for Logistics Intelligence frontend */

/** Resolve API base URL for backend communication. */
const API_BASE = (() => {
    // Explicit override (useful for Railway / custom local routing).
    const override = (localStorage.getItem('apiBase') || '').trim();
    if (override) {
        return override.replace(/\/+$/, '');
    }

    // In production, frontend and backend can share origin.
    // In local dev, frontend may be served from 8080 while API is on 8000.
    const host = String(window.location.hostname || '').trim().toLowerCase();
    const isLoopback = host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === '[::1]';
    if (isLoopback && window.location.port !== '' && window.location.port !== '8000') {
        const apiHost = host === '::1' || host === '[::1]' ? '[::1]' : host;
        return `http://${apiHost}:8000`;
    }
    return '';
})();

/** Maximum upload file size in bytes (50 MB). */
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

/** Maximum chat message length. */
const MAX_MESSAGE_LENGTH = 4000;

/** Allowed industries (must match backend). */
const ALLOWED_INDUSTRIES = ['ecommerce', 'shipping_freight', 'trucking_delivery', 'other_industry'];
const DEFAULT_INDUSTRY = 'ecommerce';

/** Allowed categories per industry (must match backend). */
const ALLOWED_CATEGORIES = {
    ecommerce: ['checkout_conversion', 'order_fulfillment', 'return_rate', 'cart_abandonment', 'demand_forecasting', 'inventory_management'],
    shipping_freight: ['port_congestion', 'freight_cost', 'container_utilization', 'transit_time', 'customs_clearance', 'carrier_performance'],
    trucking_delivery: ['on_time_delivery', 'route_optimization', 'fleet_utilization', 'fuel_cost', 'driver_performance', 'last_mile_delivery'],
    other_industry: [],
};

/** Validate an industry value against the whitelist. Returns the value if valid, empty string if not. */
function validateIndustry(value) {
    const normalized = _normalizeIndustryValue(value);
    if (!normalized) return '';
    return ALLOWED_INDUSTRIES.includes(normalized) ? normalized : '';
}

/** Validate a category value for a given industry. Returns the value if valid, empty string if not. */
function validateCategory(value, industry) {
    if (!value) return '';
    const clean = String(value).trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    if (!clean) return '';
    const allowed = ALLOWED_CATEGORIES[industry];
    // For other_industry or unknown industries, accept any sanitized category (max 64 chars)
    if (!allowed || allowed.length === 0) return clean.slice(0, 64);
    return allowed.includes(clean) ? clean : '';
}

/* ── Theme ───────────────────────────────────────────── */

function initTheme() {
    document.documentElement.setAttribute('data-theme', 'light');
    return 'light';
}

function toggleTheme() { return 'light'; }

function currentTheme() { return 'light'; }

/* ── Session helpers ─────────────────────────────────── */

async function ensureSession() {
    const status = await fetch(`${API_BASE}/session/status`);
    const statusJson = await status.json();
    if (status.ok && statusJson.has_session) return;
    const start = await fetch(`${API_BASE}/session/start`, { method: 'POST' });
    const startJson = await start.json();
    if (!start.ok) {
        throw new Error(extractError(startJson, 'Unable to start session'));
    }
}

/* ── Task polling with backoff (#11) ─────────────────── */

async function waitForReport(taskId, { onProgress, timeoutMs = 20 * 60 * 1000 } = {}) {
    const started = Date.now();
    let interval = 1200;
    const MAX_INTERVAL = 5000;
    const BACKOFF_AFTER = 30000;

    while (true) {
        const res = await fetch(`${API_BASE}/task/${encodeURIComponent(taskId)}`);
        const payload = await res.json();
        if (!res.ok) {
            throw new Error(extractError(payload, 'Task polling failed'));
        }

        if (onProgress) onProgress(payload);

        if (payload.state === 'REPORT_READY') {
            const reportId = payload.report_id;
            const reportRes = await fetch(`${API_BASE}/report/${encodeURIComponent(reportId)}`);
            const reportJson = await reportRes.json();
            if (!reportRes.ok) {
                throw new Error(extractError(reportJson, 'Failed to fetch report'));
            }
            reportJson.report_id = reportId;
            return reportJson;
        }
        if (payload.state === 'INSUFFICIENT_CONTEXT') {
            const result = payload.result || {};
            const err = new Error(result.message || 'Insufficient business context to generate a reliable report.');
            err.insufficientContext = true;
            err.missingFields = result.missing_fields || [];
            err.missingCount = result.missing_count || 0;
            throw err;
        }
        if (payload.state === 'ERROR') {
            throw new Error(payload.error || 'Prediction task failed');
        }
        if ((Date.now() - started) > timeoutMs) {
            throw new Error('Prediction timed out');
        }

        // Exponential backoff after initial period (#11)
        if ((Date.now() - started) > BACKOFF_AFTER && interval < MAX_INTERVAL) {
            interval = Math.min(interval * 1.5, MAX_INTERVAL);
        }
        await new Promise((resolve) => setTimeout(resolve, interval));
    }
}

/* ── Error extraction ────────────────────────────────── */

function extractError(payload, fallback) {
    if (!payload) return fallback;
    if (typeof payload === 'string') return payload;
    if (payload.detail) {
        if (typeof payload.detail === 'string') return payload.detail;
        if (payload.detail.error) return String(payload.detail.error);
    }
    if (payload.error) return payload.error;
    return fallback;
}

/* ── Safe text helpers (XSS prevention) ──────────────── */

/** Escape HTML entities to prevent XSS when building markup strings. */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str || '');
    return div.innerHTML;
}

/** Format a numeric or string value for display. */
function formatValue(value) {
    if (value === null || value === undefined) return '\u2014';
    if (typeof value === 'number') {
        if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        return value.toFixed(4).replace(/\.?0+$/, '');
    }
    return String(value);
}

/* ── Industry helpers ────────────────────────────────── */

function getSelectedIndustry() {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get('industry');
    const fromStorage = sessionStorage.getItem('selectedIndustry');
    const selected = _normalizeIndustryValue(fromQuery || fromStorage || DEFAULT_INDUSTRY);
    return selected || DEFAULT_INDUSTRY;
}

function setSelectedIndustry(value) {
    const safe = _normalizeIndustryValue(value) || DEFAULT_INDUSTRY;
    sessionStorage.setItem('selectedIndustry', safe);
    const select = document.getElementById('industrySelect');
    if (select) {
        if (Array.from(select.options || []).some((opt) => String(opt.value) === safe)) {
            if (select.value !== safe) select.value = safe;
        } else if (Array.from(select.options || []).some((opt) => String(opt.value) === '__custom__')) {
            select.value = '__custom__';
        }
    }
}

function _normalizeIndustryValue(value) {
    const raw = String(value || '').trim().toLowerCase();
    if (!raw) return '';
    if (ALLOWED_INDUSTRIES.includes(raw)) return raw;
    const token = raw.replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
    if (!token) return '';
    // Strict: only return if it matches the whitelist
    if (ALLOWED_INDUSTRIES.includes(token)) return token;
    return 'other_industry';
}

function isKnownIndustry(value) {
    return ALLOWED_INDUSTRIES.includes(_normalizeIndustryValue(value));
}

function titleCaseWords(value) {
    const acronyms = new Set(['id', 'api', 'sku', 'kpi', 'roi', 'sla', 'otp', 'eta', 'csa', 'hos', 'pdf', 'csv', 'xls', 'xlsx', 'json']);
    return String(value || '')
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .split(' ')
        .filter(Boolean)
        .map((word) => {
            const lower = word.toLowerCase();
            if (acronyms.has(lower)) return lower.toUpperCase();
            return lower.charAt(0).toUpperCase() + lower.slice(1);
        })
        .join(' ');
}

function displayIndustryName(industryValue, customIndustryName = '') {
    const normalized = _normalizeIndustryValue(industryValue);
    const custom = String(customIndustryName || '').trim();
    if (normalized === 'other_industry') {
        return custom ? titleCaseWords(custom) : 'Other Industry';
    }
    return titleCaseWords(normalized || industryValue || '');
}

function displayCategoryName(categoryValue) {
    const raw = String(categoryValue || '').trim();
    if (!raw || raw === 'unclassified') return 'General Operations';
    return titleCaseWords(raw);
}
