'use strict';

// The launcher has no npm dependencies. Keep large archives and checkpoints on
// disk throughout download, hashing and extraction.
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const https = require('node:https');
const http = require('node:http');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');

class TransferError extends Error {
  constructor(message, code = 'TRANSFER_FAILED', extra = {}) {
    super(message); this.name = 'TransferError'; this.code = code;
    Object.assign(this, extra);
  }
}
function abortError() {
  const error = new Error('Operation paused or cancelled. Partial downloads are retained.');
  error.name = 'AbortError'; error.code = 'ABORT_ERR'; return error;
}
function checkAbort(signal) { if (signal?.aborted) throw abortError(); }
function validSize(value) {
  return Number.isSafeInteger(value) && value >= 0;
}
function validateHash(hash) {
  if (hash != null && !/^[a-f\d]{64}$/i.test(hash)) throw new TypeError('sha256 must contain 64 hexadecimal characters.');
}
function validateUrl(value, allowLocalHttp = false) {
  const url = new URL(value);
  const loopback = ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname);
  if (url.username || url.password || (url.protocol !== 'https:' && !(allowLocalHttp && loopback && url.protocol === 'http:'))) {
    throw new TransferError('Only HTTPS download URLs are allowed.', 'UNSAFE_URL');
  }
  return url;
}
function progressReporter(onProgress, total, initial = 0, phase = 'download') {
  let lastTime = performance.now(), lastBytes = initial, speed = 0;
  return (received, force = false) => {
    const now = performance.now(), elapsed = (now - lastTime) / 1000;
    if (!force && elapsed < 0.2) return;
    if (elapsed > 0 && received > lastBytes) {
      const instant = (received - lastBytes) / elapsed;
      speed = speed ? 0.65 * speed + 0.35 * instant : instant;
    }
    lastTime = now; lastBytes = received;
    onProgress?.({phase, received, total: validSize(total) ? total : null,
      speed, eta: total != null && speed > 0 ? Math.max(0, total - received) / speed : null,
      percent: total > 0 ? Math.min(100, received / total * 100) : null});
  };
}
async function regularFile(filename) {
  try {
    const stat = await fsp.lstat(filename);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new TransferError(`Expected an ordinary file: ${filename}`, 'UNSAFE_FILE');
    return stat;
  } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
async function verify({path: filename, size, sha256, onProgress, signal}) {
  validateHash(sha256); checkAbort(signal);
  if (size != null && !validSize(size)) throw new TypeError('Invalid expected file size.');
  const stat = await regularFile(filename);
  if (!stat) return {ok: false, sha256: null, size: 0, reason: 'missing'};
  if (size != null && stat.size !== size) return {ok: false, sha256: null, size: stat.size, reason: 'size'};
  const hash = crypto.createHash('sha256');
  const report = progressReporter(onProgress, stat.size, 0, 'verify');
  let bytes = 0;
  const stream = fs.createReadStream(filename, {highWaterMark: 4 * 1024 * 1024});
  const abort = () => stream.destroy(abortError());
  signal?.addEventListener('abort', abort, {once: true});
  try {
    report(0, true);
    for await (const chunk of stream) { checkAbort(signal); hash.update(chunk); bytes += chunk.length; report(bytes); }
    const digest = hash.digest('hex'); report(bytes, true);
    return {ok: bytes === stat.size && (sha256 == null || digest === sha256.toLowerCase()), sha256: digest, size: bytes,
      reason: bytes !== stat.size ? 'changed' : sha256 != null && digest !== sha256.toLowerCase() ? 'sha256' : null};
  } finally { signal?.removeEventListener('abort', abort); stream.destroy(); }
}
function nativeRequest(urlValue, {headers, signal, timeoutMs, allowLocalHttp, redirects, requestTransport}) {
  checkAbort(signal);
  const url = validateUrl(urlValue, allowLocalHttp);
  return new Promise((resolve, reject) => {
    // Electron's Chromium transport honors system proxies and the OS trust
    // store. Certificate checks stay enabled; credentials and cookies are not
    // sent. Redirects are followed here, never silently by the transport.
    const req = requestTransport({url: url.href, method: 'GET', headers: {
      'User-Agent': 'LEVIRDetNet-Launcher/3', 'Accept-Encoding': 'identity', ...headers},
      credentials: 'omit', useSessionCookies: false, cache: 'no-store',
      redirect: 'manual', bypassCustomProtocolHandlers: true});
    let response, timer, stopped = false, currentUrl = url.href;
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); };
    const fail = error => {
      if (stopped) return;
      stopped = true; cleanup();
      response?.destroy(error); req.abort(); reject(error);
    };
    const abort = () => fail(abortError());
    const armTimeout = () => {
      clearTimeout(timer);
      if (stopped) return;
      timer = setTimeout(() => fail(new TransferError('The download connection timed out.', 'ETIMEDOUT')), timeoutMs);
      timer.unref?.();
    };
    signal?.addEventListener('abort', abort, {once: true});
    req.on('redirect', (_status, _method, redirectUrl) => {
      try {
        checkAbort(signal);
        if (redirects++ >= 8) throw new TransferError('Too many download redirects.', 'REDIRECT_LIMIT');
        currentUrl = validateUrl(new URL(redirectUrl, currentUrl).href, allowLocalHttp).href;
        armTimeout(); req.followRedirect();
      } catch (error) { fail(error); }
    });
    req.on('error', error => {
      if (signal?.aborted) fail(abortError());
      else fail(new TransferError(`The download network request failed: ${error.message}`, 'NETWORK_ERROR', {retryable: true}));
    });
    req.on('response', res => {
      response = res;
      res.once('error', () => {});
      if (stopped || signal?.aborted) { res.destroy(abortError()); return; }
      // Observe readiness without consuming bytes before the downloader has
      // attached its async iterator. The native response is a Node Readable.
      res.on('readable', armTimeout);
      res.once('end', () => { stopped = true; cleanup(); });
      res.once('close', () => { stopped = true; cleanup(); req.abort(); });
      armTimeout(); resolve({response: res, url: currentUrl});
    });
    // Some Electron versions emit the writable request's close event when
    // upload finishes, before response headers; timeout/error/response events
    // govern the network transaction instead.
    armTimeout();
    if (signal?.aborted) abort(); else req.end();
  });
}
function request(urlValue, {headers = {}, signal, timeoutMs = 45000, allowLocalHttp = false, redirects = 0, requestTransport} = {}) {
  checkAbort(signal);
  if (requestTransport != null) {
    if (typeof requestTransport !== 'function') throw new TypeError('requestTransport must be a function.');
    return nativeRequest(urlValue, {headers, signal, timeoutMs, allowLocalHttp, redirects, requestTransport});
  }
  const url = validateUrl(urlValue, allowLocalHttp);
  return new Promise((resolve, reject) => {
    const req = (url.protocol === 'https:' ? https : http).get(url, {
      headers: {'User-Agent': 'LEVIRDetNet-Launcher/2', 'Accept-Encoding': 'identity', ...headers},
      // Node's default certificate and hostname validation remains enabled.
    });
    let response;
    const abort = () => { req.destroy(abortError()); response?.destroy(abortError()); };
    signal?.addEventListener('abort', abort, {once: true});
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => req.destroy(new TransferError('The download connection timed out.', 'ETIMEDOUT')));
    req.on('response', res => {
      response = res;
      if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
        res.resume(); signal?.removeEventListener('abort', abort);
        if (redirects >= 8) { reject(new TransferError('Too many download redirects.', 'REDIRECT_LIMIT')); return; }
        let next;
        try { next = validateUrl(new URL(res.headers.location, url).href, allowLocalHttp); }
        catch (error) { reject(error); return; }
        // Request headers never contain credentials or cookies from the host.
        resolve(request(next, {headers, signal, timeoutMs, allowLocalHttp, redirects: redirects + 1}));
      } else {
        const cleanup = () => signal?.removeEventListener('abort', abort);
        res.once('close', cleanup); res.once('end', cleanup);
        res.once('error', () => {}); // The body consumer reports stream errors.
        resolve({response: res, url: url.href});
      }
    });
    req.once('close', () => { if (!response) signal?.removeEventListener('abort', abort); });
  });
}
async function smallText(response, limit = 1024 * 1024) {
  const chunks = []; let bytes = 0;
  for await (const chunk of response) {
    bytes += chunk.length;
    if (bytes > limit) { response.destroy(); throw new TransferError('Unexpected HTML response is too large.', 'HTML_RESPONSE'); }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}
function htmlDecode(value) {
  return value.replace(/&amp;/gi, '&').replace(/&quot;/gi, '"').replace(/&#39;|&apos;/gi, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)));
}
function attributes(tag) {
  const result = {};
  for (const match of tag.matchAll(/([\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g)) {
    result[match[1].toLowerCase()] = htmlDecode(match[2] ?? match[3] ?? match[4]);
  }
  return result;
}
function driveConfirmation(html, baseUrl, expectedId) {
  const base = new URL(baseUrl);
  const allowed = ['drive.google.com', 'drive.usercontent.google.com', 'docs.google.com'];
  if (base.protocol !== 'https:' || !allowed.includes(base.hostname)) return null;
  for (const form of html.matchAll(/<form\b([^>]*)>([\s\S]*?)<\/form>/gi)) {
    const attr = attributes(form[1]);
    if ((attr.method || 'get').toLowerCase() !== 'get' || !attr.action) continue;
    const next = new URL(attr.action, base);
    if (next.protocol !== 'https:' || !allowed.includes(next.hostname) || next.username || next.password || !/\/(?:uc|download)$/.test(next.pathname)) continue;
    for (const input of form[2].matchAll(/<input\b([^>]*)>/gi)) {
      const field = attributes(input[1]);
      if (field.name && field.value != null && ['id', 'export', 'confirm', 'uuid', 'authuser'].includes(field.name)) next.searchParams.set(field.name, field.value);
    }
    if (!next.searchParams.has('confirm') || (expectedId && next.searchParams.get('id') !== expectedId)) continue;
    return next.href;
  }
  return null;
}
async function wait(ms, signal) {
  checkAbort(signal);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve(); }, ms);
    const abort = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); reject(abortError()); };
    signal?.addEventListener('abort', abort, {once: true});
  });
}
function manualError(error, url, destination) {
  if (error.name === 'AbortError') return error;
  error.downloadUrl = url; error.destination = destination;
  error.manual = `Download ${url} manually and save it as ${destination}, then retry. Existing files have not been overwritten.`;
  return error;
}
async function download(options) {
  const {destination: rawDestination, size, sha256, driveId, signal, onProgress, allowLocalHttp = false, timeoutMs = 45000, retries = 2} = options;
  validateHash(sha256); checkAbort(signal);
  if (size != null && !validSize(size)) throw new TypeError('Invalid expected download size.');
  if (driveId != null && !/^[\w-]{10,200}$/.test(driveId)) throw new TypeError('Invalid Google Drive file ID.');
  const destination = path.resolve(rawDestination);
  const originalUrl = driveId ? `https://drive.usercontent.google.com/download?id=${encodeURIComponent(driveId)}&export=download&authuser=0&confirm=t` : options.url;
  validateUrl(originalUrl, allowLocalHttp);
  const manualUrl = driveId ? `https://drive.google.com/file/d/${driveId}/view` : originalUrl;
  try {
    const existing = await regularFile(destination);
    if (existing) {
      // Unknown-content files cannot be declared verified just because a path exists.
      if (sha256 == null && size == null) throw new TransferError(`A file already exists at ${destination}. Move it aside before retrying.`, 'EXISTING_UNVERIFIED');
      const result = await verify({path: destination, size, sha256, signal, onProgress});
      if (!result.ok) throw new TransferError(`An existing file failed integrity verification: ${destination}. Move it aside or replace it manually.`, 'EXISTING_INVALID');
      return destination;
    }
    await fsp.mkdir(path.dirname(destination), {recursive: true});
    const part = `${destination}.part`, metaPath = `${part}.json`;
    let lastError;
    for (let attempt = 0; attempt <= retries; attempt++) {
      checkAbort(signal);
      let activeResponse;
      try {
        let offset = (await regularFile(part))?.size || 0;
        let metadata = null;
        try { metadata = JSON.parse(await fsp.readFile(metaPath, 'utf8')); } catch (error) { if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error; }
        const validator = metadata?.url === originalUrl ? metadata.etag || metadata.lastModified : null;
        // Hash-pinned files can resume without an HTTP validator. Unpinned code
        // archives must restart if the origin supplied no object validator.
        if (offset && (size != null && offset > size || !sha256 && !validator)) offset = 0;
        const headers = offset ? {Range: `bytes=${offset}-`, ...(validator ? {'If-Range': validator} : {})} : {};
        let currentUrl = originalUrl, result;
        for (let confirmation = 0; confirmation < 3; confirmation++) {
          options._onNetworkState?.(true);
          result = await request(currentUrl, {headers, signal, timeoutMs, allowLocalHttp, requestTransport: options.requestTransport});
          const res = result.response;
          activeResponse = res;
          if (/\b(?:text\/html|application\/xhtml\+xml)\b/i.test(res.headers['content-type'] || '')) {
            const html = await smallText(res);
            const next = driveId ? driveConfirmation(html, result.url, driveId) : null;
            if (next && next !== currentUrl) { currentUrl = next; result = null; continue; }
            throw new TransferError('The server returned a sign-in, sharing-permission, or download-quota page instead of a file. Open the manual download link in your browser.', 'HTML_RESPONSE');
          }
          break;
        }
        if (!result) throw new TransferError('Google Drive download confirmation did not complete.', 'DRIVE_CONFIRMATION');
        const res = result.response;
        if (res.statusCode === 416) {
          const complete = /^bytes \*\/(\d+)$/.exec(res.headers['content-range'] || ''); res.resume();
          if (!(offset && complete && Number(complete[1]) === offset && (size == null || size === offset))) {
            throw new TransferError('The server rejected the partial-file range. Move the .part file aside and retry.', 'BAD_RANGE');
          }
        } else {
          if (![200, 206].includes(res.statusCode)) {
            const status = res.statusCode; res.resume();
            throw new TransferError(`Download server returned HTTP ${status}.`, `HTTP_${status}`, {retryable: status === 429 || status >= 500});
          }
          let total = size ?? null;
          if (res.statusCode === 206) {
            const range = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(res.headers['content-range'] || '');
            if (!range || Number(range[1]) !== offset || Number(range[2]) < offset || range[3] !== '*' && Number(range[2]) >= Number(range[3])) {
              res.destroy(); throw new TransferError('The server returned an invalid partial-file range.', 'BAD_RANGE');
            }
            if (range[3] !== '*') total = Number(range[3]);
          } else {
            offset = 0; // A server ignoring Range must never append a full file.
            const length = Number(res.headers['content-length']);
            if (res.headers['content-length'] && validSize(length)) total = length;
          }
          if (size != null && total != null && size !== total) {
            res.destroy(); throw new TransferError(`Download size does not match the release manifest (${total} versus ${size} bytes).`, 'SIZE_MISMATCH');
          }
          const report = progressReporter(onProgress, total, offset);
          let received = offset, first = true;
          // Exclusive metadata files are not executable; symlinks are rejected.
          options._onNetworkState?.('suspend');
          if (await fsp.lstat(metaPath).catch(error => error.code === 'ENOENT' ? null : Promise.reject(error))) await regularFile(metaPath);
          await fsp.writeFile(metaPath, JSON.stringify({url: originalUrl, etag: res.headers.etag, lastModified: res.headers['last-modified']}));
          const handle = await fsp.open(part, offset ? 'r+' : 'w');
          options._onNetworkState?.('resume');
          report(received, true);
          try {
            for await (const chunk of res) {
              checkAbort(signal);
              if (first && offset === 0 && /^\s*(?:<!doctype\s+html|<html\b|<head\b|<form\b)/i.test(chunk.subarray(0, 256).toString('utf8'))) {
                throw new TransferError('The server returned an HTML error page instead of the requested file.', 'HTML_RESPONSE');
              }
              first = false;
              if (size != null && received + chunk.length > size) throw new TransferError('Download exceeds the expected release size.', 'SIZE_MISMATCH');
              options._onNetworkBytes?.(chunk.length);
              options._onNetworkState?.('suspend');
              try {
                let written = 0;
                while (written < chunk.length) { const result = await handle.write(chunk, written, chunk.length - written, received + written); written += result.bytesWritten; }
              } finally { options._onNetworkState?.('resume'); }
              received += chunk.length; report(received);
            }
            options._onNetworkState?.(false);
            if (total != null && received !== total) throw new TransferError('The download ended before all expected bytes arrived.', 'INCOMPLETE', {retryable: true});
            await handle.sync(); report(received, true);
          } finally { await handle.close(); res.destroy(); }
        }
        options._onNetworkState?.(false);
        checkAbort(signal);
        const checked = await verify({path: part, size, sha256, signal, onProgress});
        if (!checked.ok) {
          throw new TransferError('The downloaded file failed integrity verification. Move the .part file aside before downloading again, or use the manual link.', 'HASH_MISMATCH');
        }
        // Hard-link creation is exclusive on both Windows and Linux; unlike
        // rename(), it cannot silently replace a file created concurrently.
        try { await fsp.link(part, destination); }
        catch (error) {
          if (!['ENOTSUP', 'EOPNOTSUPP', 'EPERM', 'EXDEV'].includes(error.code)) throw error;
          // Removable/exFAT volumes may not support hard links. COPYFILE_EXCL
          // preserves the same never-overwrite guarantee on those filesystems.
          await fsp.copyFile(part, destination, fs.constants.COPYFILE_EXCL);
        }
        await fsp.unlink(part);
        await fsp.unlink(metaPath).catch(() => {});
        return destination;
      } catch (error) {
        options._onNetworkState?.(false);
        activeResponse?.destroy();
        if (options.requestTransport && /net::ERR_|fetch failed|failed to fetch/i.test(error.message || '')) {
          error = new TransferError(`The download network request failed: ${error.message}`, 'NETWORK_ERROR', {retryable: true});
        }
        checkAbort(signal); lastError = error;
        const retryable = error.retryable || ['ECONNRESET', 'ECONNREFUSED', 'ETIMEDOUT', 'EPIPE', 'ERR_STREAM_PREMATURE_CLOSE', 'EAI_AGAIN'].includes(error.code);
        if (!retryable || attempt === retries) throw error;
        onProgress?.({phase: 'retry', received: (await regularFile(part))?.size || 0, total: size ?? null, speed: 0, eta: null, percent: null, attempt: attempt + 1, message: error.message});
        await wait(Math.min(4000, 750 * 2 ** attempt), signal);
      }
    }
    throw lastError;
  } catch (error) { throw manualError(error, manualUrl, destination); }
}

// Switch only between release files pinned to the same size and SHA-256. A
// Range request can therefore reuse bytes from another host without trusting a
// host-specific ETag; the completed file is still hashed before being published.
async function downloadWithMirrors(options) {
  const {mirrors = [], signal, onProgress, allowLocalHttp = false,
    slowSpeedBytesPerSecond = 1024 * 1024, slowGraceMs = 10000,
    slowWindowMs = 30000, mirrorRetries = 2} = options;
  if (!Array.isArray(mirrors)) throw new TypeError('mirrors must be an array.');
  if (!mirrors.length) return download(options);
  if (!validSize(options.size) || !options.sha256) throw new TypeError('Mirror downloads require an expected size and SHA-256.');
  validateHash(options.sha256);
  for (const [key, value] of Object.entries({slowSpeedBytesPerSecond, slowGraceMs, slowWindowMs, mirrorRetries})) {
    if (!Number.isFinite(value) || value < 0 || key === 'slowWindowMs' && !value || key === 'mirrorRetries' && !Number.isInteger(value)) throw new TypeError(`Invalid ${key}.`);
  }
  checkAbort(signal);
  const primaryUrl = options.url || (options.driveId ? `https://drive.google.com/file/d/${options.driveId}/view` : null);
  const sources = [{name: options.sourceName || (options.driveId ? 'Google Drive' : 'Primary'),
    url: primaryUrl, driveId: options.driveId}];
  for (const mirror of mirrors) {
    if (!mirror || typeof mirror.name !== 'string' || !mirror.name.trim()) throw new TypeError('Each mirror needs a name and HTTPS URL.');
    validateUrl(mirror.url, allowLocalHttp);
    if (!sources.some(source => source.url === mirror.url)) sources.push({name: mirror.name, url: mirror.url});
  }
  validateUrl(primaryUrl, allowLocalHttp);
  const destination = path.resolve(options.destination);
  const attempts = [], rejectedParts = [];
  let lastError;
  const sourceFailure = error => /^HTTP_\d+$/.test(error.code || '') || [
    'HTML_RESPONSE', 'DRIVE_CONFIRMATION', 'REDIRECT_LIMIT', 'BAD_RANGE',
    'SIZE_MISMATCH', 'HASH_MISMATCH', 'INCOMPLETE', 'NETWORK_SLOW', 'NETWORK_ERROR',
    'ECONNRESET', 'ECONNREFUSED', 'ECONNABORTED', 'ETIMEDOUT', 'EPIPE',
    'ERR_STREAM_PREMATURE_CLOSE', 'EAI_AGAIN', 'ENOTFOUND',
    'CERT_HAS_EXPIRED', 'CERT_NOT_YET_VALID', 'UNABLE_TO_VERIFY_LEAF_SIGNATURE',
    'UNABLE_TO_GET_ISSUER_CERT_LOCALLY', 'DEPTH_ZERO_SELF_SIGNED_CERT',
    'SELF_SIGNED_CERT_IN_CHAIN', 'ERR_TLS_CERT_ALTNAME_INVALID',
  ].includes(error.code);
  const details = error => {
    error.attemptedSources = attempts.map(item => ({...item}));
    error.manualSources = sources.map(source => ({name: source.name, url: source.url}));
    error.rejectedParts = [...rejectedParts];
    return manualError(error, sources[sources.length - 1].url, destination);
  };
  const discardInvalidPart = async () => {
    // Keep failed bytes for diagnosis instead of overwriting a user's final
    // file. Exclusive hard-link/copy also makes the quarantine name collision
    // safe; a fresh transfer will start at offset zero.
    const part = `${destination}.part`;
    if (!await regularFile(part)) return;
    const rejected = `${part}.invalid-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
    try { await fsp.link(part, rejected); }
    catch (error) {
      if (!['ENOTSUP', 'EOPNOTSUPP', 'EPERM', 'EXDEV'].includes(error.code)) throw error;
      await fsp.copyFile(part, rejected, fs.constants.COPYFILE_EXCL);
    }
    await fsp.unlink(part);
    const metaPath = `${part}.json`;
    if (await regularFile(metaPath)) await fsp.rename(metaPath, `${rejected}.json`);
    rejectedParts.push(rejected);
  };
  for (let index = 0; index < sources.length; index++) {
    const source = sources[index];
    let restarted = false;
    for (;;) {
      checkAbort(signal);
      const resumedBytes = (await regularFile(`${destination}.part`))?.size || 0;
      const internal = new AbortController();
      const abort = () => internal.abort();
      signal?.addEventListener('abort', abort, {once: true});
      if (signal?.aborted) abort();
      let timer = null, active = false, started = 0, bytes = 0, baseline = null, suspendedAt = null, slowError;
      const stopMonitor = () => { if (timer) clearInterval(timer); timer = null; active = false; baseline = null; suspendedAt = null; };
      const networkState = enabled => {
        if (enabled === 'suspend') {
          if (active && suspendedAt == null) suspendedAt = performance.now();
          return;
        }
        if (enabled === 'resume') {
          if (suspendedAt != null) {
            const pausedFor = performance.now() - suspendedAt;
            started += pausedFor;
            if (baseline) baseline.time += pausedFor;
            suspendedAt = null;
          }
          return;
        }
        if (!enabled) { stopMonitor(); return; }
        if (active || index === sources.length - 1 || slowSpeedBytesPerSecond === 0) return;
        active = true; started = performance.now(); bytes = 0; baseline = null;
        // Independent of body progress callbacks: header waits and a body that
        // sends no bytes must also trigger the fallback.
        timer = setInterval(() => {
          if (suspendedAt != null) return;
          const now = performance.now(), elapsed = now - started;
          if (elapsed < slowGraceMs) return;
          if (!baseline) { baseline = {time: now, bytes}; return; }
          const window = now - baseline.time;
          if (window < slowWindowMs) return;
          const measuredSpeed = (bytes - baseline.bytes) / (window / 1000);
          if (measuredSpeed < slowSpeedBytesPerSecond) {
            slowError = new TransferError(`${source.name} remained below ${slowSpeedBytesPerSecond} bytes/s for ${Math.round(window / 1000)} seconds; switching download source.`,
              'NETWORK_SLOW', {speed: measuredSpeed});
            stopMonitor(); internal.abort();
          } else baseline = {time: now, bytes};
        }, Math.max(10, Math.min(250, slowWindowMs / 10)));
        timer.unref?.();
      };
      const report = event => {
        if (['verify', 'retry', 'extract'].includes(event.phase)) stopMonitor();
        onProgress?.({...event, source: source.name, sourceUrl: source.url});
      };
      try {
        return await download({...options, url: source.url, driveId: source.driveId,
          signal: internal.signal, retries: index === 0 ? 0 : mirrorRetries,
          onProgress: report, _onNetworkState: networkState,
          _onNetworkBytes: count => { bytes += count; }});
      } catch (caught) {
        checkAbort(signal); // A user Pause always wins over an internal timeout.
        const error = slowError || caught;
        if (error.name === 'AbortError' || !sourceFailure(error)) throw details(error);
        lastError = error;
        attempts.push({name: source.name, url: source.url, code: error.code, message: error.message});
        if (error.code === 'HASH_MISMATCH') {
          // A slow primary may already have supplied bad bytes. If a resumed
          // mirror fails the hash, retry that mirror once from zero rather than
          // repeatedly appending to the same corrupt prefix.
          const retryFresh = index > 0 && resumedBytes > 0 && !restarted;
          if (retryFresh || index < sources.length - 1) {
            try { await discardInvalidPart(); } catch (localError) { throw details(localError); }
          }
          if (retryFresh) {
            restarted = true;
            report({phase: 'source', received: 0, total: options.size, percent: 0,
              speed: 0, eta: null, reason: 'HASH_MISMATCH',
              message: 'The resumed file failed verification; retrying this source from the beginning.'});
            continue;
          }
        }
        if (index + 1 < sources.length) {
          const next = sources[index + 1];
          const received = (await regularFile(`${destination}.part`))?.size || 0;
          onProgress?.({phase: 'source', source: next.name, sourceUrl: next.url,
            previousSource: source.name, reason: error.code, message: error.message,
            received, total: options.size, percent: options.size ? received / options.size * 100 : null,
            speed: 0, eta: null});
        }
        break;
      } finally {
        stopMonitor(); signal?.removeEventListener('abort', abort);
      }
    }
  }
  throw details(lastError);
}

// Validate the ZIP directory before invoking an OS extractor. This deliberately
// rejects ZIP64 (unnecessary for the small source archive), links, special files,
// ambiguous Windows names, duplicate entries and any path escaping the target.
function safeEntry(name) {
  if (!name || name.includes('\0') || name.includes('\\') || name.startsWith('/') || name.includes(':')) return false;
  const parts = name.replace(/\/$/, '').split('/');
  return parts.every(part => part && part !== '.' && part !== '..' && !/[. ]$/.test(part) && !/[<>"|?*\x00-\x1f]/.test(part) && !/^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)/i.test(part));
}
async function inspectZip(archive, signal) {
  checkAbort(signal);
  const stat = await regularFile(archive);
  if (!stat || stat.size < 22) throw new TransferError('Invalid or missing ZIP archive.', 'INVALID_ZIP');
  const handle = await fsp.open(archive, 'r');
  try {
    const tailSize = Math.min(stat.size, 65557), tail = Buffer.alloc(tailSize);
    await handle.read(tail, 0, tailSize, stat.size - tailSize);
    let end = -1;
    for (let i = tail.length - 22; i >= 0; i--) {
      if (tail.readUInt32LE(i) === 0x06054b50 && i + 22 + tail.readUInt16LE(i + 20) === tail.length) { end = i; break; }
    }
    if (end < 0) throw new TransferError('ZIP end directory was not found.', 'INVALID_ZIP');
    const entries = tail.readUInt16LE(end + 10), directorySize = tail.readUInt32LE(end + 12), directoryOffset = tail.readUInt32LE(end + 16);
    if (tail.readUInt16LE(end + 4) || tail.readUInt16LE(end + 6) || tail.readUInt16LE(end + 8) !== entries || entries === 0xffff || directoryOffset === 0xffffffff || directorySize > 64 * 1024 * 1024 || directoryOffset + directorySize > stat.size - tailSize + end) throw new TransferError('Split, ZIP64 or oversized ZIP directories are not supported.', 'INVALID_ZIP');
    const directory = Buffer.alloc(directorySize); await handle.read(directory, 0, directorySize, directoryOffset);
    const names = new Map(), result = []; let at = 0, expanded = 0;
    for (let index = 0; index < entries; index++) {
      checkAbort(signal);
      if (at + 46 > directory.length || directory.readUInt32LE(at) !== 0x02014b50) throw new TransferError('Invalid ZIP entry directory.', 'INVALID_ZIP');
      const flags = directory.readUInt16LE(at + 8), method = directory.readUInt16LE(at + 10), uncompressed = directory.readUInt32LE(at + 24);
      const nameSize = directory.readUInt16LE(at + 28), extraSize = directory.readUInt16LE(at + 30), commentSize = directory.readUInt16LE(at + 32);
      const attr = directory.readUInt32LE(at + 38), localOffset = directory.readUInt32LE(at + 42), next = at + 46 + nameSize + extraSize + commentSize;
      if (next > directory.length || flags & 1 || ![0, 8].includes(method) || uncompressed === 0xffffffff || localOffset === 0xffffffff) throw new TransferError('Encrypted or unsupported ZIP entry.', 'INVALID_ZIP');
      const nameBytes = directory.subarray(at + 46, at + 46 + nameSize), name = nameBytes.toString('utf8');
      if (!safeEntry(name) || name.includes('\ufffd')) throw new TransferError(`Unsafe ZIP path: ${name}`, 'UNSAFE_ZIP');
      const type = (attr >>> 16) & 0xf000;
      if (type && ![0x4000, 0x8000].includes(type)) throw new TransferError(`ZIP links or special files are not allowed: ${name}`, 'UNSAFE_ZIP');
      const key = name.replace(/\/$/, '').toLowerCase();
      if (names.has(key)) throw new TransferError(`Duplicate ZIP path: ${name}`, 'UNSAFE_ZIP');
      names.set(key, name.endsWith('/'));
      const local = Buffer.alloc(30); await handle.read(local, 0, 30, localOffset);
      if (local.readUInt32LE(0) !== 0x04034b50 || local.readUInt16LE(26) !== nameSize || local.readUInt16LE(8) !== method) throw new TransferError('ZIP local headers disagree with the directory.', 'INVALID_ZIP');
      const localName = Buffer.alloc(nameSize); await handle.read(localName, 0, nameSize, localOffset + 30);
      if (!localName.equals(nameBytes)) throw new TransferError('ZIP path differs between headers.', 'UNSAFE_ZIP');
      expanded += uncompressed;
      if (expanded > 16 * 1024 ** 3 || entries > 60000) throw new TransferError('ZIP extraction exceeds the safety size limit.', 'ZIP_TOO_LARGE');
      result.push({name, size: uncompressed, directory: name.endsWith('/')}); at = next;
    }
    for (const [name] of names) {
      const parts = name.split('/'); parts.pop();
      while (parts.length) { if (names.get(parts.join('/')) === false) throw new TransferError('ZIP file/directory path collision.', 'UNSAFE_ZIP'); parts.pop(); }
    }
    return {entries: result, expandedSize: expanded};
  } finally { await handle.close(); }
}
function run(command, args, signal) {
  checkAbort(signal);
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {windowsHide: true, stdio: ['ignore', 'ignore', 'pipe']});
    let stderr = '';
    child.stderr.on('data', chunk => { if (stderr.length < 8000) stderr += chunk.toString(); });
    const abort = () => child.kill(); signal?.addEventListener('abort', abort, {once: true});
    child.once('error', error => { signal?.removeEventListener('abort', abort); reject(error); });
    child.once('close', code => {
      signal?.removeEventListener('abort', abort);
      if (signal?.aborted) reject(abortError());
      else if (code) reject(new TransferError(`ZIP extraction failed (${code}): ${stderr.trim()}`, 'EXTRACTION_FAILED'));
      else resolve();
    });
  });
}
async function extractZip({archive, destination, signal, onProgress}) {
  const target = path.resolve(destination), parent = path.dirname(target);
  if (target === parent) throw new TransferError('Extraction to a filesystem root is not allowed.', 'UNSAFE_DESTINATION');
  const inspected = await inspectZip(archive, signal);
  await fsp.mkdir(parent, {recursive: true});
  await fsp.mkdir(target); // Refuse any existing destination; never merge or overwrite.
  try {
    onProgress?.({phase: 'extract', received: 0, total: inspected.expandedSize, percent: null, speed: 0, eta: null});
    if (process.platform === 'win32') {
      const quote = text => `'${text.replace(/'/g, "''")}'`;
      const script = `$ErrorActionPreference = 'Stop'; Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::ExtractToDirectory(${quote(path.resolve(archive))}, ${quote(target)})`;
      await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', Buffer.from(script, 'utf16le').toString('base64')], signal);
    } else {
      await run('unzip', ['-q', path.resolve(archive), '-d', target], signal);
    }
    checkAbort(signal);
    for (const entry of inspected.entries) {
      const filename = path.join(target, ...entry.name.split('/'));
      const stat = await fsp.lstat(filename);
      if (stat.isSymbolicLink() || (entry.directory ? !stat.isDirectory() : !stat.isFile() || stat.size !== entry.size)) throw new TransferError('Extracted files do not match the validated ZIP directory.', 'EXTRACTION_INVALID');
    }
    onProgress?.({phase: 'extract', received: inspected.expandedSize, total: inspected.expandedSize, percent: 100, speed: 0, eta: 0});
    return target;
  } catch (error) {
    // Only the exact, newly-created target is removed; its parent is preserved.
    if (path.dirname(target) === parent && target !== parent) await fsp.rm(target, {recursive: true, force: true}).catch(() => {});
    throw error;
  }
}

module.exports = {download, downloadWithMirrors, verify, extractZip, inspectZip, TransferError,
  _test: {driveConfirmation, safeEntry, validateUrl, request}};
