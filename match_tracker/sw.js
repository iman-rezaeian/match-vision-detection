/* Stompers Match Tracker — minimal service worker.
 *
 * Goal: make the app installable and survive flaky field-side wifi.
 * Strategy:
 *   - On install: pre-cache the app shell (HTML + icons + manifest).
 *   - On fetch: network-first for the HTML document so coaches always get the
 *     latest deploy; cache-first for static CDN assets (Tailwind, React, fonts).
 *   - Never intercept Firestore / Firebase / Google auth traffic — Firestore
 *     manages its own offline persistence.
 *
 * Bump CACHE_VERSION whenever the shell or icons change so old caches purge.
 */
const CACHE_VERSION = 'stompers-v1';
const SHELL = [
  './',
  './index.html',
  './manifest.webmanifest',
  './icon-192.png',
  './icon-512.png',
  './icon-maskable-192.png',
  './icon-maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then((cache) => cache.addAll(SHELL).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Hosts we MUST NOT intercept — Firestore handles its own offline layer.
const PASSTHROUGH_HOSTS = [
  'firestore.googleapis.com',
  'firebaseio.com',
  'firebaseinstallations.googleapis.com',
  'identitytoolkit.googleapis.com',
  'securetoken.googleapis.com',
];

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  let url;
  try { url = new URL(req.url); } catch (e) { return; }

  if (PASSTHROUGH_HOSTS.some((h) => url.hostname.endsWith(h))) return;

  // Network-first for HTML navigation: coaches always get the freshest deploy
  // when online; fall back to cached shell when offline.
  if (req.mode === 'navigate' || (req.destination === 'document')) {
    event.respondWith(
      fetch(req)
        .then((resp) => {
          const copy = resp.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(req, copy)).catch(() => {});
          return resp;
        })
        .catch(() => caches.match(req).then((r) => r || caches.match('./index.html')))
    );
    return;
  }

  // Cache-first for everything else (CDN scripts, fonts, icons).
  event.respondWith(
    caches.match(req).then((cached) => {
      if (cached) return cached;
      return fetch(req).then((resp) => {
        if (resp && resp.status === 200 && (resp.type === 'basic' || resp.type === 'cors')) {
          const copy = resp.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(req, copy)).catch(() => {});
        }
        return resp;
      }).catch(() => cached);
    })
  );
});
