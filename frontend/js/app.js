// Variables & State
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// DOM Elements
const toastEl = document.getElementById('toast');
const loginPhraseEl = document.getElementById('login-phrase');

// Panel switching
function switchTab(tab, element = null) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active-panel'));

    const btn = element || document.querySelector(`.tab-btn[onclick*="${tab}"]`);
    if (btn) btn.classList.add('active');

    const panel = document.getElementById(`${tab}-panel`);
    if (panel) panel.classList.add('active-panel');

    if (tab === 'login') {
        fetchLivenessPhrase();
    }
}

// Fetch liveness phrase from API
async function fetchLivenessPhrase() {
    try {
        const res = await fetch('/api/liveness-phrase');
        const data = await res.json();
        if (loginPhraseEl) loginPhraseEl.innerText = data.phrase;
    } catch (e) {
        if (loginPhraseEl) loginPhraseEl.innerText = 'Connection error. Please refresh.';
    }
}

if (document.getElementById('login-phrase')) fetchLivenessPhrase();

// Safely extract a human-readable error string from a FastAPI response
// FastAPI can return detail as: a string, an object, or an array of validation error objects
function getErrorMessage(data, fallback = 'An error occurred') {
    if (!data) return fallback;
    const d = data.detail;
    if (!d) return data.message || fallback;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
        // FastAPI 422 validation errors: [{loc, msg, type}, ...]
        return d.map(e => e.msg || JSON.stringify(e)).join(', ');
    }
    if (typeof d === 'object') return d.msg || JSON.stringify(d);
    return String(d);
}

function showToast(message, isError = false) {
    if (!toastEl) return;
    // Guard: if someone accidentally passes an object, stringify it
    toastEl.innerText = (typeof message === 'string') ? message : JSON.stringify(message);
    if (isError) toastEl.classList.add('error');
    else toastEl.classList.remove('error');
    toastEl.classList.add('show');
    setTimeout(() => toastEl.classList.remove('show'), 3500);
}

// Setup MediaStream
async function setupAudio() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showToast('Audio recording not supported on this origin.', true);
            return false;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const preferredTypes = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/mp4'
        ];
        const pickedType = preferredTypes.find(t => MediaRecorder.isTypeSupported(t));
        mediaRecorder = pickedType
            ? new MediaRecorder(stream, { mimeType: pickedType })
            : new MediaRecorder(stream);
        mediaRecorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };
        return true;
    } catch (err) {
        showToast('Mic access denied: ' + err.message, true);
        return false;
    }
}

// Record a timed block and return Blob on completion
async function startRecordingBlock(buttonId, visualizerId, statusId, durationMs, callback) {
    if (isRecording) return;
    const valid = await setupAudio();
    if (!valid) return;

    isRecording = true;
    audioChunks = [];

    const btn = document.getElementById(buttonId);
    const viz = document.getElementById(visualizerId);
    const statusEl = document.getElementById(statusId);

    if (!btn || !viz || !statusEl) {
        isRecording = false;
        return;
    }

    btn.classList.add('recording');
    viz.classList.add('active');
    btn.disabled = true;

    const totalSecs = durationMs / 1000;
    let elapsed = 0;
    statusEl.innerText = `Recording... ${totalSecs}s`;

    const ticker = setInterval(() => {
        elapsed++;
        const remaining = totalSecs - elapsed;
        if (remaining > 0) statusEl.innerText = `Recording... ${remaining}s`;
    }, 1000);

    const recorder = mediaRecorder;
    recorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });

        // Always release microphone tracks to avoid keeping the mic active.
        const stream = recorder.stream;
        if (stream) stream.getTracks().forEach(track => track.stop());

        callback(blob);
        isRecording = false;
        btn.disabled = false;
    };

    recorder.start();

    setTimeout(() => {
        clearInterval(ticker);
        btn.classList.remove('recording');
        viz.classList.remove('active');
        statusEl.innerText = 'Processing...';
        recorder.stop();
    }, durationMs);
}

// ENROLLMENT WIZARD (3 steps x 15 seconds)
// 35+ word sentences with maximum phoneme diversity for ECAPA-TDNN.
// Covers plosives (p/b/t/d/k/g), fricatives (f/v/s/z/sh), nasals (m/n), vowels.
const ENROLL_PHRASES = [
    'The bright morning sun rises slowly over the distant mountains, painting the sky in shades of gold and orange while birds begin to sing their gentle songs in the trees.',
    'Speaking every word carefully and clearly helps the system capture the unique patterns in my voice, so I must take my time and speak each syllable with full confidence.',
    'My voice contains special frequency patterns that no other person in the world has, and by recording these phrases the system will learn to recognize me accurately every time.'
];

let enrollStep = 0;
let enrollBlobs = [];

function updateEnrollUI() {
    const stepLabelEl = document.getElementById('enroll-step-label');
    const phraseEl = document.getElementById('enroll-phrase');
    const statusEl = document.getElementById('register-status');

    if (stepLabelEl) stepLabelEl.innerText = `Step ${enrollStep + 1} of 3 - Read the phrase aloud clearly`;
    if (phraseEl) phraseEl.innerText = ENROLL_PHRASES[enrollStep];
    if (statusEl) statusEl.innerText = `Click to record sample ${enrollStep + 1} of 3 (15s)`;

    for (let i = 1; i <= 3; i++) {
        const dot = document.getElementById(`step-dot-${i}`);
        if (!dot) continue;
        if (i - 1 < enrollStep) dot.style.color = '#00f2fe'; // done
        else if (i - 1 === enrollStep) dot.style.color = '#4facfe'; // current
        else dot.style.color = '#555'; // pending
    }
}

const registerBtn = document.getElementById('register-record-btn');
if (registerBtn) {
    updateEnrollUI();

    registerBtn.addEventListener('click', () => {
        const usernameInput = document.getElementById('register-username');
        const username = usernameInput ? usernameInput.value.trim() : '';

        if (!username) {
            showToast('Please enter a username first', true);
            return;
        }

        startRecordingBlock('register-record-btn', 'register-visualizer', 'register-status', 15000, async (blob) => {
            if (!blob || blob.size === 0) {
                showToast('Recording failed. Please allow mic access and try again.', true);
                enrollStep = 0;
                enrollBlobs = [];
                updateEnrollUI();
                return;
            }
            enrollBlobs.push(blob);

            if (enrollStep < 2) {
                enrollStep++;
                updateEnrollUI();
                showToast(`Sample ${enrollStep} recorded. Read the next phrase carefully.`);
            } else {
                // All 3 done - submit
                const registerStatus = document.getElementById('register-status');
                if (registerStatus) registerStatus.innerText = 'Uploading voice profile...';

                const formData = new FormData();
                formData.append('username', username);
                formData.append('audio1', enrollBlobs[0], 'sample1.webm');
                formData.append('audio2', enrollBlobs[1], 'sample2.webm');
                formData.append('audio3', enrollBlobs[2], 'sample3.webm');

                try {
                    const res = await fetch('/api/register', { method: 'POST', body: formData });
                    const data = await res.json();

                    if (res.ok) {
                        showToast(data.message || 'Registration successful!');
                        if (usernameInput) usernameInput.value = '';
                        enrollStep = 0;
                        enrollBlobs = [];
                        updateEnrollUI();
                        setTimeout(() => switchTab('login'), 800);
                    } else {
                        showToast(getErrorMessage(data, 'Registration failed'), true);
                        enrollStep = 0;
                        enrollBlobs = [];
                        updateEnrollUI();
                    }
                } catch (e) {
                    showToast('Network error: ' + e.message, true);
                    enrollStep = 0;
                    enrollBlobs = [];
                    updateEnrollUI();
                }
            }
        });
    });
}

// LOGIN (10-second recording)
const loginBtn = document.getElementById('login-record-btn');
if (loginBtn) {
    loginBtn.addEventListener('click', () => {
        const usernameEl = document.getElementById('login-username');
        const username = usernameEl ? usernameEl.value.trim() : '';

        startRecordingBlock('login-record-btn', 'login-visualizer', 'login-status', 10000, async (blob) => {
            const formData = new FormData();
            if (username) formData.append('username', username);
            formData.append('audio', blob, 'login.webm');

            try {
                const res = await fetch('/api/login', { method: 'POST', body: formData });
                const data = await res.json();

                if (res.ok) {
                    showToast(`Login Success! - ${data.username}`);
                    localStorage.setItem('voice_token', data.access_token);
                    localStorage.setItem('voice_username', data.username);
                    localStorage.setItem('voice_confidence', data.confidence);
                    localStorage.setItem('voice_risk', data.risk_level);
                    setTimeout(() => {
                        window.location.href = '/dashboard';
                    }, 1000);
                } else {
                    showToast(getErrorMessage(data, 'Authentication failed'), true);
                    fetchLivenessPhrase();
                }
            } catch (e) {
                showToast('Network error: ' + e.message, true);
            }

            const loginStatus = document.getElementById('login-status');
            if (loginStatus) loginStatus.innerText = 'Click to speak (10s)';
        });
    });
}

function logout() {
    localStorage.clear();
    window.location.href = '/';
}
