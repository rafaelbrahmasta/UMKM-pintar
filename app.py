from flask import Flask, render_template, request, jsonify
import sqlite3
import pickle
import random
import re
import os
import numpy as np
from datetime import datetime
 
app = Flask(__name__)
 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATABASE   = os.path.join(BASE_DIR, 'umkm_pintar.db')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_model.pkl')
 
# =====================================================
# OLLAMA SETUP
# =====================================================
 
import ollama as ollama_lib
 
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_READY = False
 
SYSTEM_PROMPT = """Kamu adalah asisten AI bernama UMKM Pintar, ahli bisnis dan keuangan UMKM Indonesia.
 
Tugasmu membantu pemilik UMKM dengan:
- Analisis keuangan: laba kotor, laba bersih, margin profit, BEP, HPP, ROI
- Strategi pemasaran digital untuk UMKM
- Manajemen arus kas dan pembukuan sederhana
- Sumber modal: KUR, BPUM, P2P lending
- Legalitas usaha: NIB, PIRT, sertifikat halal
- Pajak UMKM (PP 23/2018)
- Tips manajemen stok dan operasional
 
Aturan menjawab:
- Selalu jawab dalam Bahasa Indonesia yang ramah dan mudah dipahami
- Gunakan contoh angka dalam Rupiah (Rp) jika menjelaskan perhitungan
- Jawaban ringkas tapi lengkap, maksimal 3-4 paragraf
- Gunakan bullet point jika ada daftar
- Jika ditanya di luar topik UMKM/bisnis, arahkan kembali ke topik bisnis"""
 
def cek_ollama():
    global OLLAMA_READY
    try:
        models      = ollama_lib.list()
        model_names = [m.model for m in models.models]
        print(f"   Model tersedia: {model_names}")
        # Cek exact match atau partial match
        if any(OLLAMA_MODEL in name or name in OLLAMA_MODEL for name in model_names):
            OLLAMA_READY = True
            print(f"✅ Ollama siap! Model: {OLLAMA_MODEL}")
        else:
            OLLAMA_READY = False
            print(f"⚠️  Model {OLLAMA_MODEL} tidak ditemukan.")
    except Exception as e:
        print(f"⚠️  Ollama tidak berjalan: {e}")
    return OLLAMA_READY
 
def tanya_ollama(pesan: str, history: list = []) -> str:
    """Kirim pesan ke Ollama dan dapat balasan."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": pesan})
    response = ollama_lib.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.7, "top_p": 0.9, "num_predict": 512}
    )
    return response.message.content.strip()
 
# Simpan history chat per sesi (in-memory)
chat_sessions = {}
 
# =====================================================
# INISIALISASI DATABASE
# =====================================================
 
def init_db():
    conn   = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            confidence   REAL DEFAULT 0,
            source       TEXT DEFAULT 'tfidf',
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transaksi (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            jenis      TEXT NOT NULL,
            kategori   TEXT NOT NULL,
            nominal    REAL NOT NULL,
            keterangan TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_bulanan (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            bulan             TEXT NOT NULL,
            tahun             INTEGER NOT NULL,
            total_pendapatan  REAL DEFAULT 0,
            hpp               REAL DEFAULT 0,
            biaya_operasional REAL DEFAULT 0,
            biaya_pemasaran   REAL DEFAULT 0,
            biaya_lain        REAL DEFAULT 0,
            catatan           TEXT,
            laba_kotor        REAL DEFAULT 0,
            laba_bersih       REAL DEFAULT 0,
            margin_profit     REAL DEFAULT 0,
            created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migrasi: tambah kolom source jika tabel lama belum punya
    try:
        cursor.execute("ALTER TABLE chat_logs ADD COLUMN source TEXT DEFAULT 'tfidf'")
    except Exception:
        pass
    conn.commit()
    conn.close()
    print("✅ Database siap!")
 
# =====================================================
# PREPROCESSING TF-IDF
# =====================================================
 
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","ada","pada",
    "bisa","saya","kamu","anda","adalah","juga","sudah","akan","atau","jika",
    "maka","ya","ok","oke","deh","dong","sih","nih","lah","kah","nya","pun",
    "agar","supaya","tetapi","namun","tapi","karena","sebab","jadi","mau",
    "ingin","minta","mohon","buat","bikin","dapat","bagi","oleh","secara",
    "hal","lebih","kepada","dalam","sebuah","suatu"
}
 
def preprocess(text: str) -> str:
    text = text.lower().strip()
    pairs = [
        (r"\bumkm\b",  "usaha mikro kecil menengah umkm"),
        (r"\bbep\b",   "break even point bep titik impas"),
        (r"\bhpp\b",   "harga pokok penjualan produksi hpp"),
        (r"\broi\b",   "return on investment roi"),
        (r"\bkur\b",   "kredit usaha rakyat kur"),
        (r"\bnib\b",   "nomor induk berusaha nib"),
        (r"\bswot\b",  "strengths weaknesses opportunities threats swot"),
        (r"\busp\b",   "unique selling point usp"),
        (r"\bfifo\b",  "first in first out fifo"),
    ]
    for p, r in pairs:
        text = re.sub(p, r, text)
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS_ID and len(t) > 1]
    return " ".join(tokens)
 
# =====================================================
# LOAD MODEL TF-IDF
# =====================================================
 
model          = None
responses_map  = {}
model_accuracy = 0
model_intents  = 0
 
try:
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    model          = artifacts['model']
    responses_map  = artifacts['responses_map']
    model_accuracy = artifacts.get('accuracy', 0)
    model_intents  = artifacts.get('n_intents', 0)
    print(f"✅ Model TF-IDF dimuat! Akurasi: {model_accuracy}% | Intents: {model_intents}")
except Exception as e:
    print(f"❌ Gagal load model TF-IDF: {e}")
 
# =====================================================
# HOME
# =====================================================
 
@app.route('/')
def home():
    return render_template('index.html',
        model_accuracy = model_accuracy,
        model_intents  = model_intents,
        model_loaded   = model is not None,
        ollama_ready   = OLLAMA_READY,
        ollama_model   = OLLAMA_MODEL,
    )
 
# =====================================================
# CHATBOT — Hybrid: Ollama utama, TF-IDF fallback
# =====================================================
 
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data tidak valid'}), 400

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'response': 'Pesan tidak boleh kosong.', 'tag': 'umkm', 'confidence': 100})

    try:
        from GROK import GROK
        client = GROK(api_key=os.environ.get('GROK_API_KEY'))
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        bot_response = completion.choices[0].message.content

    except Exception as e:
        print(f"GROK error: {e}")
        # Fallback ke SVM kalau GROK error
        try:
            result = get_response(user_message)
            bot_response = result['response']
        except:
            bot_response = "Maaf, sistem sedang gangguan. Coba lagi!"

    # Simpan ke database
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO chat_logs (user_message, bot_response, confidence) VALUES (?, ?, ?)',
            (user_message, bot_response, 100)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")

    return jsonify({
        'response':   bot_response,
        'tag':        'umkm',
        'confidence': 100,
        'timestamp':  datetime.now().strftime('%H:%M')
    })
 
# =====================================================
# STATUS OLLAMA
# =====================================================
 
@app.route('/ollama-status')
def ollama_status():
    return jsonify({'ready': OLLAMA_READY, 'model': OLLAMA_MODEL})
 
# =====================================================
# SIMPAN DATA BULANAN
# =====================================================
 
@app.route('/simpan-data', methods=['POST'])
def simpan_data():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Data tidak valid'}), 400
    try:
        bulan             = data.get('bulan', '')
        tahun             = int(data.get('tahun', datetime.now().year))
        total_pendapatan  = float(data.get('total_pendapatan', 0))
        hpp               = float(data.get('hpp', 0))
        biaya_operasional = float(data.get('biaya_operasional', 0))
        biaya_pemasaran   = float(data.get('biaya_pemasaran', 0))
        biaya_lain        = float(data.get('biaya_lain', 0))
        catatan           = data.get('catatan', '')
        laba_kotor        = total_pendapatan - hpp
        total_biaya       = biaya_operasional + biaya_pemasaran + biaya_lain
        laba_bersih       = laba_kotor - total_biaya
        margin_profit     = round((laba_bersih / total_pendapatan * 100), 2) if total_pendapatan > 0 else 0
 
        conn   = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM data_bulanan WHERE bulan=? AND tahun=?', (bulan, tahun))
        existing = cursor.fetchone()
        if existing:
            cursor.execute('''
                UPDATE data_bulanan SET total_pendapatan=?, hpp=?, biaya_operasional=?,
                biaya_pemasaran=?, biaya_lain=?, catatan=?, laba_kotor=?, laba_bersih=?, margin_profit=?
                WHERE bulan=? AND tahun=?
            ''', (total_pendapatan, hpp, biaya_operasional, biaya_pemasaran, biaya_lain,
                  catatan, laba_kotor, laba_bersih, margin_profit, bulan, tahun))
            action = 'diperbarui'
        else:
            cursor.execute('''
                INSERT INTO data_bulanan (bulan, tahun, total_pendapatan, hpp, biaya_operasional,
                biaya_pemasaran, biaya_lain, catatan, laba_kotor, laba_bersih, margin_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (bulan, tahun, total_pendapatan, hpp, biaya_operasional, biaya_pemasaran,
                  biaya_lain, catatan, laba_kotor, laba_bersih, margin_profit))
            action = 'disimpan'
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': f'Data {bulan} {tahun} berhasil {action}!',
                        'data': {'laba_kotor': laba_kotor, 'laba_bersih': laba_bersih, 'margin_profit': margin_profit}})
    except Exception as e:
        print(f"Error simpan-data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
 
# =====================================================
# DATA BULANAN, DASHBOARD, CHART
# =====================================================
 
@app.route('/data-bulanan')
def get_data_bulanan():
    try:
        conn   = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT id, bulan, tahun, total_pendapatan, hpp, biaya_operasional,
            biaya_pemasaran, biaya_lain, laba_kotor, laba_bersih, margin_profit, catatan, created_at
            FROM data_bulanan ORDER BY tahun DESC, id DESC LIMIT 12''')
        rows = cursor.fetchall()
        conn.close()
        keys   = ['id','bulan','tahun','total_pendapatan','hpp','biaya_operasional',
                  'biaya_pemasaran','biaya_lain','laba_kotor','laba_bersih','margin_profit','catatan','created_at']
        return jsonify({'success': True, 'data': [dict(zip(keys, r)) for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
@app.route('/dashboard-data')
def dashboard_data():
    try:
        conn   = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT bulan, tahun, total_pendapatan, hpp, biaya_operasional,
            biaya_pemasaran, biaya_lain, laba_kotor, laba_bersih, margin_profit
            FROM data_bulanan ORDER BY tahun DESC, id DESC LIMIT 1''')
        latest = cursor.fetchone()
        cursor.execute('SELECT SUM(total_pendapatan), SUM(laba_bersih) FROM data_bulanan')
        totals = cursor.fetchone()
        cursor.execute('''SELECT bulan, tahun, total_pendapatan, hpp, biaya_operasional, biaya_pemasaran, biaya_lain, laba_kotor, laba_bersih, margin_profit FROM data_bulanan ORDER BY tahun ASC, id ASC LIMIT 12''')
        history = cursor.fetchall()
        conn.close()
        if not latest:
            return jsonify({'success': True, 'empty': True})
        return jsonify({
            'success': True, 'empty': False,
            'latest': {'bulan': latest[0], 'tahun': latest[1], 'total_pendapatan': latest[2],
                       'hpp': latest[3], 'biaya_operasional': latest[4], 'biaya_pemasaran': latest[5],
                       'biaya_lain': latest[6], 'laba_kotor': latest[7], 'laba_bersih': latest[8], 'margin': latest[9]},
            'total_pendapatan': totals[0] or 0,
            'total_laba':       totals[1] or 0,
            'history': [{'bulan': r[0], 'tahun': r[1], 'total_pendapatan': r[2], 'hpp': r[3], 
             'biaya_operasional': r[4], 'biaya_pemasaran': r[5], 'biaya_lain': r[6],
             'laba_kotor': r[7], 'laba_bersih': r[8], 'margin_profit': r[9]} for r in history],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
@app.route('/chart-data')
def chart_data():
    try:
        conn   = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT bulan, total_pendapatan, laba_bersih, hpp FROM data_bulanan ORDER BY tahun ASC, id ASC LIMIT 12')
        rows = cursor.fetchall()
        conn.close()
        return jsonify({'success': True, 'labels': [r[0] for r in rows],
                        'pendapatan': [r[1] for r in rows], 'laba': [r[2] for r in rows], 'hpp': [r[3] for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
# =====================================================
# TRANSAKSI & RIWAYAT
# =====================================================
 
@app.route('/add-transaction', methods=['POST'])
def add_transaction():
    data = request.get_json()
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO transaksi (jenis, kategori, nominal, keterangan) VALUES (?, ?, ?, ?)',
            (data['jenis'], data['kategori'], float(data['nominal']), data.get('keterangan', '')))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Transaksi berhasil ditambahkan'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
 
@app.route('/hapus-data/<int:id>', methods=['DELETE'])
def hapus_data(id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM data_bulanan WHERE id = ?', (id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Data berhasil dihapus'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
@app.route('/chat-history')
def chat_history():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT user_message, bot_response, confidence, source, created_at FROM chat_logs ORDER BY id DESC LIMIT 50')
        rows = cursor.fetchall()
        conn.close()
        return jsonify({'success': True, 'data': [
            {'user': r[0], 'bot': r[1], 'confidence': r[2], 'source': r[3], 'time': r[4]} for r in rows]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
 
# =====================================================
# KALKULATOR
# =====================================================
 
@app.route('/hitung/laba', methods=['POST'])
def hitung_laba():
    data = request.get_json()
    try:
        omzet = float(data.get('omzet', 0)); hpp = float(data.get('hpp', 0))
        ops   = float(data.get('biaya_ops', 0)); mkt = float(data.get('biaya_mkt', 0))
        lain  = float(data.get('biaya_lain', 0)); unit = float(data.get('unit', 1)) or 1
        laba_kotor = omzet - hpp; total_biaya = ops + mkt + lain
        laba_bersih = laba_kotor - total_biaya
        margin = round((laba_bersih / omzet * 100), 2) if omzet > 0 else 0
        if margin >= 30:   status, saran = 'baik',          f'Margin {margin}% sangat baik! Bisnis kamu profitabel.'
        elif margin >= 15: status, saran = 'cukup',         f'Margin {margin}% cukup baik. Coba tekan HPP atau biaya operasional.'
        elif margin >= 0:  status, saran = 'perlu_evaluasi',f'Margin {margin}% terlalu tipis. Audit biaya dan pertimbangkan naikkan harga.'
        else:              status, saran = 'rugi',           'Bisnis merugi! Segera evaluasi strategi pricing.'
        return jsonify({'success': True, 'data': {'laba_kotor': laba_kotor, 'total_biaya': total_biaya,
            'laba_bersih': laba_bersih, 'margin': margin, 'per_unit': round(laba_bersih/unit,0), 'status': status, 'saran': saran}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
 
@app.route('/hitung/bep', methods=['POST'])
def hitung_bep():
    data = request.get_json()
    try:
        bt = float(data.get('biaya_tetap',0)); hj = float(data.get('harga_jual',0)); bv = float(data.get('biaya_variabel',0))
        km = hj - bv
        if km <= 0: return jsonify({'success': True, 'data': {'error': 'Harga jual harus lebih besar dari biaya variabel!'}})
        bu = bt / km; br = bu * hj
        return jsonify({'success': True, 'data': {'kontribusi_margin': round(km,0), 'bep_unit': round(bu,2),
            'bep_rupiah': round(br,0), 'saran': f'Jual minimal {round(bu)} unit (Rp {round(br):,}) untuk balik modal.'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
 
@app.route('/hitung/hpp', methods=['POST'])
def hitung_hpp():
    data = request.get_json()
    try:
        b = float(data.get('bahan_baku',0)); t = float(data.get('tenaga_kerja',0))
        o = float(data.get('overhead',0));   u = float(data.get('unit',1)) or 1
        total = b+t+o; pu = total/u
        return jsonify({'success': True, 'data': {'total_hpp': round(total,0), 'hpp_per_unit': round(pu,0),
            'harga_jual_30': round(pu/0.7,0), 'harga_jual_50': round(pu/0.5,0),
            'saran': f'HPP/unit Rp {round(pu):,}. Margin 30%: Rp {round(pu/0.7):,} | Margin 50%: Rp {round(pu/0.5):,}.'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400
 
# =====================================================
# MAIN
# =====================================================
 
if __name__ == '__main__':
    print("=" * 50)
    print("  UMKM PINTAR SERVER")
    print("=" * 50)
    init_db()
    cek_ollama()
    app.run(debug=True, host='0.0.0.0', port=5000)