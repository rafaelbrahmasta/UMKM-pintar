import sqlite3
from datetime import datetime

DATABASE = 'umkm_pintar.db'

def get_db_connection():
    """Fungsi utama untuk koneksi database"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Agar hasil query jadi dictionary
    return conn

def init_database():
    """Inisialisasi semua tabel"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # USERS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # CHAT LOGS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT,
        bot_response TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # DATA BULANAN (Tabel Utama untuk fitur kamu)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_bulanan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bulan TEXT NOT NULL,
        tahun INTEGER NOT NULL,
        total_pendapatan REAL DEFAULT 0,
        hpp REAL DEFAULT 0,
        biaya_operasional REAL DEFAULT 0,
        biaya_pemasaran REAL DEFAULT 0,
        biaya_lain REAL DEFAULT 0,
        laba_kotor REAL DEFAULT 0,
        laba_bersih REAL DEFAULT 0,
        margin_profit REAL DEFAULT 0,
        catatan TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(bulan, tahun)
    )
    ''')

    # TRANSAKSI (opsional, kalau butuh detail)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transaksi (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data_bulanan_id INTEGER,
        jenis TEXT,
        kategori TEXT,
        nominal REAL,
        keterangan TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (data_bulanan_id) REFERENCES data_bulanan(id)
    )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database & tabel berhasil diinisialisasi!")

# Jalankan inisialisasi saat pertama kali import
if __name__ == "__main__":
    init_database()
else:
    init_database()  # otomatis buat tabel saat diimport