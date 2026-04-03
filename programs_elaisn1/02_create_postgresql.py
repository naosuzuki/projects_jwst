#!/usr/bin/env python3
"""
Create PostgreSQL database for ELAIS-N1 spectroscopic redshifts.

Creates:
  - Database: elaisn1
  - Table: specz  (main spectroscopic redshift table)
  - Indexes on (ra, dec), z_spec, survey
  - Cone search function

Imports data from elaisn1_specz_database.csv

Usage:
  python 02_create_postgresql.py

Requires: psycopg2
  pip install psycopg2-binary
"""

import os
import csv
import psycopg2
from psycopg2 import sql

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_FILE = os.path.join(DATADIR, 'elaisn1_specz_database.csv')

DB_NAME = 'elaisn1'
DB_USER = 'root'
DB_HOST = '/var/run/postgresql'


def create_database():
    conn = psycopg2.connect(dbname='postgres', user=DB_USER, host=DB_HOST)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    if not cur.fetchone():
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        print(f"Created database: {DB_NAME}")
    else:
        print(f"Database {DB_NAME} already exists")
    cur.close()
    conn.close()


def create_table():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS specz CASCADE;")
    cur.execute("""
        CREATE TABLE specz (
            id          SERIAL PRIMARY KEY,
            ra          DOUBLE PRECISION NOT NULL,
            dec         DOUBLE PRECISION NOT NULL,
            z_spec      DOUBLE PRECISION NOT NULL,
            z_err       DOUBLE PRECISION,
            obj_class   VARCHAR(64),
            survey      VARCHAR(64) NOT NULL,
            reference   VARCHAR(256) NOT NULL,
            CONSTRAINT z_positive CHECK (z_spec > 0 AND z_spec < 10),
            CONSTRAINT ra_range CHECK (ra >= 0 AND ra < 360),
            CONSTRAINT dec_range CHECK (dec >= -90 AND dec <= 90)
        );
        COMMENT ON TABLE specz IS
            'Spectroscopic redshifts in the ELAIS-N1 field (RA~242.75, Dec~+55.0)';
        COMMENT ON COLUMN specz.ra IS 'Right Ascension J2000 (degrees)';
        COMMENT ON COLUMN specz.dec IS 'Declination J2000 (degrees)';
        COMMENT ON COLUMN specz.z_spec IS 'Spectroscopic redshift';
        COMMENT ON COLUMN specz.z_err IS 'Redshift uncertainty';
        COMMENT ON COLUMN specz.obj_class IS 'Object classification (GALAXY, QSO, STAR, etc.)';
        COMMENT ON COLUMN specz.survey IS 'Source survey name';
        COMMENT ON COLUMN specz.reference IS 'Literature reference';
    """)
    conn.commit()
    print("Created table: specz")
    cur.close()
    conn.close()


def import_csv():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST)
    cur = conn.cursor()
    count = 0
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            z_err = row['z_err']
            if z_err in ('', 'nan', 'NaN'):
                z_err = None
            else:
                z_err = float(z_err)
            obj_class = row.get('obj_class', '')
            if obj_class in ('nan', 'NaN', ''):
                obj_class = None
            cur.execute("""
                INSERT INTO specz (ra, dec, z_spec, z_err, obj_class, survey, reference)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                float(row['ra']), float(row['dec']), float(row['z_spec']),
                z_err, obj_class, row['survey'], row['reference'],
            ))
            count += 1
    conn.commit()
    print(f"Imported {count} rows from CSV")
    cur.close()
    conn.close()


def create_indexes():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST)
    cur = conn.cursor()
    for name, ddl in [
        ("idx_specz_radec", "CREATE INDEX idx_specz_radec ON specz (ra, dec);"),
        ("idx_specz_z", "CREATE INDEX idx_specz_z ON specz (z_spec);"),
        ("idx_specz_survey", "CREATE INDEX idx_specz_survey ON specz (survey);"),
    ]:
        cur.execute(ddl)
        print(f"  Created index: {name}")

    # Cone search function
    cur.execute("""
        CREATE OR REPLACE FUNCTION cone_search(
            ra_center DOUBLE PRECISION,
            dec_center DOUBLE PRECISION,
            radius_arcsec DOUBLE PRECISION
        ) RETURNS SETOF specz AS $$
        DECLARE
            radius_deg DOUBLE PRECISION := radius_arcsec / 3600.0;
            cos_dec DOUBLE PRECISION := COS(RADIANS(dec_center));
        BEGIN
            RETURN QUERY
            SELECT * FROM specz
            WHERE dec BETWEEN dec_center - radius_deg AND dec_center + radius_deg
              AND ra  BETWEEN ra_center - radius_deg/cos_dec AND ra_center + radius_deg/cos_dec
              AND DEGREES(ACOS(
                    SIN(RADIANS(dec)) * SIN(RADIANS(dec_center)) +
                    COS(RADIANS(dec)) * COS(RADIANS(dec_center)) *
                    COS(RADIANS(ra - ra_center))
                  )) * 3600.0 <= radius_arcsec;
        END;
        $$ LANGUAGE plpgsql;
        COMMENT ON FUNCTION cone_search IS
            'Cone search: returns sources within radius_arcsec of (ra_center, dec_center)';
    """)
    conn.commit()
    print("  Created function: cone_search(ra, dec, radius_arcsec)")
    cur.close()
    conn.close()


def verify():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST)
    cur = conn.cursor()

    print("\n" + "=" * 60)
    print("POSTGRESQL DATABASE: elaisn1")
    print("=" * 60)

    cur.execute("SELECT COUNT(*) FROM specz;")
    print(f"Total rows: {cur.fetchone()[0]}")

    cur.execute("""SELECT MIN(z_spec), MAX(z_spec), AVG(z_spec),
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY z_spec) FROM specz;""")
    zmin, zmax, zavg, zmed = cur.fetchone()
    print(f"Redshift: min={zmin:.4f}, max={zmax:.4f}, mean={zavg:.4f}, median={zmed:.4f}")

    cur.execute("SELECT MIN(ra), MAX(ra), MIN(dec), MAX(dec) FROM specz;")
    ramin, ramax, dmin, dmax = cur.fetchone()
    print(f"RA range:  [{ramin:.4f}, {ramax:.4f}]")
    print(f"Dec range: [{dmin:.4f}, {dmax:.4f}]")

    cur.execute("""SELECT survey, COUNT(*), MIN(z_spec), MAX(z_spec)
                   FROM specz GROUP BY survey ORDER BY COUNT(*) DESC;""")
    print("\nBy survey:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} sources (z = {row[2]:.4f} - {row[3]:.4f})")

    cur.execute("SELECT COUNT(*) FROM cone_search(242.75, 55.0, 60.0);")
    print(f"\nCone search test (60\" around field center): {cur.fetchone()[0]} sources")

    print("\nExample queries:")
    print("  SELECT * FROM specz WHERE z_spec BETWEEN 0.1 AND 0.3 LIMIT 10;")
    print("  SELECT * FROM cone_search(242.75, 55.0, 30.0);")
    print(f"\nConnect: psql -d {DB_NAME}")

    cur.close()
    conn.close()


def export_sql_dump():
    dump_path = os.path.join(DATADIR, 'elaisn1_specz.sql')
    ret = os.system(f'pg_dump -U {DB_USER} -h {DB_HOST} --no-owner --no-acl {DB_NAME} > {dump_path}')
    if ret == 0:
        size = os.path.getsize(dump_path)
        print(f"\nSQL dump: {dump_path} ({size/1024:.0f} KB)")
        print(f"  Restore: createdb elaisn1 && psql -d elaisn1 < elaisn1_specz.sql")
    else:
        print(f"\nSQL dump failed (exit {ret})")


if __name__ == '__main__':
    print("=" * 60)
    print("POSTGRESQL SETUP FOR ELAIS-N1 SPEC-Z DATABASE")
    print("=" * 60)
    create_database()
    create_table()
    import_csv()
    create_indexes()
    verify()
    export_sql_dump()
    print("\nDone.")
