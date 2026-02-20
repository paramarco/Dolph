-- Migration: Add multi-market support columns to security table
-- Run manually: psql -h 127.0.0.1 -p 4713 -U dolph_user -d dolph_db -f migrate_multi_market.sql

ALTER TABLE security ADD COLUMN IF NOT EXISTS timezone text DEFAULT 'America/New_York';
ALTER TABLE security ADD COLUMN IF NOT EXISTS currency text DEFAULT 'USD';
ALTER TABLE security ADD COLUMN IF NOT EXISTS exchange text DEFAULT 'SMART';
