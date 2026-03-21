-- Migration: Move security metadata from config files to database
-- Date: 2026-03-21

-- Step 1: Add new columns
ALTER TABLE security ADD COLUMN IF NOT EXISTS primary_exchange text DEFAULT 'NASDAQ';
ALTER TABLE security ADD COLUMN IF NOT EXISTS trading_times_start time DEFAULT '09:46';
ALTER TABLE security ADD COLUMN IF NOT EXISTS trading_times_end time DEFAULT '15:45';
ALTER TABLE security ADD COLUMN IF NOT EXISTS time2close time DEFAULT '15:53';
ALTER TABLE security ADD COLUMN IF NOT EXISTS board_lot integer DEFAULT 1;
ALTER TABLE security ADD COLUMN IF NOT EXISTS fallback_source text DEFAULT 'yfinance';
ALTER TABLE security ADD COLUMN IF NOT EXISTS fallback_ticker text;
ALTER TABLE security ADD COLUMN IF NOT EXISTS company_name text;
ALTER TABLE security ADD COLUMN IF NOT EXISTS sector text;
ALTER TABLE security ADD COLUMN IF NOT EXISTS beta_info text;
ALTER TABLE security ADD COLUMN IF NOT EXISTS volatility_range text;

-- Step 2: Populate fallback_ticker for securities that already exist
-- Americas (fallback_ticker = code)
UPDATE security SET fallback_ticker = code
WHERE timezone LIKE 'America/%' AND fallback_ticker IS NULL;

-- Europe (fallback_ticker = code + suffix based on market)
UPDATE security SET fallback_ticker = code || '.DE'
WHERE market = 'XETRA' AND fallback_ticker IS NULL;
UPDATE security SET fallback_ticker = code || '.MC'
WHERE market = 'BME' AND fallback_ticker IS NULL;
UPDATE security SET fallback_ticker = code || '.PA'
WHERE market = 'SBF' AND fallback_ticker IS NULL;
UPDATE security SET fallback_ticker = code || '.MI'
WHERE market = 'BVME' AND fallback_ticker IS NULL;
UPDATE security SET fallback_ticker = code || '.L'
WHERE market = 'LSE' AND fallback_ticker IS NULL;
UPDATE security SET fallback_ticker = code || '.AS'
WHERE market = 'AEB' AND fallback_ticker IS NULL;

-- Japan (fallback_ticker = code + '.T')
UPDATE security SET fallback_ticker = code || '.T'
WHERE market = 'TSEJ' AND fallback_ticker IS NULL;

-- Hong Kong (fallback_ticker = zero-padded code + '.HK')
UPDATE security SET fallback_ticker = LPAD(code, 4, '0') || '.HK'
WHERE market = 'SEHK' AND fallback_ticker IS NULL;

-- Step 3: Set primary_exchange per market
UPDATE security SET primary_exchange = 'IBIS' WHERE market = 'XETRA';
UPDATE security SET primary_exchange = 'BM' WHERE market = 'BME';
UPDATE security SET primary_exchange = 'SBF' WHERE market = 'SBF';
UPDATE security SET primary_exchange = 'BVME' WHERE market = 'BVME';
UPDATE security SET primary_exchange = 'LSE' WHERE market = 'LSE';
UPDATE security SET primary_exchange = 'AEB' WHERE market = 'AEB';
UPDATE security SET primary_exchange = 'TSEJ' WHERE market = 'TSEJ';
UPDATE security SET primary_exchange = 'SEHK' WHERE market = 'SEHK';
-- Individual overrides for NYSE-listed NASDAQ stocks
UPDATE security SET primary_exchange = 'NYSE' WHERE code IN ('SHOP', 'SNAP', 'NET', 'ALB');

-- Step 4: Set trading_times and time2close per region
-- Europe (XETRA/BME/SBF/BVME/AEB defaults)
UPDATE security SET trading_times_start = '09:16', trading_times_end = '15:30', time2close = '15:38'
WHERE market IN ('XETRA', 'BME', 'SBF', 'BVME', 'AEB');
-- UK (LSE)
UPDATE security SET trading_times_start = '09:46', trading_times_end = '14:40', time2close = '14:45'
WHERE market = 'LSE';
-- Japan (TSE)
UPDATE security SET trading_times_start = '09:05', trading_times_end = '15:20', time2close = '15:25'
WHERE market = 'TSEJ';
-- Hong Kong (HKEX)
UPDATE security SET trading_times_start = '09:35', trading_times_end = '15:55', time2close = '15:58'
WHERE market = 'SEHK';

-- Step 5: Set board_lot for Asian markets
UPDATE security SET board_lot = 100 WHERE market IN ('TSEJ', 'SEHK');

-- Step 6: Descriptive fields — Americas
UPDATE security SET company_name = 'Tesla', sector = 'EV/energy', beta_info = 'very volatile', volatility_range = '2-4%' WHERE code = 'TSLA';
UPDATE security SET company_name = 'AMD', sector = 'semiconductor', beta_info = 'beta ~1.7', volatility_range = '2-3%' WHERE code = 'AMD';
UPDATE security SET company_name = 'Apple', sector = 'tech megacap', beta_info = 'liquid', volatility_range = '1-2%' WHERE code = 'AAPL';
UPDATE security SET company_name = 'Intel', sector = 'semiconductor', beta_info = 'moderate volatility', volatility_range = '1.5-2.5%' WHERE code = 'INTC';
UPDATE security SET company_name = 'NVIDIA', sector = 'AI/GPU', beta_info = 'very volatile', volatility_range = '2-4%' WHERE code = 'NVDA';
UPDATE security SET company_name = 'SoFi', sector = 'fintech', beta_info = 'beta ~1.8', volatility_range = '3-5%' WHERE code = 'SOFI';
UPDATE security SET company_name = 'MARA Holdings', sector = 'bitcoin mining', beta_info = 'extreme volatility', volatility_range = '4-8%' WHERE code = 'MARA';
UPDATE security SET company_name = 'Rivian', sector = 'EV startup', beta_info = 'volatile', volatility_range = '3-5%' WHERE code = 'RIVN';
UPDATE security SET company_name = 'Robinhood', sector = 'fintech/trading', beta_info = 'volatile', volatility_range = '3-5%' WHERE code = 'HOOD';
UPDATE security SET company_name = 'Super Micro', sector = 'AI servers', beta_info = 'extreme volatility', volatility_range = '4-8%' WHERE code = 'SMCI';
UPDATE security SET company_name = 'DraftKings', sector = 'sports betting', beta_info = 'volatile', volatility_range = '2-4%' WHERE code = 'DKNG';
UPDATE security SET company_name = 'MicroStrategy', sector = 'bitcoin treasury', beta_info = 'extreme volatility', volatility_range = '4-8%' WHERE code = 'MSTR';
UPDATE security SET company_name = 'Amazon', sector = 'e-commerce/cloud', beta_info = 'liquid', volatility_range = '1.5-2.5%' WHERE code = 'AMZN';
UPDATE security SET company_name = 'Microsoft', sector = 'tech megacap', beta_info = 'liquid', volatility_range = '1-2%' WHERE code = 'MSFT';
UPDATE security SET company_name = 'Penn Entertainment', sector = 'sports betting', beta_info = 'beta ~2.0', volatility_range = '3-5%' WHERE code = 'PENN';
UPDATE security SET company_name = 'Affirm', sector = 'fintech/BNPL', beta_info = 'beta ~2.5', volatility_range = '3-5%' WHERE code = 'AFRM';
UPDATE security SET company_name = 'Palantir', sector = 'AI/data analytics', beta_info = 'beta ~2.5', volatility_range = '3-5%' WHERE code = 'PLTR';
UPDATE security SET company_name = 'Shopify', sector = 'e-commerce platform', beta_info = 'beta ~2.0', volatility_range = '2-4%' WHERE code = 'SHOP';
UPDATE security SET company_name = 'Roblox', sector = 'gaming/metaverse', beta_info = 'beta ~2.0', volatility_range = '2-4%' WHERE code = 'RBLX';
UPDATE security SET company_name = 'CrowdStrike', sector = 'cybersecurity', beta_info = 'beta ~1.5', volatility_range = '2-3%' WHERE code = 'CRWD';
UPDATE security SET company_name = 'Snap', sector = 'social media/AR', beta_info = 'beta ~1.5', volatility_range = '2-4%' WHERE code = 'SNAP';
UPDATE security SET company_name = 'Roku', sector = 'streaming tech', beta_info = 'beta ~2.0', volatility_range = '3-5%' WHERE code = 'ROKU';
UPDATE security SET company_name = 'Enphase Energy', sector = 'solar/clean energy', beta_info = 'beta ~1.8', volatility_range = '3-5%' WHERE code = 'ENPH';
UPDATE security SET company_name = 'Cloudflare', sector = 'cloud/cybersecurity', beta_info = 'beta ~1.5', volatility_range = '2-4%' WHERE code = 'NET';
UPDATE security SET company_name = 'Moderna', sector = 'biotech/vaccines', beta_info = 'beta ~1.8', volatility_range = '3-5%' WHERE code = 'MRNA';
UPDATE security SET company_name = 'First Solar', sector = 'solar manufacturing', beta_info = 'beta ~1.5', volatility_range = '2-4%' WHERE code = 'FSLR';
UPDATE security SET company_name = 'Albemarle', sector = 'lithium/chemicals', beta_info = 'beta ~1.5', volatility_range = '2-4%' WHERE code = 'ALB';
UPDATE security SET company_name = 'DoorDash', sector = 'delivery platform', beta_info = 'beta ~1.3', volatility_range = '2-3%' WHERE code = 'DASH';

-- Step 7: Descriptive fields — Europe
UPDATE security SET company_name = 'Stabilus', sector = 'industrial', beta_info = 'moderate volatility', volatility_range = '1.5-2.5%' WHERE code = 'SBX';
UPDATE security SET company_name = 'Infineon', sector = 'semiconductor', beta_info = 'beta 1.83', volatility_range = '2-3%' WHERE code = 'IFX';
UPDATE security SET company_name = 'Deutsche Bank', sector = 'banking', beta_info = 'beta 1.46', volatility_range = '1.5-2.5%' WHERE code = 'DBK';
UPDATE security SET company_name = 'Siemens Energy', sector = 'energy', beta_info = 'beta 1.60-1.81', volatility_range = '2-3%' WHERE code = 'ENR';
UPDATE security SET company_name = 'BBVA', sector = 'banking', beta_info = 'beta 1.25', volatility_range = '1.5-2.5%' WHERE code = 'BBVA';
UPDATE security SET company_name = 'Santander', sector = 'banking', beta_info = 'beta 1.20', volatility_range = '1.5-2.5%' WHERE code = 'SAN';
UPDATE security SET company_name = 'Societe Generale', sector = 'banking', beta_info = 'beta 1.39', volatility_range = '2-3%' WHERE code = 'GLE';
UPDATE security SET company_name = 'STMicro', sector = 'semiconductor', beta_info = 'beta 1.22', volatility_range = '2-3%' WHERE code = 'STMPA';
UPDATE security SET company_name = 'UniCredit', sector = 'banking', beta_info = 'beta 1.28', volatility_range = '2-3%' WHERE code = 'UCG';
UPDATE security SET company_name = 'Barclays', sector = 'banking', beta_info = 'beta 1.98', volatility_range = '2-3%' WHERE code = 'BARC';
UPDATE security SET company_name = 'ASML Holding', sector = 'semiconductor equip', beta_info = 'beta ~1.3', volatility_range = '2-3%' WHERE code = 'ASML';
UPDATE security SET company_name = 'BNP Paribas', sector = 'banking', beta_info = 'beta ~1.3', volatility_range = '1.5-2.5%' WHERE code = 'BNP';
UPDATE security SET company_name = 'Commerzbank', sector = 'banking', beta_info = 'beta ~1.4', volatility_range = '2-3%' WHERE code = 'CBK';
UPDATE security SET company_name = 'Flutter Entertainment', sector = 'sports betting', beta_info = 'beta ~1.3', volatility_range = '2-3%' WHERE code = 'FLTR';
UPDATE security SET company_name = 'Airbus', sector = 'aerospace/defense', beta_info = 'beta ~1.3', volatility_range = '1.5-2.5%' WHERE code = 'AIR';
UPDATE security SET company_name = 'Adidas', sector = 'sportswear', beta_info = 'beta ~1.3', volatility_range = '2-3%' WHERE code = 'ADS';
UPDATE security SET company_name = 'Renault', sector = 'automotive', beta_info = 'beta ~1.5', volatility_range = '2-3%' WHERE code = 'RNO';
UPDATE security SET company_name = 'ThyssenKrupp', sector = 'industrial/steel', beta_info = 'beta ~1.6', volatility_range = '2-4%' WHERE code = 'TKA';
UPDATE security SET company_name = 'Volkswagen', sector = 'automotive', beta_info = 'beta ~1.3', volatility_range = '2-3%' WHERE code = 'VOW3';
UPDATE security SET company_name = 'SAP SE', sector = 'enterprise software', beta_info = 'beta ~1.1', volatility_range = '1.5-2.5%' WHERE code = 'SAP';
UPDATE security SET company_name = 'BASF', sector = 'chemicals', beta_info = 'beta ~1.2', volatility_range = '1.5-2.5%' WHERE code = 'BAS';
UPDATE security SET company_name = 'BMW', sector = 'automotive', beta_info = 'beta ~1.3', volatility_range = '1.5-2.5%' WHERE code = 'BMW';
UPDATE security SET company_name = 'TotalEnergies', sector = 'energy', beta_info = 'beta ~1.2', volatility_range = '1.5-2.5%' WHERE code = 'TTE';
UPDATE security SET company_name = 'DHL Group', sector = 'logistics', beta_info = 'beta ~1.2', volatility_range = '1.5-2.5%' WHERE code = 'DHL';

-- Step 8: Descriptive fields — Japan
UPDATE security SET company_name = 'Nissan Motor', sector = 'automotive', volatility_range = '2.5%' WHERE code = '7201';
UPDATE security SET company_name = 'Metaplanet', sector = 'Bitcoin proxy', volatility_range = '8.1%' WHERE code = '3350';
UPDATE security SET company_name = 'NTT', sector = 'telecom', volatility_range = '2.0%' WHERE code = '9432';
UPDATE security SET company_name = 'Rakuten Group', sector = 'e-commerce/fintech', volatility_range = '2.0%' WHERE code = '4755';
UPDATE security SET company_name = 'Nippon Steel', sector = 'steel', volatility_range = '2.0%' WHERE code = '5401';
UPDATE security SET company_name = 'SoftBank Corp', sector = 'telecom', volatility_range = '1.5%' WHERE code = '9434';
UPDATE security SET company_name = 'Mazda Motor', sector = 'automotive', volatility_range = '3.0%' WHERE code = '7261';

-- Step 9: Descriptive fields — Hong Kong
UPDATE security SET company_name = 'NIO Inc', sector = 'EV', volatility_range = '4-5%' WHERE code = '9866';
UPDATE security SET company_name = 'Kuaishou', sector = 'social/video', volatility_range = '3%' WHERE code = '1024';
UPDATE security SET company_name = 'Li Auto', sector = 'EV', volatility_range = '3-4%' WHERE code = '2015';
