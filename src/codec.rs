/// Phase 18: Variable-bit ID / Region Encoding.
///
/// This module provides compact binary representations for IDs:
///
/// **LEB128 (unsigned)**: encodes a u64 into 1-10 bytes depending on its
/// magnitude.  Small IDs (< 128) use 1 byte; IDs < 16384 use 2 bytes.
/// This is exact — no information is lost.
///
/// **Delta encoding for UnitId sequences**: when serialising a sorted or
/// locally-adjacent sequence of UnitIds, store differences between
/// consecutive values rather than absolute values.  The first element is
/// stored as-is; later elements store `current − previous`.  For IDs that
/// are clustered (typical in segmented text), most deltas fit in 1-2 bytes.
///
/// These encoders/decoders are NOT yet connected to the binary persistence
/// layer — the snapshot still uses bincode with fixed-size integers.
/// Integration is tracked as a P5 item (see CURRENT_STATE.md §58 note).
use crate::units::UnitId;

// ── LEB128 ────────────────────────────────────────────────────────────────

/// Encode `value` as unsigned LEB128 into `buf`.  Returns bytes written.
pub fn leb128_encode(mut value: u64, buf: &mut Vec<u8>) -> usize {
    let start = buf.len();
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte); // final byte: high bit clear
            break;
        } else {
            buf.push(byte | 0x80); // more bytes follow: high bit set
        }
    }
    buf.len() - start
}

/// Decode one unsigned LEB128 value from `bytes` starting at `pos`.
/// Returns `(value, new_pos)` or `None` on truncation.
pub fn leb128_decode(bytes: &[u8], pos: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut cur = pos;
    loop {
        let byte = *bytes.get(cur)?;
        cur += 1;
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((value, cur));
        }
        shift += 7;
        if shift >= 64 {
            return None; // overflow
        }
    }
}

// ── UnitId packing ────────────────────────────────────────────────────────

/// Pack a `UnitId` into a single u64 for encoding:
///   bit 0 = is_chunk flag (1 = chunk, 0 = primitive)
///   bits 1..63 = raw id
pub fn unit_id_to_u64(u: UnitId) -> u64 {
    ((u.raw() as u64) << 1) | (u.is_chunk() as u64)
}

/// Unpack a `UnitId` from a packed u64.
pub fn unit_id_from_u64(v: u64) -> UnitId {
    let is_chunk = (v & 1) != 0;
    let raw = (v >> 1) as u32;
    if is_chunk { UnitId::chunk(raw) } else { UnitId::primitive(raw) }
}

/// Encode a single UnitId as LEB128.
pub fn encode_unit(u: UnitId, buf: &mut Vec<u8>) -> usize {
    leb128_encode(unit_id_to_u64(u), buf)
}

/// Decode a single UnitId from LEB128.
pub fn decode_unit(bytes: &[u8], pos: usize) -> Option<(UnitId, usize)> {
    let (v, next) = leb128_decode(bytes, pos)?;
    Some((unit_id_from_u64(v), next))
}

// ── Delta encoding for UnitId sequences ──────────────────────────────────

/// Encode a sequence of UnitIds using delta + LEB128.
///
/// Each element is stored as `current_packed − previous_packed`, using
/// LEB128 on the packed representation.  The first element is stored as-is
/// (delta from 0).  For adjacent IDs, most deltas are < 4, fitting in 1 byte.
pub fn encode_units_delta(units: &[UnitId], buf: &mut Vec<u8>) {
    let mut prev: u64 = 0;
    for &u in units {
        let packed = unit_id_to_u64(u);
        // Use i64 delta stored as zigzag-encoded u64 for negative deltas.
        let delta = zigzag_encode(packed as i64 - prev as i64);
        leb128_encode(delta, buf);
        prev = packed;
    }
}

/// Decode a delta-encoded sequence of `count` UnitIds from `bytes[pos..]`.
/// Returns `(units, new_pos)` or `None` on truncation / overflow.
pub fn decode_units_delta(bytes: &[u8], pos: usize, count: usize) -> Option<(Vec<UnitId>, usize)> {
    let mut units = Vec::with_capacity(count);
    let mut cur = pos;
    let mut prev: u64 = 0;
    for _ in 0..count {
        let (delta_zz, next) = leb128_decode(bytes, cur)?;
        cur = next;
        let delta = zigzag_decode(delta_zz);
        let packed = (prev as i64 + delta) as u64;
        units.push(unit_id_from_u64(packed));
        prev = packed;
    }
    Some((units, cur))
}

// ── Zigzag encoding (maps i64 → u64 for LEB128 of signed deltas) ─────────

#[inline]
pub fn zigzag_encode(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

#[inline]
pub fn zigzag_decode(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::UnitId;

    fn p(id: u32) -> UnitId { UnitId::primitive(id) }
    fn c(id: u32) -> UnitId { UnitId::chunk(id) }

    // ── LEB128 round-trips ────────────────────────────────────────────────

    #[test]
    fn vb01_leb128_single_byte() {
        let mut buf = Vec::new();
        leb128_encode(42, &mut buf);
        assert_eq!(buf, [42]);
        let (v, pos) = leb128_decode(&buf, 0).unwrap();
        assert_eq!(v, 42);
        assert_eq!(pos, 1);
    }

    #[test]
    fn vb02_leb128_multi_byte() {
        let mut buf = Vec::new();
        leb128_encode(300, &mut buf);
        assert_eq!(buf.len(), 2);
        let (v, _) = leb128_decode(&buf, 0).unwrap();
        assert_eq!(v, 300);
    }

    #[test]
    fn vb03_leb128_large_value() {
        let mut buf = Vec::new();
        leb128_encode(u64::MAX, &mut buf);
        let (v, _) = leb128_decode(&buf, 0).unwrap();
        assert_eq!(v, u64::MAX);
    }

    #[test]
    fn vb04_leb128_zero() {
        let mut buf = Vec::new();
        leb128_encode(0, &mut buf);
        assert_eq!(buf, [0]);
        let (v, _) = leb128_decode(&buf, 0).unwrap();
        assert_eq!(v, 0);
    }

    // ── UnitId round-trips ────────────────────────────────────────────────

    #[test]
    fn vb05_encode_decode_primitive() {
        let mut buf = Vec::new();
        encode_unit(p(7), &mut buf);
        let (u, _) = decode_unit(&buf, 0).unwrap();
        assert_eq!(u, p(7));
    }

    #[test]
    fn vb06_encode_decode_chunk() {
        let mut buf = Vec::new();
        encode_unit(c(42), &mut buf);
        let (u, _) = decode_unit(&buf, 0).unwrap();
        assert_eq!(u, c(42));
    }

    #[test]
    fn vb07_primitive_smaller_than_chunk_same_raw() {
        // raw=7: primitive packed = 14 (1 byte), chunk packed = 15 (1 byte).
        let mut pb = Vec::new(); let mut cb = Vec::new();
        encode_unit(p(7), &mut pb);
        encode_unit(c(7), &mut cb);
        assert_eq!(pb.len(), 1);
        assert_eq!(cb.len(), 1);
    }

    // ── Delta encoding round-trips ────────────────────────────────────────

    #[test]
    fn vb08_delta_encode_decode_sorted() {
        let units = vec![p(1), p(2), p(3), p(4), p(5)];
        let mut buf = Vec::new();
        encode_units_delta(&units, &mut buf);
        let (decoded, _) = decode_units_delta(&buf, 0, units.len()).unwrap();
        assert_eq!(decoded, units);
    }

    #[test]
    fn vb09_delta_encode_decode_mixed() {
        let units = vec![p(10), c(3), p(7), c(100)];
        let mut buf = Vec::new();
        encode_units_delta(&units, &mut buf);
        let (decoded, _) = decode_units_delta(&buf, 0, units.len()).unwrap();
        assert_eq!(decoded, units);
    }

    #[test]
    fn vb10_delta_compact_for_adjacent() {
        // Adjacent IDs: most deltas = 1, should fit in 1 byte each.
        let units: Vec<UnitId> = (0..10).map(p).collect();
        let mut delta_buf = Vec::new();
        let mut flat_buf  = Vec::new();
        encode_units_delta(&units, &mut delta_buf);
        for &u in &units { encode_unit(u, &mut flat_buf); }
        // Delta should be <= flat (equal or smaller).
        assert!(delta_buf.len() <= flat_buf.len(),
            "delta ({}) should be <= flat ({})", delta_buf.len(), flat_buf.len());
    }

    #[test]
    fn vb11_zigzag_roundtrip() {
        for n in [-100i64, -1, 0, 1, 100, i64::MIN/2, i64::MAX/2] {
            assert_eq!(zigzag_decode(zigzag_encode(n)), n);
        }
    }
}
