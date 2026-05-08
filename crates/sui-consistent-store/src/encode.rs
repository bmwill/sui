// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Encoding traits for keys and values stored in the database.
//!
//! Encoding is a property of the Rust type, not of the column family.
//! Schema authors implement [`Encode`] and [`Decode`] on bespoke wrapper
//! types and pin the on-disk byte representation at the point each new
//! type is introduced. Two motivations:
//!
//! - Migration. Introducing a new wrapper type alongside an old one
//!   (with a different on-disk representation) is a non-disruptive way
//!   to evolve a schema. Dual-write the two types, then cut over.
//! - Explicit choice. The on-disk representation is a decision; binding
//!   the encoding to the type forces the author to make it deliberately.
//!
//! # The append contract
//!
//! [`Encode::encode_into`] **appends** to the supplied buffer rather
//! than overwriting it. This lets call sites encode multiple values
//! sequentially into a single buffer (recording offsets) and pass
//! non-overlapping subslices to functions that need several byte
//! strings live at once. Implementations that clear or truncate the
//! buffer break this contract.
//!
//! # Owned vs. borrowed values
//!
//! Only owned decode is supported in this version of the crate. A
//! borrowed-decode trait was scoped out and may return when a real
//! schema needs zero-copy value views; the byte path is already
//! reachable today via `DbMap::get_raw` (arriving in a later commit).

use crate::error::DecodeError;
use crate::error::EncodeError;

/// Encode a value into bytes.
///
/// Implementations append to the supplied buffer. Call sites may pass
/// a non-empty buffer (for example, when encoding several values into
/// the same allocation), and any prefix already present must be
/// preserved.
///
/// # Examples
///
/// ```
/// use sui_consistent_store::Encode;
/// use sui_consistent_store::error::EncodeError;
///
/// struct U64BeKey(u64);
///
/// impl Encode for U64BeKey {
///     fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
///         buf.extend_from_slice(&self.0.to_be_bytes());
///         Ok(())
///     }
/// }
///
/// let key = U64BeKey(42);
/// assert_eq!(key.encode().unwrap(), [0, 0, 0, 0, 0, 0, 0, 42]);
/// ```
pub trait Encode {
    /// Append the encoded form of `self` to `buf`.
    ///
    /// Implementations must not clear or truncate `buf`. They may grow
    /// it and append; any bytes already present must remain unchanged.
    fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError>;

    /// Encode `self` into a freshly allocated `Vec<u8>`.
    ///
    /// A convenience wrapper over [`encode_into`](Self::encode_into).
    /// Internal call sites in this crate prefer the in-place form so
    /// they can amortize buffer allocations across many encodes.
    fn encode(&self) -> Result<Vec<u8>, EncodeError> {
        let mut buf = Vec::new();
        self.encode_into(&mut buf)?;
        Ok(buf)
    }
}

/// Decode a value from bytes.
///
/// Implementations consume the entire byte slice. Trailing bytes that
/// the decoder does not recognize are an error; callers that want to
/// decode a prefix should not use this trait directly.
///
/// # Examples
///
/// ```
/// use sui_consistent_store::Decode;
/// use sui_consistent_store::Encode;
/// use sui_consistent_store::error::DecodeError;
/// use sui_consistent_store::error::EncodeError;
///
/// #[derive(Debug, PartialEq, Eq)]
/// struct U64BeKey(u64);
///
/// impl Encode for U64BeKey {
///     fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
///         buf.extend_from_slice(&self.0.to_be_bytes());
///         Ok(())
///     }
/// }
///
/// impl Decode for U64BeKey {
///     fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
///         let arr: [u8; 8] = bytes.try_into().map_err(|_| {
///             DecodeError::msg(format!(
///                 "expected 8 bytes for U64BeKey, got {}",
///                 bytes.len(),
///             ))
///         })?;
///         Ok(U64BeKey(u64::from_be_bytes(arr)))
///     }
/// }
///
/// let bytes = U64BeKey(42).encode().unwrap();
/// assert_eq!(U64BeKey::decode(&bytes).unwrap(), U64BeKey(42));
/// ```
pub trait Decode: Sized {
    /// Decode `bytes` into a value of `Self`.
    fn decode(bytes: &[u8]) -> Result<Self, DecodeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rolled big-endian `u64` for tests. Big-endian gives a
    /// prefix-preserving lexicographic ordering when used as a key.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct U64BeKey(u64);

    impl Encode for U64BeKey {
        fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), EncodeError> {
            buf.extend_from_slice(&self.0.to_be_bytes());
            Ok(())
        }
    }

    impl Decode for U64BeKey {
        fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                DecodeError::msg(format!(
                    "expected 8 bytes for U64BeKey, got {}",
                    bytes.len(),
                ))
            })?;
            Ok(U64BeKey(u64::from_be_bytes(arr)))
        }
    }

    #[test]
    fn round_trip() {
        let original = U64BeKey(0x0123_4567_89AB_CDEF);
        let bytes = original.encode().unwrap();
        let decoded = U64BeKey::decode(&bytes).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn encode_default_method_matches_encode_into() {
        let value = U64BeKey(42);
        let mut into_buf = Vec::new();
        value.encode_into(&mut into_buf).unwrap();
        assert_eq!(value.encode().unwrap(), into_buf);
    }

    #[test]
    fn encode_into_appends() {
        let mut buf = vec![0xAA, 0xBB, 0xCC];
        U64BeKey(1).encode_into(&mut buf).unwrap();
        assert_eq!(buf[..3], [0xAA, 0xBB, 0xCC]);
        assert_eq!(buf[3..], [0, 0, 0, 0, 0, 0, 0, 1]);
    }

    #[test]
    fn two_values_share_one_buffer() {
        // Demonstrates the call-site pattern used by `Batch::put` once
        // it lands: encode key, record offset, encode value, slice.
        let mut buf = Vec::new();
        U64BeKey(1).encode_into(&mut buf).unwrap();
        let k_end = buf.len();
        U64BeKey(2).encode_into(&mut buf).unwrap();

        let bytes = buf.as_slice();
        let k_slice = &bytes[..k_end];
        let v_slice = &bytes[k_end..];

        assert_eq!(U64BeKey::decode(k_slice).unwrap(), U64BeKey(1));
        assert_eq!(U64BeKey::decode(v_slice).unwrap(), U64BeKey(2));
    }

    #[test]
    fn decode_short_bytes_errors() {
        let err = U64BeKey::decode(&[0, 0, 0, 0]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("got 4"), "unexpected message: {msg}");
    }

    #[test]
    fn decode_long_bytes_errors() {
        let err = U64BeKey::decode(&[0; 9]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("got 9"), "unexpected message: {msg}");
    }
}
