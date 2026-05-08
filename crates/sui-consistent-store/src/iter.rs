// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Typed iteration over a column family.
//!
//! Two iterator types are exposed: [`Iter`] for forward iteration in
//! lexicographic key order, and [`RevIter`] for reverse iteration.
//! Both implement [`Iterator`] yielding
//! `Result<(K, V), Error>` so that decode failures are reported as
//! per-item errors rather than poisoning the entire scan; the
//! iterator stops yielding after the first error.
//!
//! Construct an iterator via [`DbMap::iter`](crate::DbMap::iter),
//! [`iter_rev`](crate::DbMap::iter_rev),
//! [`iter_prefix`](crate::DbMap::iter_prefix), or
//! [`iter_rev_prefix`](crate::DbMap::iter_rev_prefix).
//!
//! # Prefix iteration
//!
//! `iter_prefix` and `iter_rev_prefix` accept a separate prefix type
//! whose [`Encode`] form is treated as a byte prefix of the full
//! key's encoded form. The crate does *not* statically verify that
//! the prefix encoding really is a prefix of the key encoding; that
//! is a property of the schema's chosen encoding scheme. In practice,
//! schemas that encode compound keys with a prefix-preserving
//! representation (for example, big-endian fixed-int tuples) and
//! pass tuple prefixes get the right behavior.
//!
//! Internally, prefix iteration sets RocksDB's
//! `iterate_lower_bound` and `iterate_upper_bound` to the encoded
//! prefix and its lexicographic successor, so the underlying scan
//! stops cleanly at the end of the prefix range without any
//! per-item filtering on the Rust side.

use std::marker::PhantomData;

use rocksdb::DBRawIterator;

use crate::Decode;
use crate::error::Error;

/// Forward iterator over a column family in lexicographic key order.
///
/// Constructed by [`DbMap::iter`](crate::DbMap::iter) or
/// [`DbMap::iter_prefix`](crate::DbMap::iter_prefix).
pub struct Iter<'d, K, V> {
    inner: Option<DBRawIterator<'d>>,
    _phantom: PhantomData<fn() -> (K, V)>,
}

/// Reverse iterator over a column family in reverse lexicographic
/// key order.
///
/// Constructed by [`DbMap::iter_rev`](crate::DbMap::iter_rev) or
/// [`DbMap::iter_rev_prefix`](crate::DbMap::iter_rev_prefix).
pub struct RevIter<'d, K, V> {
    inner: Option<DBRawIterator<'d>>,
    _phantom: PhantomData<fn() -> (K, V)>,
}

impl<'d, K, V> Iter<'d, K, V> {
    pub(crate) fn new(inner: DBRawIterator<'d>) -> Self {
        Self {
            inner: Some(inner),
            _phantom: PhantomData,
        }
    }
}

impl<'d, K, V> RevIter<'d, K, V> {
    pub(crate) fn new(inner: DBRawIterator<'d>) -> Self {
        Self {
            inner: Some(inner),
            _phantom: PhantomData,
        }
    }
}

impl<K, V> Iterator for Iter<'_, K, V>
where
    K: Decode,
    V: Decode,
{
    type Item = Result<(K, V), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        next_step(&mut self.inner, true)
    }
}

impl<K, V> Iterator for RevIter<'_, K, V>
where
    K: Decode,
    V: Decode,
{
    type Item = Result<(K, V), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        next_step(&mut self.inner, false)
    }
}

/// Compute the lexicographic successor of `prefix`, treating it as a
/// variable-length byte string.
///
/// Returns `Some(succ)` such that `succ` is the smallest byte string
/// strictly greater than `prefix` in lexicographic order. Returns
/// `None` if `prefix` is composed entirely of `0xFF` bytes (and so
/// has no successor in this ordering), in which case the prefix
/// extends to the end of the key space.
pub(crate) fn next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut next = prefix.to_vec();
    while let Some(last) = next.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(next);
        }
        next.pop();
    }
    None
}

/// Shared `next` body for both directions. `forward = true` advances
/// via `DBRawIterator::next` after yielding; `forward = false`
/// advances via `prev`.
fn next_step<K, V>(
    slot: &mut Option<DBRawIterator<'_>>,
    forward: bool,
) -> Option<Result<(K, V), Error>>
where
    K: Decode,
    V: Decode,
{
    let inner = slot.as_mut()?;
    if !inner.valid() {
        // Either the iterator is exhausted or it errored. `status`
        // distinguishes the two.
        let err = inner.status().err();
        *slot = None;
        return err.map(|e| Err(Error::Rocksdb(e)));
    }

    let key_bytes = inner.key()?;
    let value_bytes = inner.value()?;

    let item = match (K::decode(key_bytes), V::decode(value_bytes)) {
        (Ok(k), Ok(v)) => Ok((k, v)),
        (Err(e), _) | (_, Err(e)) => {
            *slot = None;
            return Some(Err(Error::Decode(e)));
        }
    };

    if forward {
        inner.next();
    } else {
        inner.prev();
    }
    Some(item)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_prefix_increments_last_byte() {
        assert_eq!(next_prefix(&[1, 2, 3]), Some(vec![1, 2, 4]));
    }

    #[test]
    fn next_prefix_carries_when_last_is_0xff() {
        assert_eq!(next_prefix(&[1, 2, 0xFF]), Some(vec![1, 3]));
    }

    #[test]
    fn next_prefix_carries_through_multiple_0xff_bytes() {
        assert_eq!(next_prefix(&[1, 0xFF, 0xFF]), Some(vec![2]));
    }

    #[test]
    fn next_prefix_returns_none_for_all_0xff() {
        assert_eq!(next_prefix(&[0xFF, 0xFF]), None);
    }

    #[test]
    fn next_prefix_returns_none_for_empty() {
        assert_eq!(next_prefix(&[]), None);
    }
}
