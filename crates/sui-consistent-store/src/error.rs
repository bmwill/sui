// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Error types for the crate.
//!
//! Error structs in this module wrap a boxed inner so they fit in a
//! single pointer on the success path of `Result<T, _>`. The boxed
//! payload is allocated only when an error actually occurs.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

/// Type-erased dynamic error used as the source on error chains.
type DynError = Box<dyn Error + Send + Sync + 'static>;

/// An error returned by [`Encode::encode_into`].
///
/// Carries a free-form message and an optional source error. The
/// struct is one pointer wide; the payload lives on the heap and is
/// allocated only when an error actually fires.
///
/// [`Encode::encode_into`]: crate::Encode::encode_into
///
/// # Examples
///
/// ```
/// use sui_consistent_store::error::EncodeError;
///
/// let e = EncodeError::msg("buffer too small");
/// assert_eq!(e.to_string(), "encode failed: buffer too small");
/// ```
#[derive(Debug)]
pub struct EncodeError(Box<EncodeErrorInner>);

#[derive(Debug)]
struct EncodeErrorInner {
    message: Cow<'static, str>,
    source: Option<DynError>,
}

/// An error returned by [`Decode::decode`].
///
/// Carries a free-form message and an optional source error. The
/// struct is one pointer wide; the payload lives on the heap and is
/// allocated only when an error actually fires.
///
/// [`Decode::decode`]: crate::Decode::decode
///
/// # Examples
///
/// ```
/// use sui_consistent_store::error::DecodeError;
///
/// let e = DecodeError::msg("expected 8 bytes, got 4");
/// assert_eq!(e.to_string(), "decode failed: expected 8 bytes, got 4");
/// ```
#[derive(Debug)]
pub struct DecodeError(Box<DecodeErrorInner>);

#[derive(Debug)]
struct DecodeErrorInner {
    message: Cow<'static, str>,
    source: Option<DynError>,
}

impl EncodeError {
    /// Construct an error from a message alone.
    pub fn msg(message: impl Into<Cow<'static, str>>) -> Self {
        Self(Box::new(EncodeErrorInner {
            message: message.into(),
            source: None,
        }))
    }

    /// Construct an error with a message and an underlying source.
    ///
    /// The source is exposed via [`std::error::Error::source`] so that
    /// callers walking the error chain can recover the original
    /// failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::io;
    ///
    /// use sui_consistent_store::error::EncodeError;
    ///
    /// let io_err = io::Error::other("disk full");
    /// let e = EncodeError::with_source("write failed", io_err);
    /// assert!(e.source().is_some());
    /// ```
    pub fn with_source(message: impl Into<Cow<'static, str>>, source: impl Into<DynError>) -> Self {
        Self(Box::new(EncodeErrorInner {
            message: message.into(),
            source: Some(source.into()),
        }))
    }
}

impl DecodeError {
    /// Construct an error from a message alone.
    pub fn msg(message: impl Into<Cow<'static, str>>) -> Self {
        Self(Box::new(DecodeErrorInner {
            message: message.into(),
            source: None,
        }))
    }

    /// Construct an error with a message and an underlying source.
    ///
    /// The source is exposed via [`std::error::Error::source`] so that
    /// callers walking the error chain can recover the original
    /// failure.
    pub fn with_source(message: impl Into<Cow<'static, str>>, source: impl Into<DynError>) -> Self {
        Self(Box::new(DecodeErrorInner {
            message: message.into(),
            source: Some(source.into()),
        }))
    }
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "encode failed: {}", self.0.message)
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "decode failed: {}", self.0.message)
    }
}

impl Error for EncodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0
            .source
            .as_deref()
            .map(|e| e as &(dyn Error + 'static))
    }
}

impl Error for DecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0
            .source
            .as_deref()
            .map(|e| e as &(dyn Error + 'static))
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    #[test]
    fn encode_error_size_is_one_pointer() {
        assert_eq!(
            std::mem::size_of::<EncodeError>(),
            std::mem::size_of::<usize>(),
        );
    }

    #[test]
    fn decode_error_size_is_one_pointer() {
        assert_eq!(
            std::mem::size_of::<DecodeError>(),
            std::mem::size_of::<usize>(),
        );
    }

    #[test]
    fn encode_error_display() {
        let e = EncodeError::msg("oops");
        assert_eq!(e.to_string(), "encode failed: oops");
    }

    #[test]
    fn decode_error_display() {
        let e = DecodeError::msg("nope");
        assert_eq!(e.to_string(), "decode failed: nope");
    }

    #[test]
    fn encode_error_source_chain() {
        let inner = io::Error::other("underlying");
        let e = EncodeError::with_source("wrapper", inner);
        let src = Error::source(&e).expect("source should be set");
        assert_eq!(src.to_string(), "underlying");
    }

    #[test]
    fn decode_error_source_chain() {
        let inner = io::Error::new(io::ErrorKind::InvalidData, "bad bytes");
        let e = DecodeError::with_source("wrapper", inner);
        let src = Error::source(&e).expect("source should be set");
        assert_eq!(src.to_string(), "bad bytes");
    }

    #[test]
    fn encode_error_no_source_when_msg_only() {
        let e = EncodeError::msg("alone");
        assert!(Error::source(&e).is_none());
    }

    #[test]
    fn decode_error_accepts_owned_or_static_message() {
        let owned = DecodeError::msg(String::from("owned"));
        let borrowed = DecodeError::msg("static");
        assert_eq!(owned.to_string(), "decode failed: owned");
        assert_eq!(borrowed.to_string(), "decode failed: static");
    }
}
