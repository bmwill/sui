// Copyright (c) Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use http::{request, response, HeaderMap};

mod body;
mod future;
mod layer;
mod service;

pub use self::{
    body::CallbackBody, body::ResponseBody, future::ResponseFuture, layer::CallbackLayer,
    service::Callback,
};

pub trait MakeCallbackHandler {
    type Handler: ResponseHandler;

    fn make_handler(&self, request: &request::Parts) -> Self::Handler;
}

pub trait ResponseHandler {
    fn on_response(&mut self, response: &response::Parts);
    fn on_error<E>(&mut self, error: &E)
    where
        E: std::fmt::Display + 'static;

    fn on_body_chunk<B>(&mut self, _chunk: &B)
    where
        B: bytes::Buf,
    {
        // do nothing
    }

    fn on_end_of_stream(&mut self, _trailers: Option<&HeaderMap>) {
        // do nothing
    }
}

pub trait BodyHandler {
    fn on_error<E>(&mut self, error: &E)
    where
        E: std::fmt::Display + 'static;

    fn on_body_chunk<B>(&mut self, _chunk: &B)
    where
        B: bytes::Buf;

    fn on_end_of_stream(&mut self, _trailers: Option<&HeaderMap>);
}

impl<T: ResponseHandler> BodyHandler for T {
    fn on_error<E>(&mut self, error: &E)
    where
        E: std::fmt::Display + 'static,
    {
        ResponseHandler::on_error(self, error)
    }

    fn on_body_chunk<B>(&mut self, chunk: &B)
    where
        B: bytes::Buf,
    {
        ResponseHandler::on_body_chunk(self, chunk)
    }

    fn on_end_of_stream(&mut self, trailers: Option<&HeaderMap>) {
        ResponseHandler::on_end_of_stream(self, trailers)
    }
}

impl BodyHandler for () {
    fn on_error<E>(&mut self, _error: &E)
    where
        E: std::fmt::Display + 'static,
    {
    }

    fn on_body_chunk<B>(&mut self, _chunk: &B)
    where
        B: bytes::Buf,
    {
    }

    fn on_end_of_stream(&mut self, _trailers: Option<&HeaderMap>) {}
}
