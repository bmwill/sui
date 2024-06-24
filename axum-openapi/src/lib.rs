use std::future::Future;

use axum::{
    extract::{Request, State},
    handler::Handler,
    http::Method,
    response::Response,
    routing::{MethodFilter, MethodRouter},
    Router,
};
use openapiv3::v3_1::{Operation, Parameter, RequestBody, Responses};

pub trait ApiEndpoint<S> {
    fn method(&self) -> axum::http::Method;
    fn path(&self) -> &'static str;
    fn hidden(&self) -> bool;
    fn deprecated(&self) -> bool;
    fn register_schemas(&self, generator: &mut SchemaGenerator);
    // fn parameters(&self, gen: &mut Registry) -> Vec<Parameter>;
    // fn request_body(&self) -> RequestBody;
    // fn response(&self) -> Responses;
    // summary description tags

    fn handler(&self) -> RouteHandler<S>;
}

pub struct RouteHandler<S> {
    method: axum::http::Method,
    handler: MethodRouter<S>,
}

impl<S: Clone> RouteHandler<S> {
    pub fn new<H, T>(method: axum::http::Method, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
        S: Send + Sync + 'static,
    {
        let handler = MethodRouter::new().on(method.clone().try_into().unwrap(), handler);

        Self { method, handler }
    }

    pub fn method(&self) -> &axum::http::Method {
        &self.method
    }
}

pub struct Api<'a, S> {
    endpoints: Vec<&'a dyn ApiEndpoint<S>>,
    info: openapi::Info,
}

pub use openapiv3::v3_1 as openapi;
use schemars::gen::SchemaGenerator;

impl<'a, S> Api<'a, S> {
    pub fn to_router(&self) -> axum::Router<S>
    where
        S: Clone + Send + Sync + 'static,
    {
        let mut router = Router::new();
        for endpoint in &self.endpoints {
            let handler = endpoint.handler();
            assert_eq!(handler.method(), endpoint.method());

            router = router.route(endpoint.path(), handler.handler);
        }

        router
    }

    pub fn openapi(&self) -> openapiv3::versioned::OpenApi {
        self.gen_openapi(self.info.clone())
    }

    /// Internal routine for constructing the OpenAPI definition describing this
    /// API in its JSON form.
    fn gen_openapi(&self, info: openapi::Info) -> openapiv3::versioned::OpenApi {
        let mut openapi = openapi::OpenApi::default();

        openapi.info = info;

        // Gather up the ad hoc tags from endpoints
        // let endpoint_tags = (&self.router)
        //     .into_iter()
        //     .flat_map(|(_, _, endpoint)| {
        //         endpoint
        //             .tags
        //             .iter()
        //             .filter(|tag| !self.tag_config.tag_definitions.contains_key(*tag))
        //     })
        //     .cloned()
        //     .collect::<HashSet<_>>()
        //     .into_iter()
        //     .map(|tag| openapiv3::Tag {
        //         name: tag,
        //         ..Default::default()
        //     });

        // // Bundle those with the explicit tags provided by the consumer
        // openapi.tags = self
        //     .tag_config
        //     .tag_definitions
        //     .iter()
        //     .map(|(name, details)| openapiv3::Tag {
        //         name: name.clone(),
        //         description: details.description.clone(),
        //         external_docs: details.external_docs.as_ref().map(|e| {
        //             openapiv3::ExternalDocumentation {
        //                 description: e.description.clone(),
        //                 url: e.url.clone(),
        //                 ..Default::default()
        //             }
        //         }),
        //         ..Default::default()
        //     })
        //     .chain(endpoint_tags)
        //     .collect();

        // // Sort the tags for stability
        // openapi.tags.sort_by(|a, b| a.name.cmp(&b.name));

        let settings = schemars::gen::SchemaSettings::openapi3();
        let mut generator = schemars::gen::SchemaGenerator::new(settings);
        // let mut definitions = indexmap::IndexMap::<String, schemars::schema::Schema>::new();

        // let mut paths = openapiv3::v3_1::Paths::default();

        for endpoint in &self.endpoints {
            // Skip hidden endpoints
            if endpoint.hidden() {
                continue;
            }

            //XXX This is a temporary, simplified generation of an OpenApi sepec which only collect
            // type schemas for requests/responses

            endpoint.register_schemas(&mut generator);

            //TODO finish actual generation of openapi spec
            // let path = paths
            //     .paths
            //     .entry(endpoint.path().to_owned())
            //     .or_insert(openapi::ReferenceOr::Item(openapi::PathItem::default()));

            // let pathitem = match path {
            //     openapi::ReferenceOr::Item(ref mut item) => item,
            //     _ => panic!("reference not expected"),
            // };

            // let method_ref = match endpoint.method() {
            //     Method::DELETE => &mut pathitem.delete,
            //     Method::GET => &mut pathitem.get,
            //     Method::HEAD => &mut pathitem.head,
            //     Method::OPTIONS => &mut pathitem.options,
            //     Method::PATCH => &mut pathitem.patch,
            //     Method::POST => &mut pathitem.post,
            //     Method::PUT => &mut pathitem.put,
            //     Method::TRACE => &mut pathitem.trace,
            //     other => panic!("unexpected method `{}`", other),
            // };
            // let mut operation = openapi::Operation::default();
            // operation.operation_id = Some(endpoint.operation_id.clone());
            // operation.summary = endpoint.summary.clone();
            // operation.description = endpoint.description.clone();
            // operation.tags = endpoint.tags.clone();
            // operation.deprecated = endpoint.deprecated;
        }

        //XXX This is copy-pasted from dropshot as inspiration
        // for (path, method, endpoint) in &self.router {
        //     if !endpoint.visible {
        //         continue;
        //     }
        //     let path = openapi
        //         .paths
        //         .paths
        //         .entry(path)
        //         .or_insert(openapiv3::ReferenceOr::Item(openapiv3::PathItem::default()));

        //     let pathitem = match path {
        //         openapiv3::ReferenceOr::Item(ref mut item) => item,
        //         _ => panic!("reference not expected"),
        //     };

        //     let method_ref = match &method[..] {
        //         "GET" => &mut pathitem.get,
        //         "PUT" => &mut pathitem.put,
        //         "POST" => &mut pathitem.post,
        //         "DELETE" => &mut pathitem.delete,
        //         "OPTIONS" => &mut pathitem.options,
        //         "HEAD" => &mut pathitem.head,
        //         "PATCH" => &mut pathitem.patch,
        //         "TRACE" => &mut pathitem.trace,
        //         other => panic!("unexpected method `{}`", other),
        //     };
        //     let mut operation = openapiv3::Operation::default();
        //     operation.operation_id = Some(endpoint.operation_id.clone());
        //     operation.summary = endpoint.summary.clone();
        //     operation.description = endpoint.description.clone();
        //     operation.tags = endpoint.tags.clone();
        //     operation.deprecated = endpoint.deprecated;

        //     operation.parameters = endpoint
        //         .parameters
        //         .iter()
        //         .filter_map(|param| {
        //             let (name, location) = match &param.metadata {
        //                 ApiEndpointParameterMetadata::Body(_) => return None,
        //                 ApiEndpointParameterMetadata::Path(name) => {
        //                     (name, ApiEndpointParameterLocation::Path)
        //                 }
        //                 ApiEndpointParameterMetadata::Query(name) => {
        //                     (name, ApiEndpointParameterLocation::Query)
        //                 }
        //             };

        //             let schema = match &param.schema {
        //                 ApiSchemaGenerator::Static {
        //                     schema,
        //                     dependencies,
        //                 } => {
        //                     definitions.extend(dependencies.clone());
        //                     j2oas_schema(None, schema)
        //                 }
        //                 _ => {
        //                     unimplemented!("this may happen for complex types")
        //                 }
        //             };

        //             let parameter_data = openapiv3::ParameterData {
        //                 name: name.clone(),
        //                 description: param.description.clone(),
        //                 required: param.required,
        //                 deprecated: None,
        //                 format: openapiv3::ParameterSchemaOrContent::Schema(schema),
        //                 example: None,
        //                 examples: indexmap::IndexMap::new(),
        //                 extensions: indexmap::IndexMap::new(),
        //                 explode: None,
        //             };
        //             match location {
        //                 ApiEndpointParameterLocation::Query => {
        //                     Some(openapiv3::ReferenceOr::Item(openapiv3::Parameter::Query {
        //                         parameter_data,
        //                         allow_reserved: false,
        //                         style: openapiv3::QueryStyle::Form,
        //                         allow_empty_value: None,
        //                     }))
        //                 }
        //                 ApiEndpointParameterLocation::Path => {
        //                     Some(openapiv3::ReferenceOr::Item(openapiv3::Parameter::Path {
        //                         parameter_data,
        //                         style: openapiv3::PathStyle::Simple,
        //                     }))
        //                 }
        //             }
        //         })
        //         .collect::<Vec<_>>();

        //     operation.request_body = endpoint
        //         .parameters
        //         .iter()
        //         .filter_map(|param| {
        //             let mime_type = match &param.metadata {
        //                 ApiEndpointParameterMetadata::Body(ct) => ct.mime_type(),
        //                 _ => return None,
        //             };

        //             let (name, js) = match &param.schema {
        //                 ApiSchemaGenerator::Gen { name, schema } => {
        //                     (Some(name()), schema(&mut generator))
        //                 }
        //                 ApiSchemaGenerator::Static {
        //                     schema,
        //                     dependencies,
        //                 } => {
        //                     definitions.extend(dependencies.clone());
        //                     (None, schema.as_ref().clone())
        //                 }
        //             };
        //             let schema = j2oas_schema(name.as_ref(), &js);

        //             let mut content = indexmap::IndexMap::new();
        //             content.insert(
        //                 mime_type.to_string(),
        //                 openapiv3::MediaType {
        //                     schema: Some(schema),
        //                     ..Default::default()
        //                 },
        //             );

        //             Some(openapiv3::ReferenceOr::Item(openapiv3::RequestBody {
        //                 content,
        //                 required: true,
        //                 ..Default::default()
        //             }))
        //         })
        //         .next();

        //     match &endpoint.extension_mode {
        //         ExtensionMode::None => {}
        //         ExtensionMode::Paginated(first_page_schema) => {
        //             operation.extensions.insert(
        //                 crate::pagination::PAGINATION_EXTENSION.to_string(),
        //                 first_page_schema.clone(),
        //             );
        //         }
        //         ExtensionMode::Websocket => {
        //             operation.extensions.insert(
        //                 crate::websocket::WEBSOCKET_EXTENSION.to_string(),
        //                 serde_json::json!({}),
        //             );
        //         }
        //     }

        //     let response = if let Some(schema) = &endpoint.response.schema {
        //         let (name, js) = match schema {
        //             ApiSchemaGenerator::Gen { name, schema } => {
        //                 (Some(name()), schema(&mut generator))
        //             }
        //             ApiSchemaGenerator::Static {
        //                 schema,
        //                 dependencies,
        //             } => {
        //                 definitions.extend(dependencies.clone());
        //                 (None, schema.as_ref().clone())
        //             }
        //         };
        //         let mut content = indexmap::IndexMap::new();
        //         if !is_empty(&js) {
        //             content.insert(
        //                 CONTENT_TYPE_JSON.to_string(),
        //                 openapiv3::MediaType {
        //                     schema: Some(j2oas_schema(name.as_ref(), &js)),
        //                     ..Default::default()
        //                 },
        //             );
        //         }

        //         let headers = endpoint
        //             .response
        //             .headers
        //             .iter()
        //             .map(|header| {
        //                 let schema = match &header.schema {
        //                     ApiSchemaGenerator::Static {
        //                         schema,
        //                         dependencies,
        //                     } => {
        //                         definitions.extend(dependencies.clone());
        //                         j2oas_schema(None, schema)
        //                     }
        //                     _ => {
        //                         unimplemented!("this may happen for complex types")
        //                     }
        //                 };

        //                 (
        //                     header.name.clone(),
        //                     openapiv3::ReferenceOr::Item(openapiv3::Header {
        //                         description: header.description.clone(),
        //                         style: openapiv3::HeaderStyle::Simple,
        //                         required: header.required,
        //                         deprecated: None,
        //                         format: openapiv3::ParameterSchemaOrContent::Schema(schema),
        //                         example: None,
        //                         examples: indexmap::IndexMap::new(),
        //                         extensions: indexmap::IndexMap::new(),
        //                     }),
        //                 )
        //             })
        //             .collect();

        //         let response = openapiv3::Response {
        //             description: if let Some(description) = &endpoint.response.description {
        //                 description.clone()
        //             } else {
        //                 // TODO: perhaps we should require even free-form
        //                 // responses to have a description since it's required
        //                 // by OpenAPI.
        //                 "".to_string()
        //             },
        //             content,
        //             headers,
        //             ..Default::default()
        //         };
        //         response
        //     } else {
        //         // If no schema was specified, the response is hand-rolled. In
        //         // this case we'll fall back to the default response type which
        //         // we assume to be inclusive of errors. The media type and
        //         // and schema will similarly be maximally permissive.
        //         let mut content = indexmap::IndexMap::new();
        //         content.insert(
        //                 "*/*".to_string(),
        //                 openapiv3::MediaType {
        //                     schema: Some(openapiv3::ReferenceOr::Item(openapiv3::Schema {
        //                         schema_data: openapiv3::SchemaData::default(),
        //                         schema_kind: openapiv3::SchemaKind::Any(
        //                             openapiv3::AnySchema::default(),
        //                         ),
        //                     })),
        //                     ..Default::default()
        //                 },
        //             );
        //         openapiv3::Response {
        //             // TODO: perhaps we should require even free-form
        //             // responses to have a description since it's required
        //             // by OpenAPI.
        //             description: "".to_string(),
        //             content,
        //             ..Default::default()
        //         }
        //     };

        //     if let Some(code) = &endpoint.response.success {
        //         operation.responses.responses.insert(
        //             openapiv3::StatusCode::Code(code.as_u16()),
        //             openapiv3::ReferenceOr::Item(response),
        //         );

        //         // 4xx and 5xx responses all use the same error information
        //         let err_ref = openapiv3::ReferenceOr::ref_("#/components/responses/Error");
        //         operation
        //             .responses
        //             .responses
        //             .insert(openapiv3::StatusCode::Range(4), err_ref.clone());
        //         operation
        //             .responses
        //             .responses
        //             .insert(openapiv3::StatusCode::Range(5), err_ref);
        //     } else {
        //         operation.responses.default = Some(openapiv3::ReferenceOr::Item(response))
        //     }

        //     // Drop in the operation.
        //     method_ref.replace(operation);
        // }

        // let components = &mut openapi
        //     .components
        //     .get_or_insert_with(openapiv3::Components::default);

        // // All endpoints share an error response
        // let responses = &mut components.responses;
        // let mut content = indexmap::IndexMap::new();
        // content.insert(
        //     CONTENT_TYPE_JSON.to_string(),
        //     openapiv3::MediaType {
        //         schema: Some(j2oas_schema(
        //             None,
        //             &generator.subschema_for::<HttpErrorResponseBody>(),
        //         )),
        //         ..Default::default()
        //     },
        // );

        // responses.insert(
        //     "Error".to_string(),
        //     openapiv3::ReferenceOr::Item(openapiv3::Response {
        //         description: "Error".to_string(),
        //         content,
        //         ..Default::default()
        //     }),
        // );

        // // Add the schemas for which we generated references.
        // let schemas = &mut components.schemas;

        // let root_schema = generator.into_root_schema_for::<()>();
        // root_schema.definitions.iter().for_each(|(key, schema)| {
        //     schemas.insert(key.clone(), j2oas_schema(None, schema));
        // });

        // definitions.into_iter().for_each(|(key, schema)| {
        //     if !schemas.contains_key(&key) {
        //         schemas.insert(key, j2oas_schema(None, &schema));
        //     }
        // });

        openapiv3::versioned::OpenApi::Version31(openapi)
    }
}
